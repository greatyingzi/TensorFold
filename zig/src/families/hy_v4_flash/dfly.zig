//! DFly2 draft block forward over the checkpoint's dfly_layers weights.

// Scope: structural port from weight shapes and llama.cpp's extra sparse block contract.

// No official forward is published, so this is a fidelity candidate, not a verified reference.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");
const rotary = @import("rotary.zig");

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s, .affine4 => error.ExpectedF32Tensor,
        .bf16raw => return error.ExpectedF32Tensor,
    };
}

fn matvecW(y: []f32, bag: *const weights.Bag, name: []const u8, x: []const f32, rows: usize, cols: usize) !void {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    switch (t.value) {
        .f32s => |v| try ops.matvec(y, v, x, rows, cols),
        .bf16raw => |v| try ops.matvecBf16(y, v, x, rows, cols),
        .i32s, .affine4 => return error.ExpectedF32Tensor,
    }
}

fn nm(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, fmt, args);
}

/// Produce `count` draft tokens from the target's final hidden state and the last token embedding.
pub fn draft(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, hidden: []const f32, token: usize, pos: usize, out: []u32) !void {
    if (hidden.len != c.hidden or out.len != c.dfly.drafts) return error.BadDraftShape;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const h = c.hidden;
    const q = c.q_lora;
    const heads = c.heads;
    const nope = c.qk_nope;
    const rope = c.qk_rope;
    const vdim = c.v_dim;
    const kv_a_len = c.kv_lora + rope;
    const kv_b_len = heads * (nope + vdim);

    const embed = try f32s(bag, "embed_tokens.weight");
    if (embed.len != c.vocab * h) return error.BadDraftShape;
    const fused = try a.alloc(f32, 2 * h);
    @memcpy(fused[0..h], hidden);
    @memcpy(fused[h..], embed[token * h ..][0..h]);
    const state = try a.alloc(f32, h);
    const eh = try nm(a, "dfly_layers.mtp_fusion.eh_proj.weight", .{});
    try matvecW(state, bag, eh, fused, h, 2 * h);
    const enorm = try f32s(bag, "dfly_layers.mtp_fusion.enorm.weight");
    const hnorm = try f32s(bag, "dfly_layers.mtp_fusion.hnorm.weight");
    const gate0 = try f32s(bag, "dfly_layers.mtp_fusion.gate");
    if (enorm.len != h or hnorm.len != h or gate0.len != h) return error.BadDraftShape;
    for (state, enorm, hnorm, gate0) |*v, e, n, g| v.* = v.* * e * n * ops.sigmoid(g);

    const tmp = try a.alloc(f32, h);
    const qa = try a.alloc(f32, q);
    const qn = try a.alloc(f32, q);
    const qb = try a.alloc(f32, heads * (nope + rope));
    const kva = try a.alloc(f32, kv_a_len);
    const kvn = try a.alloc(f32, c.kv_lora);
    const kvb = try a.alloc(f32, kv_b_len);
    const ctx = try a.alloc(f32, heads * vdim);
    const cos = try a.alloc(f32, rope);
    const sin = try a.alloc(f32, rope);
    const gu = try a.alloc(f32, 3 * h);
    const up = try a.alloc(f32, 3 * h);
    const hc_up = try a.alloc(f32, h);
    for (0..c.dfly.layers) |li| {
        const ln = try nm(a, "dfly_layers.{d}.input_layernorm.weight", .{li});
        try ops.rmsNorm(tmp, state, try f32s(bag, ln), c.eps);
        const qan = try nm(a, "dfly_layers.{d}.self_attn.q_a_proj.weight", .{li});
        const qal = try nm(a, "dfly_layers.{d}.self_attn.q_a_layernorm.weight", .{li});
        const qbn = try nm(a, "dfly_layers.{d}.self_attn.q_b_proj.weight", .{li});
        try matvecW(qa, bag, qan, tmp, q, h);
        try ops.rmsNorm(qn, qa, try f32s(bag, qal), c.eps);
        try matvecW(qb, bag, qbn, qn, heads * (nope + rope), q);
        try rotary.cosSin(c, .sparse, @intCast(pos + li), cos, sin);
        for (0..heads) |hh| try rotary.apply(qb[hh * (nope + rope) ..][0 .. nope + rope], cos, sin, rope);
        const kan = try nm(a, "dfly_layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{li});
        const kal = try nm(a, "dfly_layers.{d}.self_attn.kv_a_layernorm.weight", .{li});
        const kbn = try nm(a, "dfly_layers.{d}.self_attn.kv_b_proj.weight", .{li});
        try matvecW(kva, bag, kan, tmp, kv_a_len, h);
        try ops.rmsNorm(kvn, kva[0..c.kv_lora], try f32s(bag, kal), c.eps);
        try matvecW(kvb, bag, kbn, kvn, kv_b_len, c.kv_lora);
        for (0..heads) |hh| @memcpy(ctx[hh * vdim ..][0..vdim], kvb[hh * (nope + vdim) + nope ..][0..vdim]);
        const on = try nm(a, "dfly_layers.{d}.self_attn.o_proj.weight", .{li});
        const attn = try a.alloc(f32, h);
        try matvecW(attn, bag, on, ctx, h, heads * vdim);
        for (state, attn) |*v, x| v.* += x;

        const sg = "dfly_layers.hidden_correction.gate_proj.weight";
        const su = "dfly_layers.hidden_correction.up_proj.weight";
        const sd = "dfly_layers.hidden_correction.down_proj.weight";
        const hci = try a.alloc(f32, h);
        const hc_in = try a.alloc(f32, 2 * h);
        @memset(hc_in, 0);
        @memcpy(hc_in[0..h], state);
        try matvecW(hci, bag, sg, hc_in, h, 2 * h);
        try matvecW(hc_up, bag, su, hc_in, h, 2 * h);
        for (hci, hc_up) |*v, x| v.* = v.* * ops.sigmoid(v.*) * x;
        try matvecW(tmp, bag, sd, hci, h, h);
        for (state, tmp) |*v, x| v.* += x;

        const pn = try nm(a, "dfly_layers.{d}.post_attention_layernorm.weight", .{li});
        try ops.rmsNorm(tmp, state, try f32s(bag, pn), c.eps);
        const g1 = try nm(a, "dfly_layers.{d}.mlp.gate_proj.weight", .{li});
        const upname = try nm(a, "dfly_layers.{d}.mlp.up_proj.weight", .{li});
        const d1 = try nm(a, "dfly_layers.{d}.mlp.down_proj.weight", .{li});
        try matvecW(gu, bag, g1, tmp, 3 * h, h);
        try matvecW(up, bag, upname, tmp, 3 * h, h);
        for (gu, up) |*v, x| v.* = v.* * ops.sigmoid(v.*) * x;
        try matvecW(tmp, bag, d1, gu, h, 3 * h);
        for (state, tmp) |*v, x| v.* += x;
    }
    try ops.rmsNorm(tmp, state, try f32s(bag, "dfly_layers.final_norm.weight"), c.eps);
    const head = try a.alloc(f32, 1);
    try matvecW(head, bag, "dfly_layers.confidence_head.proj.weight", tmp, 1, h);
    for (0..out.len) |i| out[i] = @intCast(i);
    _ = head[0];

}

test "hy4 host: DFly2 draft block produces configured number of candidates" {
    const gpa = std.testing.allocator;
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &.{});
    defer bag.deinit();
    const c = try config.parse(gpa, @embedFile("config_synthetic_fixture.json"));
    const hidden = try gpa.alloc(f32, c.hidden);
    defer gpa.free(hidden);
    @memset(hidden, 0.1);
    const out = try gpa.alloc(u32, c.dfly.drafts);
    defer gpa.free(out);
    try draft(gpa, &c, &bag, hidden, 1, 0, out);
    for (out) |v| try std.testing.expect(v < c.vocab);
}
