//! Full layer-stack forward for one token, composing iHC, Engram, MLA, rotary and MoE/FFN.

// Scope: single token with self-only attention and no indexer recall; multi-token attention is next.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");
const ihc = @import("ihc.zig");
const engram = @import("engram.zig");
const moe = @import("moe.zig");
const rotary = @import("rotary.zig");
const graph = @import("graph.zig");

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s => error.ExpectedF32Tensor,
    };
}

fn named(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, fmt, args);
}

fn denseFfn(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, layer: usize, x: []const f32, out: []f32) !void {
    const gate = try gpa.alloc(f32, c.intermediate);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, c.intermediate);
    defer gpa.free(up);
    const gn = try named(gpa, "layers.{d}.mlp.gate_proj.weight", .{layer});
    defer gpa.free(gn);
    const un = try named(gpa, "layers.{d}.mlp.up_proj.weight", .{layer});
    defer gpa.free(un);
    const dn = try named(gpa, "layers.{d}.mlp.down_proj.weight", .{layer});
    defer gpa.free(dn);
    try ops.matvec(gate, try f32s(bag, gn), x, c.intermediate, c.hidden);
    try ops.matvec(up, try f32s(bag, un), x, c.intermediate, c.hidden);
    for (gate, up) |*g, u| g.* = g.* * ops.sigmoid(g.*) * u;
    try ops.matvec(out, try f32s(bag, dn), gate, c.hidden, c.intermediate);
}

fn mlaSelf(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, layer: usize, kind: config.Kind, x: []const f32, out: []f32) !void {
    const heads = if (kind == .sliding) c.swa_heads else c.heads;
    const nope = if (kind == .sliding) c.swa_qk_nope else c.qk_nope;
    const rope = if (kind == .sliding) c.swa_qk_rope else c.qk_rope;
    const vdim = if (kind == .sliding) c.swa_v_dim else c.v_dim;
    const q_out = heads * (nope + rope);
    const kv_a_len = c.kv_lora + rope;
    const kv_b_len = heads * (nope + vdim);

    const qa = try gpa.alloc(f32, c.q_lora);
    defer gpa.free(qa);
    const qan = try named(gpa, "layers.{d}.self_attn.q_a_proj.weight", .{layer});
    defer gpa.free(qan);
    try ops.matvec(qa, try f32s(bag, qan), x, c.q_lora, c.hidden);
    const qn = try gpa.alloc(f32, c.q_lora);
    defer gpa.free(qn);
    const qnl = try named(gpa, "layers.{d}.self_attn.q_a_layernorm.weight", .{layer});
    defer gpa.free(qnl);
    try ops.rmsNorm(qn, qa, try f32s(bag, qnl), c.eps);
    const q = try gpa.alloc(f32, q_out);
    defer gpa.free(q);
    const qbn = try named(gpa, "layers.{d}.self_attn.q_b_proj.weight", .{layer});
    defer gpa.free(qbn);
    try ops.matvec(q, try f32s(bag, qbn), qn, q_out, c.q_lora);

    const cos = try gpa.alloc(f32, rope);
    defer gpa.free(cos);
    const sin = try gpa.alloc(f32, rope);
    defer gpa.free(sin);
    try rotary.cosSin(c, kind, 0, cos, sin);
    for (0..heads) |h| try rotary.apply(q[h * (nope + rope) ..][0 .. nope + rope], cos, sin, rope);

    const kva = try gpa.alloc(f32, kv_a_len);
    defer gpa.free(kva);
    const kan = try named(gpa, "layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{layer});
    defer gpa.free(kan);
    try ops.matvec(kva, try f32s(bag, kan), x, kv_a_len, c.hidden);
    const kvn = try gpa.alloc(f32, c.kv_lora);
    defer gpa.free(kvn);
    const kvnl = try named(gpa, "layers.{d}.self_attn.kv_a_layernorm.weight", .{layer});
    defer gpa.free(kvnl);
    try ops.rmsNorm(kvn, kva[0..c.kv_lora], try f32s(bag, kvnl), c.eps);
    const kvb = try gpa.alloc(f32, kv_b_len);
    defer gpa.free(kvb);
    const kbn = try named(gpa, "layers.{d}.self_attn.kv_b_proj.weight", .{layer});
    defer gpa.free(kbn);
    try ops.matvec(kvb, try f32s(bag, kbn), kvn, kv_b_len, c.kv_lora);

    const ctx = try gpa.alloc(f32, heads * vdim);
    defer gpa.free(ctx);
    for (0..heads) |h| @memcpy(ctx[h * vdim ..][0..vdim], kvb[h * (nope + vdim) + nope ..][0..vdim]);
    const on = try named(gpa, "layers.{d}.self_attn.o_proj.weight", .{layer});
    defer gpa.free(on);
    try ops.matvec(out, try f32s(bag, on), ctx, c.hidden, heads * vdim);
}

/// Run one token through every layer and return the final logits.
pub fn oneToken(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, token: usize, logits: []f32) !void {
    if (token >= c.vocab or logits.len != c.vocab) return error.BadForwardShape;
    try graph.validate(c);
    const h = c.hidden;
    const hc = c.hc_mult;
    const embed = try f32s(bag, "embed_tokens.weight");
    if (embed.len != c.vocab * h) return error.BadForwardShape;

    var streams: [16][]f32 = undefined;
    var next: [16][]f32 = undefined;
    for (0..hc) |s| {
        streams[s] = try gpa.alloc(f32, h);
        next[s] = try gpa.alloc(f32, h);
        @memcpy(streams[s], embed[token * h ..][0..h]);
    }
    defer for (streams[0..hc]) |v| gpa.free(v);
    defer for (next[0..hc]) |v| gpa.free(v);

    const mixed = try gpa.alloc(f32, h);
    defer gpa.free(mixed);
    const ffn = try gpa.alloc(f32, h);
    defer gpa.free(ffn);
    for (0..c.layers) |layer| {
        const plan = try graph.layerPlan(c, layer);
        if (plan.engram) |ei| {
            if (c.engram.layers[ei].id - 1 != layer) return error.BadForwardShape;
            const ngram = [3]u32{ @intCast(token), @intCast(token), @intCast(token) };
            var residual: [16][]f32 = undefined;
            for (0..hc) |s| residual[s] = next[s];
            try engram.inject(gpa, c, bag, layer, ngram, streams[0..hc], residual[0..hc]);
            for (0..hc) |s| {
                for (0..h) |i| streams[s][i] += next[s][i];
            }
        }

        const ag = try ihc.gates(gpa, c, bag, layer, "attn_hc", streams[0..hc]);
        defer ihc.gatesDeinit(gpa, ag);
        try ihc.collapse(ag, streams[0..hc], mixed);
        const an = try named(gpa, "layers.{d}.input_layernorm.weight", .{layer});
        defer gpa.free(an);
        try ops.rmsNorm(mixed, mixed, try f32s(bag, an), c.eps);
        try mlaSelf(gpa, c, bag, layer, plan.kind, mixed, ffn);
        for (0..hc) |s| {
            for (0..h) |i| next[s][i] = ag.post[s] * ffn[i] + streams[s][i];
        }
        for (0..hc) |s| @memcpy(streams[s], next[s]);

        const fg = try ihc.gates(gpa, c, bag, layer, "ffn_hc", streams[0..hc]);
        defer ihc.gatesDeinit(gpa, fg);
        try ihc.collapse(fg, streams[0..hc], mixed);
        const pn = try named(gpa, "layers.{d}.post_attention_layernorm.weight", .{layer});
        defer gpa.free(pn);
        try ops.rmsNorm(mixed, mixed, try f32s(bag, pn), c.eps);
        if (plan.moe) try moe.latentMoe(gpa, c, bag, layer, mixed, ffn) else try denseFfn(gpa, c, bag, layer, mixed, ffn);
        for (0..hc) |s| {
            for (0..h) |i| next[s][i] = fg.post[s] * ffn[i] + streams[s][i];
        }
        for (0..hc) |s| @memcpy(streams[s], next[s]);
    }

    const final = try gpa.alloc(f32, h);
    defer gpa.free(final);
    try ihc.head(gpa, c, bag, streams[0..hc], final);
    try ops.rmsNorm(final, final, try f32s(bag, "norm.weight"), c.eps);
    try ops.matvec(logits, try f32s(bag, "lm_head.weight"), final, c.vocab, h);
}

test "hy4 host: all synthetic layers run end to end" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try config.parse(gpa, @embedFile("config_synthetic_fixture.json"));
    const logits = try gpa.alloc(f32, c.vocab);
    defer gpa.free(logits);
    try oneToken(gpa, &c, &bag, 1, logits);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));
}


test "hy4 host: affine4 quantized experts run the same forward as bf16" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bf = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic-bf16", &meta);
    defer bf.deinit();
    var q4 = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic-q4", &meta);
    defer q4.deinit();
    const c = try config.parse(gpa, @embedFile("config_synthetic_q4_fixture.json"));
    var a = try @import("session.zig").Session.init(gpa, &c, &bf, 4);
    defer a.deinit();
    var b = try @import("session.zig").Session.init(gpa, &c, &q4, 4);
    defer b.deinit();
    const la = try a.step(1);
    const lb = try b.step(1);
    for (la) |v| try std.testing.expect(std.math.isFinite(v));
    for (lb) |v| try std.testing.expect(std.math.isFinite(v));
    try std.testing.expectEqual(try @import("spec.zig").pick(la), try @import("spec.zig").pick(lb));
}
