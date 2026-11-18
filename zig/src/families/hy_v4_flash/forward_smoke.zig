//! Single-token host forward smoke on the synthetic checkpoint.

// Scope: layer 0, self-only attention, no indexer, no Engram and no iHC streams.

// It proves tensor plumbing and shape arithmetic, not model fidelity.
const std = @import("std");
const config = @import("config.zig");
const graph = @import("graph.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s => error.ExpectedF32Tensor,
    };
}

fn silu(x: f32) f32 {
    return x * ops.sigmoid(x);
}

/// Fill `logits` with one greedy token's scores from the synthetic checkpoint.
pub fn oneToken(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, token: usize, logits: []f32) !void {
    if (token >= c.vocab or logits.len != c.vocab) return error.BadForwardShape;
    try graph.validate(c);
    const h = c.hidden;
    const embed = try f32s(bag, "embed_tokens.weight");
    if (embed.len != c.vocab * h) return error.BadForwardShape;

    const x = try gpa.alloc(f32, h);
    defer gpa.free(x);
    @memcpy(x, embed[token * h ..][0..h]);

    const plan = try graph.layerPlan(c, 0);
    const nope = c.qk_nope;
    const vdim = c.v_dim;
    const heads = c.heads;
    const q_out = heads * (2 * nope + c.qk_rope);
    const kv_a_len = c.kv_lora + c.kv_heads * c.qk_rope;
    const kv_b_len = heads * (nope + vdim);

    const xn = try gpa.alloc(f32, h);
    defer gpa.free(xn);
    try ops.rmsNorm(xn, x, try f32s(bag, "layers.0.input_layernorm.weight"), c.eps);

    const qa = try gpa.alloc(f32, c.q_lora);
    defer gpa.free(qa);
    try ops.matvec(qa, try f32s(bag, "layers.0.self_attn.q_a_proj.weight"), xn, c.q_lora, h);
    const qn = try gpa.alloc(f32, c.q_lora);
    defer gpa.free(qn);
    try ops.rmsNorm(qn, qa, try f32s(bag, "layers.0.self_attn.q_a_layernorm.weight"), c.eps);
    const q = try gpa.alloc(f32, q_out);
    defer gpa.free(q);
    try ops.matvec(q, try f32s(bag, "layers.0.self_attn.q_b_proj.weight"), qn, q_out, c.q_lora);

    const kva = try gpa.alloc(f32, kv_a_len);
    defer gpa.free(kva);
    try ops.matvec(kva, try f32s(bag, "layers.0.self_attn.kv_a_proj_with_mqa.weight"), xn, kv_a_len, h);
    const kvn = try gpa.alloc(f32, c.kv_lora);
    defer gpa.free(kvn);
    try ops.rmsNorm(kvn, kva[0..c.kv_lora], try f32s(bag, "layers.0.self_attn.kv_a_layernorm.weight"), c.eps);
    const kvb = try gpa.alloc(f32, kv_b_len);
    defer gpa.free(kvb);
    try ops.matvec(kvb, try f32s(bag, "layers.0.self_attn.kv_b_proj.weight"), kvn, kv_b_len, c.kv_lora);

    const ctx = try gpa.alloc(f32, heads * vdim);
    defer gpa.free(ctx);
    for (0..heads) |head| {
        const src = kvb[head * (nope + vdim) + nope ..][0..vdim];
        @memcpy(ctx[head * vdim ..][0..vdim], src);
    }
    const attn = try gpa.alloc(f32, h);
    defer gpa.free(attn);
    try ops.matvec(attn, try f32s(bag, "layers.0.self_attn.o_proj.weight"), ctx, h, heads * vdim);
    for (x, attn) |*dst, v| dst.* += v;

    if (plan.moe) return error.SparseLayerNotSupportedInSmoke;
    const xn2 = try gpa.alloc(f32, h);
    defer gpa.free(xn2);
    try ops.rmsNorm(xn2, x, try f32s(bag, "layers.0.post_attention_layernorm.weight"), c.eps);
    const gate = try gpa.alloc(f32, c.intermediate);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, c.intermediate);
    defer gpa.free(up);
    try ops.matvec(gate, try f32s(bag, "layers.0.mlp.gate_proj.weight"), xn2, c.intermediate, h);
    try ops.matvec(up, try f32s(bag, "layers.0.mlp.up_proj.weight"), xn2, c.intermediate, h);
    for (gate, up) |*g, u| g.* = silu(g.*) * u;
    const ffn = try gpa.alloc(f32, h);
    defer gpa.free(ffn);
    try ops.matvec(ffn, try f32s(bag, "layers.0.mlp.down_proj.weight"), gate, h, c.intermediate);
    for (x, ffn) |*dst, v| dst.* += v;

    const final = try gpa.alloc(f32, h);
    defer gpa.free(final);
    try ops.rmsNorm(final, x, try f32s(bag, "norm.weight"), c.eps);
    try ops.matvec(logits, try f32s(bag, "lm_head.weight"), final, c.vocab, h);
}

test "hy4 host: one synthetic token runs end to end through the smoke path" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try config.parse(gpa, @embedFile("config_synthetic_fixture.json"));
    const logits = try gpa.alloc(f32, c.vocab);
    defer gpa.free(logits);
    try oneToken(gpa, &c, &bag, 1, logits);
    for (logits) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
