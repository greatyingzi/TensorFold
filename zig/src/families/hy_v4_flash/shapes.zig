//! Shapes implied by config for every backbone tensor the engine will read.
const std = @import("std");
const cfg = @import("config.zig");
const graph = @import("graph.zig");
const st = @import("../../core/safetensors.zig");

pub const Shape = struct {
    rank: u8,
    dtype: st.DType = .bf16,
    a: usize = 0,
    b: usize = 0,
    c: usize = 0,
    d: usize = 0,

    pub fn is(self: Shape, rank: u8, shape: []const usize) bool {
        if (rank != self.rank or rank > 4) return false;
        const want = [_]usize{ self.a, self.b, self.c, self.d };
        return std.mem.eql(usize, want[0..rank], shape[0..rank]);
    }
};

fn v1(a: usize) Shape {
    return .{ .rank = 1, .a = a };
}
fn i32v1(a: usize) Shape {
    return .{ .rank = 1, .a = a, .dtype = .i32 };
}
fn f32v1(a: usize) Shape {
    return .{ .rank = 1, .a = a, .dtype = .f32 };
}
fn f32v2(a: usize, b: usize) Shape {
    return .{ .rank = 2, .a = a, .b = b, .dtype = .f32 };
}
fn v2(a: usize, b: usize) Shape {
    return .{ .rank = 2, .a = a, .b = b };
}
fn v3(a: usize, b: usize, c: usize) Shape {
    return .{ .rank = 3, .a = a, .b = b, .c = c };
}

fn v4(a: usize, b: usize, c: usize, d: usize) Shape {
    return .{ .rank = 4, .a = a, .b = b, .c = c, .d = d };
}

fn layerOf(name: []const u8) ?usize {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(usize, rest[0..dot], 10) catch null;
}

fn tailOf(name: []const u8, layer: usize) []const u8 {
    var buf: [32]u8 = undefined;
    const prefix = std.fmt.bufPrint(&buf, "model.layers.{d}.", .{layer}) catch return "";
    return name[prefix.len..];
}

fn primes(c: *const cfg.Config, layer: usize) usize {
    for (c.engram.layers[0..c.engram.count]) |e| {
        if (e.id - 1 == layer) {
            var n: usize = 0;
            for (e.primes) |p| n += p;
            return n;
        }
    }
    return 0;
}

/// Null means the tensor belongs to a subsystem without a shape contract yet.
pub fn expected(c: *const cfg.Config, name: []const u8, lookup_len: usize) ?Shape {
    if (std.mem.eql(u8, name, "model.embed_tokens.weight") or std.mem.eql(u8, name, "lm_head.weight")) return v2(c.vocab, c.hidden);
    if (std.mem.eql(u8, name, "model.norm.weight")) return v1(c.hidden);
    if (std.mem.startsWith(u8, name, "model.hc_head.hc_head_fn")) return f32v2(c.hc_mult, c.hc_mult * c.hidden);
    if (std.mem.startsWith(u8, name, "model.hc_head.hc_head_base")) return f32v1(c.hc_mult);
    if (std.mem.startsWith(u8, name, "model.hc_head.hc_head_scale")) return f32v1(1);

    const dp = "model.dfly_layers.";
    if (std.mem.startsWith(u8, name, dp)) {
        const rest = name[dp.len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.');
        if (dot) |d| {
            const idx = std.fmt.parseInt(usize, rest[0..d], 10) catch {
        if (std.mem.eql(u8, rest, "final_norm.weight") or std.mem.eql(u8, rest, "hidden_correction.embed_norm.weight") or std.mem.eql(u8, rest, "hidden_correction.hidden_norm.weight") or std.mem.eql(u8, rest, "mtp_fusion.enorm.weight") or std.mem.eql(u8, rest, "mtp_fusion.hnorm.weight") or std.mem.eql(u8, rest, "mtp_fusion.gate")) return v1(c.hidden);
        if (std.mem.eql(u8, rest, "hidden_correction.gate_proj.weight") or std.mem.eql(u8, rest, "hidden_correction.up_proj.weight")) return v2(c.hidden, 2 * c.hidden);
        if (std.mem.eql(u8, rest, "hidden_correction.down_proj.weight")) return v2(c.hidden, c.hidden);
        if (std.mem.eql(u8, rest, "mtp_fusion.eh_proj.weight")) return v2(c.hidden, 2 * c.hidden);
        if (std.mem.eql(u8, rest, "confidence_head.proj.bias")) return v1(1);
        if (std.mem.eql(u8, rest, "confidence_head.proj.weight")) return v2(1, c.hidden);
                return null;
            };
            if (idx >= c.dfly.layers) return null;
            const ds = rest[d + 1 ..];
            if (std.mem.eql(u8, ds, "input_layernorm.weight") or std.mem.eql(u8, ds, "post_attention_layernorm.weight")) return v1(c.hidden);
            if (std.mem.eql(u8, ds, "mlp.gate_proj.weight") or std.mem.eql(u8, ds, "mlp.up_proj.weight")) return v2(3 * c.hidden, c.hidden);
            if (std.mem.eql(u8, ds, "mlp.down_proj.weight")) return v2(c.hidden, 3 * c.hidden);
            if (std.mem.eql(u8, ds, "self_attn.q_a_proj.weight")) return v2(c.q_lora, c.hidden);
            if (std.mem.eql(u8, ds, "self_attn.q_a_layernorm.weight")) return v1(c.q_lora);
            if (std.mem.eql(u8, ds, "self_attn.q_b_proj.weight")) return v2(c.swa_heads * (c.swa_qk_nope + c.swa_qk_rope), c.q_lora);
            if (std.mem.eql(u8, ds, "self_attn.kv_a_proj_with_mqa.weight")) return v2(c.kv_lora + c.qk_rope, c.hidden);
            if (std.mem.eql(u8, ds, "self_attn.kv_a_layernorm.weight")) return v1(c.kv_lora);
            if (std.mem.eql(u8, ds, "self_attn.kv_b_proj.weight")) return v2(c.swa_heads * (c.swa_qk_nope + c.swa_v_dim), c.kv_lora);
            if (std.mem.eql(u8, ds, "self_attn.o_proj.weight")) return v2(c.hidden, c.swa_heads * c.swa_v_dim);
            if (std.mem.eql(u8, ds, "self_attn.post_proj_norm.weight")) return v1(c.hidden);
            if (std.mem.eql(u8, ds, "self_attn.learnable_sink_param")) return v1(c.swa_heads);
            if (std.mem.eql(u8, ds, "self_attn.linear_gate.weight")) return v2(c.swa_heads * c.swa_v_dim, c.hidden);
            return null;
        }
        return null;
    }
    const layer = layerOf(name) orelse return null;
    if (layer >= c.layers) return null;
    const s = tailOf(name, layer);
    const hc = c.hc_mult;
    if (std.mem.startsWith(u8, s, "hc_attn_layer.hc_pre.hc_fn") or std.mem.startsWith(u8, s, "hc_mlp_layer.hc_pre.hc_fn")) return f32v2(2 * hc, hc * c.hidden);
    if (std.mem.startsWith(u8, s, "hc_attn_layer.hc_pre.hc_base") or std.mem.startsWith(u8, s, "hc_mlp_layer.hc_pre.hc_base")) return f32v1(2 * hc);
    if (std.mem.startsWith(u8, s, "hc_attn_layer.hc_pre.hc_scale") or std.mem.startsWith(u8, s, "hc_mlp_layer.hc_pre.hc_scale")) return f32v1(2);
    if (std.mem.eql(u8, s, "input_layernorm.weight") or std.mem.eql(u8, s, "post_attention_layernorm.weight")) return v1(c.hidden);

    const kind = c.layer[layer].kind;
    const heads = if (kind == .sliding) c.swa_heads else c.heads;
    const nope = if (kind == .sliding) c.swa_qk_nope else c.qk_nope;
    const rope = if (kind == .sliding) c.swa_qk_rope else c.qk_rope;
    const vdim = if (kind == .sliding) c.swa_v_dim else c.v_dim;
    if (std.mem.eql(u8, s, "self_attn.q_a_proj.weight")) return v2(c.q_lora, c.hidden);
    if (std.mem.eql(u8, s, "self_attn.q_a_layernorm.weight")) return v1(c.q_lora);
    if (std.mem.eql(u8, s, "self_attn.q_b_proj.weight")) return v2(heads * (nope + rope), c.q_lora);
    if (std.mem.eql(u8, s, "self_attn.kv_a_proj_with_mqa.weight")) return v2(c.kv_lora + rope, c.hidden);
    if (std.mem.eql(u8, s, "self_attn.kv_a_layernorm.weight")) return v1(c.kv_lora);
    if (std.mem.eql(u8, s, "self_attn.kv_b_proj.weight")) return v2(heads * (nope + vdim), c.kv_lora);
    if (std.mem.eql(u8, s, "self_attn.o_proj.weight")) return v2(c.hidden, heads * vdim);
    if (std.mem.eql(u8, s, "self_attn.post_proj_norm.weight")) return v1(c.hidden);
    if (std.mem.eql(u8, s, "self_attn.learnable_sink_param")) return f32v1(heads);
    if (std.mem.eql(u8, s, "self_attn.linear_gate.weight")) return v2(heads * vdim, c.hidden);

    const ip = "self_attn.compressed_indexer.";
    if (std.mem.startsWith(u8, s, ip)) {
        const x = s[ip.len..];
        if (std.mem.eql(u8, x, "head_weight_projection.weight")) return v2(c.index_heads, c.hidden);
        if (std.mem.eql(u8, x, "intra_group_ape")) return f32v2(c.index_ratio, c.index_dim);
        if (std.mem.eql(u8, x, "key_gate_projection.weight")) return f32v2(c.index_dim, c.index_dim);
        if (std.mem.eql(u8, x, "key_norm.weight")) return f32v1(c.index_dim);
        if (std.mem.eql(u8, x, "token_key_norm.weight")) return v1(c.index_dim);
        if (std.mem.eql(u8, x, "key_value_projection.weight")) return f32v2(c.index_dim, c.index_dim);
        if (std.mem.eql(u8, x, "token_key_projection.weight")) return v2(c.index_dim, c.hidden);
        if (std.mem.eql(u8, x, "query_projection.weight")) return v2(c.index_heads * c.index_dim, c.q_lora);
    }

    if (std.mem.eql(u8, s, "mlp.gate_proj.weight") or std.mem.eql(u8, s, "mlp.up_proj.weight")) return v2(c.intermediate, c.hidden);
    if (std.mem.eql(u8, s, "mlp.down_proj.weight")) return v2(c.hidden, c.intermediate);
    if (std.mem.eql(u8, s, "mlp.post_fc2_norm.weight")) return v1(c.hidden);
    if (std.mem.eql(u8, s, "mlp.gate.weight")) return v2(c.experts, c.hidden);
    if (std.mem.eql(u8, s, "mlp.gate.e_score_correction_bias")) return f32v1(c.experts);
    if (std.mem.eql(u8, s, "mlp.experts.gate_up_proj")) return v3(c.experts, c.hidden, c.latent);
    if (std.mem.eql(u8, s, "mlp.experts.down_proj")) return v3(c.experts, c.latent, c.latent);
    if (std.mem.eql(u8, s, "mlp.routed_expert_up_proj.weight")) return v2(c.hidden, c.latent);
    if (std.mem.eql(u8, s, "mlp.routed_expert_down_proj.weight")) return v2(c.latent, c.hidden);
    if (std.mem.eql(u8, s, "mlp.shared_experts.gate_proj.weight") or std.mem.eql(u8, s, "mlp.shared_experts.up_proj.weight")) return v2(c.latent, c.hidden);
    if (std.mem.eql(u8, s, "mlp.shared_experts.down_proj.weight")) return v2(c.hidden, c.latent);
    if (std.mem.eql(u8, s, "mlp.combine_output_norm.weight")) return v1(c.hidden);

    if (std.mem.startsWith(u8, s, "engram_layer.")) {
        const head = c.engram.embed / c.engram.heads;
        if (std.mem.eql(u8, s, "engram_layer.lookup_table")) return i32v1(lookup_len);
        if (std.mem.eql(u8, s, "engram_layer.multi_head_embedding.embedding.weight")) return v2(primes(c, layer), c.engram.embed / c.engram.heads);
        if (std.mem.eql(u8, s, "engram_layer.value_proj.weight")) return v2(c.hidden, (c.engram.ngram - 1) * c.engram.embed);
        if (std.mem.eql(u8, s, "engram_layer.value_proj.bias")) return v1(c.hidden);
        if (std.mem.eql(u8, s, "engram_layer.unified_pre_gamma")) return v1(c.hidden);
        if (std.mem.startsWith(u8, s, "engram_layer.unified_per_hc_scales.")) return v1(c.hidden);
        const ep = "engram_layer.";
        const rest = s[ep.len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const kind_name = rest[0..dot];
        const after = rest[dot + 1 ..];
        if (std.mem.eql(u8, kind_name, "key_projs")) {
            if (std.mem.endsWith(u8, after, "weight")) return v2(c.hidden, (c.engram.ngram - 1) * c.engram.embed);
            if (std.mem.endsWith(u8, after, "bias")) return v1(c.hidden);
        }
        if (std.mem.eql(u8, kind_name, "lookup_per_head_norms")) return v1(head);
        if (std.mem.eql(u8, kind_name, "norm_key") or std.mem.eql(u8, kind_name, "norm_query")) return v1(c.hidden);
    }
    if (std.mem.startsWith(u8, s, "vit.")) return visionShape(c, s);
    return null;
}

test "hy4 host: shape contract covers the synthetic backbone tensors" {
    const c = try @import("config.zig").parse(std.testing.allocator, @embedFile("config_synthetic_fixture.json"));
    const probes = [_][]const u8{
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.engram_layer.lookup_table",
        "model.hc_head.hc_head_fn",
        "model.embed_tokens.weight",
    };
    for (probes) |name| try std.testing.expect(expected(&c, name, c.vocab) != null);
    try std.testing.expect(expected(&c, "vit.layers.0.self_attn.q_proj.weight", c.vocab) == null);
}

fn visionShape(c: *const cfg.Config, name: []const u8) ?Shape {
    const h = c.vision.hidden;
    const v = c.vision;
    if (std.mem.eql(u8, name, "vit.embeddings.patch_embedding.weight")) return v4(h, v.channels, v.patch, v.patch);
    if (std.mem.eql(u8, name, "vit.embeddings.patch_embedding.bias")) return v1(h);
    if (std.mem.eql(u8, name, "vit.embeddings.position_embedding.weight")) return v2(v.positions, h);
    if (std.mem.startsWith(u8, name, "vit.layers.")) {
        const rest = name["vit.layers.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        _ = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
        const tail = rest[dot + 1 ..];
        if (std.mem.endsWith(u8, tail, ".weight") or std.mem.endsWith(u8, tail, ".bias")) {
            const suffix: []const u8 = if (std.mem.endsWith(u8, tail, ".weight")) ".weight" else ".bias";
            const stem = tail[0 .. tail.len - suffix.len];
            if (std.mem.eql(u8, stem, "input_layernorm") or std.mem.eql(u8, stem, "post_attention_layernorm")) return v1(h);
            if (std.mem.eql(u8, stem, "self_attn.q_proj") or std.mem.eql(u8, stem, "self_attn.k_proj") or std.mem.eql(u8, stem, "self_attn.v_proj") or std.mem.eql(u8, stem, "self_attn.o_proj")) return if (std.mem.endsWith(u8, tail, ".bias")) v1(h) else v2(h, h);
            if (std.mem.eql(u8, stem, "mlp.dense_h_to_4h")) return if (std.mem.endsWith(u8, tail, ".bias")) v1(v.intermediate) else v2(v.intermediate, h);
            if (std.mem.eql(u8, stem, "mlp.dense_4h_to_h")) return if (std.mem.endsWith(u8, tail, ".bias")) v1(h) else v2(h, v.intermediate);
        }
        return null;
    }
    if (std.mem.eql(u8, name, "vit.perceive.image_newline")) return v1(v.out);
    if (std.mem.eql(u8, name, "vit.perceive.norm.weight")) return v1(h);
    if (std.mem.eql(u8, name, "vit.perceive.proj.0.weight")) return v4(h * 4, h, v.spatial_merge, v.spatial_merge);
    if (std.mem.eql(u8, name, "vit.perceive.proj.0.bias")) return v1(h * 4);
    if (std.mem.eql(u8, name, "vit.perceive.proj.2.weight")) return v4(v.out, h * 4, 1, 1);
    if (std.mem.eql(u8, name, "vit.perceive.proj.2.bias")) return v1(v.out);
    return null;
}
