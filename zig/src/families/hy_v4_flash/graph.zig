//! Hy4 Flash layer graph derived from config alone, so loader and planner agree without weights.
const std = @import("std");
const cfg = @import("config.zig");

pub const LayerPlan = struct {
    kind: cfg.Kind,
    moe: bool,
    /// Index into Config.engram.layers, or null for a plain layer.
    engram: ?usize = null,
    /// The compressed indexer only exists on sparse-attention layers.
    indexer: bool = false,
    /// Tensors the release stores for one backbone layer of this shape.
    tensors: u16 = 0,
};

/// Counts measured from the released index: dense 30, sparse MoE 36, SWA MoE 28, Engram +41.
pub fn layerPlan(c: *const cfg.Config, layer: usize) !LayerPlan {
    if (layer >= c.layers) return error.LayerOutOfBounds;
    const l = c.layer[layer];
    var plan = LayerPlan{
        .kind = l.kind,
        .moe = l.moe,
        .indexer = l.kind == .sparse,
        .tensors = if (l.moe) (if (l.kind == .sparse) 36 else 28) else 30,
    };
    for (c.engram.layers[0..c.engram.count], 0..) |e, i| {
        if (e.id - 1 == layer) {
            plan.engram = i;
            plan.tensors += 41;
            break;
        }
    }
    return plan;
}

pub fn expectedBackboneTensors(c: *const cfg.Config) !usize {
    var total: usize = 0;
    for (0..c.layers) |i| total += (try layerPlan(c, i)).tensors;
    return total;
}

pub const KvDims = struct {
    latent: u32,
    rope: u32,
    heads: u32,

    pub fn values(self: KvDims) usize {
        return @as(usize, self.latent) + @as(usize, self.rope);
    }
};

/// MLA stores one compressed latent plus its RoPE part per token, with kind-specific head counts.
pub fn kvDims(c: *const cfg.Config, kind: cfg.Kind) KvDims {
    return .{
        .latent = c.kv_lora,
        .rope = if (kind == .sliding) c.swa_qk_rope else c.qk_rope,
        .heads = if (kind == .sliding) c.swa_heads else c.heads,
    };
}

pub fn indexerGroups(c: *const cfg.Config) usize {
    return @as(usize, c.index_candidate_topk) / @as(usize, c.index_group_topk);
}

/// The shape decisions that must hold before any tensor is read.
pub fn validate(c: *const cfg.Config) error{UnsupportedHy4Geometry}!void {
    if (c.layers == 0 or c.hidden == 0 or c.heads == 0 or c.swa_heads == 0 or c.kv_heads == 0) return error.UnsupportedHy4Geometry;
    if (c.heads % c.kv_heads != 0 or c.swa_heads % c.kv_heads != 0) return error.UnsupportedHy4Geometry;
    if (c.qk_nope == 0 or c.qk_rope == 0 or c.v_dim == 0 or c.kv_lora == 0 or c.q_lora == 0) return error.UnsupportedHy4Geometry;
    if (c.moe_intermediate == 0 or c.latent == 0 or c.latent > c.moe_intermediate) return error.UnsupportedHy4Geometry;
    if (c.experts == 0 or c.topk == 0 or c.topk > c.experts or c.dense_layers == 0 or c.dense_layers >= c.layers) return error.UnsupportedHy4Geometry;
    if (c.window == 0 or c.index_ratio == 0 or c.index_group_topk == 0 or c.index_candidate_topk == 0) return error.UnsupportedHy4Geometry;
    if (c.index_ratio > c.window or c.index_group_topk > c.index_candidate_topk) return error.UnsupportedHy4Geometry;
    if (!c.allow_no_engram and c.engram.count == 0) return error.UnsupportedHy4Geometry;
    if (c.engram.heads == 0 or c.engram.embed == 0) return error.UnsupportedHy4Geometry;
    if (c.dfly.layers == 0 or c.dfly.drafts == 0 or c.dfly.target_count == 0) return error.UnsupportedHy4Geometry;
    for (0..c.layers) |i| {
        if (i >= c.dense_layers and !c.layer[i].moe) return error.UnsupportedHy4Geometry;
    }
}

test "hy4 host: the layer graph matches the released layer types" {
    const c = try @import("config.zig").parse(std.testing.allocator, @embedFile("config_fixture.json"));
    try validate(&c);
    try std.testing.expectEqual(@as(usize, 1430), try expectedBackboneTensors(&c));
    const dense = try layerPlan(&c, 0);
    try std.testing.expect(!dense.moe and dense.indexer and dense.engram == null);
    try std.testing.expectEqual(@as(u16, 30), dense.tensors);
    const eng_sparse = try layerPlan(&c, 1);
    try std.testing.expect(eng_sparse.moe and eng_sparse.indexer and eng_sparse.engram != null);
    try std.testing.expectEqual(@as(u16, 77), eng_sparse.tensors);
    const plain_sparse = try layerPlan(&c, 5);
    try std.testing.expectEqual(@as(u16, 36), plain_sparse.tensors);
    const eng_swa = try layerPlan(&c, 11);
    try std.testing.expectEqual(@as(u16, 69), eng_swa.tensors);
    const plain_swa = try layerPlan(&c, 2);
    try std.testing.expect(!plain_swa.indexer);
    try std.testing.expectEqual(@as(u16, 28), plain_swa.tensors);
    try std.testing.expectEqual(@as(usize, 4), indexerGroups(&c));
}

test "hy4 host: MLA dimensions are per attention kind" {
    const c = try @import("config.zig").parse(std.testing.allocator, @embedFile("config_fixture.json"));
    const sparse = kvDims(&c, .sparse);
    const swa = kvDims(&c, .sliding);
    try std.testing.expectEqual(@as(u32, 512), sparse.latent);
    try std.testing.expectEqual(@as(u32, 64), sparse.rope);
    try std.testing.expectEqual(@as(u32, 32), sparse.heads);
    try std.testing.expectEqual(@as(u32, 64), swa.heads);
    try std.testing.expectEqual(@as(usize, 576), sparse.values());
    try std.testing.expectError(error.LayerOutOfBounds, layerPlan(&c, c.layers));
}


test "hy4 host: the synthetic layer graph matches the generated fixture" {
    const c = try @import("config.zig").parse(std.testing.allocator, @embedFile("config_synthetic_fixture.json"));
    try validate(&c);
    try std.testing.expectEqual(@as(usize, 171), try expectedBackboneTensors(&c));
}
