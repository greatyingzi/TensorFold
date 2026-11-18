//! Compressed Lightning Indexer: pool 4-token groups and pick the groups a query recalls.

// Scope: pooling, scoring, locality mask and recall selection; rotary phases come from the caller.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s => error.ExpectedF32Tensor,
    };
}

/// Pool one group's `ratio` slots with a learned gated softmax, then RMSNorm the result.
pub fn poolOne(gpa: std.mem.Allocator, bag: *const weights.Bag, layer: usize, slots: []const []const f32, out: []f32) !void {
    const prefix = try std.fmt.allocPrint(gpa, "layers.{d}.self_attn.compressed_indexer.", .{layer});
    defer gpa.free(prefix);
    const wv = try std.mem.concat(gpa, u8, &.{ prefix, "key_value_projection.weight" });
    defer gpa.free(wv);
    const wg = try std.mem.concat(gpa, u8, &.{ prefix, "key_gate_projection.weight" });
    defer gpa.free(wg);
    const norm = try std.mem.concat(gpa, u8, &.{ prefix, "key_norm.weight" });
    defer gpa.free(norm);
    const dim = out.len;
    for (slots) |s| if (s.len != dim) return error.BadIndexerShape;
    const value_w = try f32s(bag, wv);
    const gate_w = try f32s(bag, wg);
    if (value_w.len != dim * dim or gate_w.len != dim * dim) return error.BadIndexerShape;
    const val = try gpa.alloc(f32, dim);
    defer gpa.free(val);
    const gate = try gpa.alloc(f32, dim);
    defer gpa.free(gate);
    const weighted = try gpa.alloc(f32, dim);
    defer gpa.free(weighted);
    @memset(weighted, 0);
    var max_logit = -std.math.inf(f32);
    for (slots, 0..) |slot, i| {
        try ops.matvec(val, value_w, slot, dim, dim);
        try ops.matvec(gate, gate_w, slot, dim, dim);
        const ape = try std.mem.concat(gpa, u8, &.{ prefix, "intra_group_ape" });
        defer gpa.free(ape);
        const ape_w = try f32s(bag, ape);
        if (ape_w.len < (i + 1) * dim) return error.BadIndexerShape;
        for (gate, 0..) |*g, j| {
            g.* += ape_w[i * dim + j];
            if (g.* > max_logit) max_logit = g.*;
        }
        for (gate) |*g| g.* = @exp(g.* - max_logit);
        var sum: f64 = 0;
        for (gate) |g| sum += @as(f64, g);
        for (gate) |*g| g.* = @floatCast(@as(f64, g.*) / sum);
        for (val, gate, 0..) |v, weight, j| weighted[j] += weight * v;
    }
    try ops.rmsNorm(out, weighted, try f32s(bag, norm), 1e-5);
}

/// Scores [heads, groups] from already-rotated queries and group keys, with head weights.
pub fn score(queries: []const []const f32, keys: []const []const f32, head_weights: []const f32, scores: []f32) !void {
    const heads = queries.len;
    if (head_weights.len != heads or scores.len != heads * keys.len) return error.BadIndexerShape;
    for (queries, 0..) |q, h| {
        for (keys, 0..) |k, g| {
            if (q.len != k.len) return error.BadIndexerShape;
            var dot: f64 = 0;
            for (q, k) |a, b| dot += @as(f64, a) * @as(f64, b);
            scores[h * keys.len + g] = @floatCast(@max(0, dot) * @as(f64, head_weights[h]));
        }
    }
}

/// A query recalls its own group unconditionally plus the best completed groups before it.
pub fn select(scores: []f32, groups: usize, heads: usize, group_topk: usize, query_group: usize, dst: []usize) !void {
    if (query_group >= groups or groups == 0 or heads == 0 or scores.len != heads * groups) return error.BadIndexerShape;
    const complete = @min(group_topk - 1, groups);
    if (dst.len < complete + 1) return error.BadIndexerShape;
    const logits = scores[0 .. heads * groups];
    for (0..groups) |g| {
        if (g >= query_group) {
            for (0..heads) |h| logits[h * groups + g] = -std.math.inf(f32);
        }
    }
    var ranked = try std.heap.page_allocator.alloc(f32, groups);
    defer std.heap.page_allocator.free(ranked);
    @memset(ranked, 0);
    for (0..heads) |h| {
        for (0..groups) |g| ranked[g] += scores[h * groups + g];
    }
    const idx = try std.heap.page_allocator.alloc(usize, groups);
    defer std.heap.page_allocator.free(idx);
    for (0..groups) |g| idx[g] = g;
    std.mem.sort(usize, idx, ranked, struct {
        fn lessThan(v: []f32, a: usize, b: usize) bool {
            if (v[a] != v[b]) return v[a] > v[b];
            return a < b;
        }
    }.lessThan);
    for (0..complete) |i| dst[i] = idx[i];
    dst[complete] = query_group;
}

test "hy4 host: compressed indexer pools and recalls the query's own group" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try @import("config.zig").parse(gpa, @embedFile("config_synthetic_fixture.json"));
    const row = try gpa.alloc(f32, c.index_dim);
    defer gpa.free(row);
    for (row) |*v| v.* = 0.1;
    const slots = [_][]const f32{ row, row };
    const pooled = try gpa.alloc(f32, c.index_dim);
    defer gpa.free(pooled);
    try poolOne(gpa, &bag, 0, slots[0..c.index_ratio], pooled);
    for (pooled) |v| try std.testing.expect(std.math.isFinite(v));

    const groups: usize = 8;
    const heads: usize = c.index_heads;
    const sc = try gpa.alloc(f32, heads * groups);
    defer gpa.free(sc);
    @memset(sc, 0);
    const chosen = try gpa.alloc(usize, c.index_group_topk);
    defer gpa.free(chosen);
    try select(sc, groups, heads, c.index_group_topk, 5, chosen);
    try std.testing.expectEqual(@as(usize, 5), chosen[c.index_group_topk - 1]);
    for (chosen[0 .. c.index_group_topk - 1]) |g| try std.testing.expect(g < 5);
}
