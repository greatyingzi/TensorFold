//! Hy4 Flash top-k router and LatentMoE block, following the official transformer f32 semantics.
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

pub const Route = struct {
    indices: []usize,
    weights: []f32,
};

/// sigmoid logits, group-scored bias, masked top-k, then renormalize and scale.
pub fn route(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, layer: usize, x: []const f32) !Route {
    if (x.len != c.hidden or c.n_group == 0 or c.experts % c.n_group != 0) return error.BadRouterShape;
    const prefix = std.fmt.allocPrint(gpa, "layers.{d}.mlp.", .{layer}) catch return error.OutOfMemory;
    defer gpa.free(prefix);
    const gate_name = try std.mem.concat(gpa, u8, &.{ prefix, "gate.weight" });
    defer gpa.free(gate_name);
    const bias_name = try std.mem.concat(gpa, u8, &.{ prefix, "gate.e_score_correction_bias" });
    defer gpa.free(bias_name);
    const gate = try f32s(bag, gate_name);
    const bias = try f32s(bag, bias_name);
    if (gate.len != c.experts * c.hidden or bias.len != c.experts) return error.BadRouterShape;

    const logits = try gpa.alloc(f32, c.experts);
    defer gpa.free(logits);
    try ops.matvec(logits, gate, x, c.experts, c.hidden);
    const scores = try gpa.alloc(f32, c.experts);
    defer gpa.free(scores);
    for (logits, 0..) |v, i| scores[i] = ops.sigmoid(v);
    const choice = try gpa.alloc(f32, c.experts);
    defer gpa.free(choice);
    for (scores, 0..) |v, i| choice[i] = v + bias[i];

    const per_group = c.experts / c.n_group;
    const group_scores = try gpa.alloc(f32, c.n_group);
    defer gpa.free(group_scores);
    const pick = try gpa.alloc(usize, 2);
    defer gpa.free(pick);
    for (0..c.n_group) |g| {
        const slice = choice[g * per_group ..][0..per_group];
        try ops.topk(pick, slice, 2);
        group_scores[g] = slice[pick[0]] + slice[pick[1]];
    }
    const group_rank = try gpa.alloc(usize, c.n_group);
    defer gpa.free(group_rank);
    try ops.topk(group_rank, group_scores, c.topk_group);
    var allowed = try gpa.alloc(bool, c.n_group);
    defer gpa.free(allowed);
    @memset(allowed, false);
    for (group_rank[0..c.topk_group]) |g| allowed[g] = true;

    const masked = try gpa.alloc(f32, c.experts);
    defer gpa.free(masked);
    for (0..c.experts) |e| masked[e] = if (allowed[e / per_group]) choice[e] else -std.math.inf(f32);
    const indices = try gpa.alloc(usize, c.topk);
    try ops.topk(indices, masked, c.topk);
    const chosen = try gpa.alloc(f32, c.topk);
    var total: f64 = 0;
    for (indices, 0..) |e, i| {
        chosen[i] = scores[e];
        total += @as(f64, scores[e]);
    }
    if (c.norm_topk_prob) {
        const den = total + 1e-20;
        for (chosen) |*w| w.* = @floatCast(@as(f64, w.*) / den);
    }
    for (chosen) |*w| w.* *= c.routed_scale;
    return .{ .indices = indices, .weights = chosen };
}

pub fn routeDeinit(gpa: std.mem.Allocator, r: Route) void {
    gpa.free(r.indices);
    gpa.free(r.weights);
}

fn swiglu(gate: []f32, up: []f32, limit: f32) void {
    for (gate, up) |*g, u| {
        const gc = @min(g.*, limit);
        const uc = @max(-limit, @min(u, limit));
        g.* = gc * ops.sigmoid(gc) * uc;
    }
}

/// Routed experts in the latent space plus the full-width shared expert and sandwich norm.
pub fn latentMoe(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, layer: usize, x: []const f32, out: []f32) !void {
    const lat = c.latent;
    if (x.len != c.hidden or out.len != c.hidden or lat == 0) return error.BadMoeShape;
    const prefix = try std.fmt.allocPrint(gpa, "layers.{d}.mlp.", .{layer});
    defer gpa.free(prefix);
    const name = struct {
        fn get(a: std.mem.Allocator, p: []const u8, suffix: []const u8) ![]u8 {
            return std.mem.concat(a, u8, &.{ p, suffix });
        }
    }.get;

    const r = try route(gpa, c, bag, layer, x);
    defer routeDeinit(gpa, r);
    const down_name = try name(gpa, prefix, "routed_expert_down_proj.weight");
    defer gpa.free(down_name);
    const latent_in = try gpa.alloc(f32, lat);
    defer gpa.free(latent_in);
    try ops.matvec(latent_in, try f32s(bag, down_name), x, lat, c.hidden);

    const gu_name = try name(gpa, prefix, "experts.gate_up_proj");
    defer gpa.free(gu_name);
    const dn_name = try name(gpa, prefix, "experts.down_proj");
    defer gpa.free(dn_name);
    const gu = try f32s(bag, gu_name);
    const dn = try f32s(bag, dn_name);
    const gu_rows = 2 * lat;
    if (gu.len != c.experts * gu_rows * lat or dn.len != c.experts * lat * lat) return error.BadMoeShape;
    const act = try gpa.alloc(f32, gu_rows);
    defer gpa.free(act);
    const expert_out = try gpa.alloc(f32, lat);
    defer gpa.free(expert_out);
    const acc = try gpa.alloc(f32, lat);
    defer gpa.free(acc);
    @memset(acc, 0);
    for (r.indices, r.weights) |e, w| {
        try ops.matvec(act, gu[e * gu_rows * lat ..][0 .. gu_rows * lat], latent_in, gu_rows, lat);
        swiglu(act[0..lat], act[lat..], c.swiglu_limit);
        try ops.matvec(expert_out, dn[e * lat * lat ..][0 .. lat * lat], act[0..lat], lat, lat);
        for (acc, expert_out) |*a, v| a.* += @as(f32, w) * v;
    }

    const up_name = try name(gpa, prefix, "routed_expert_up_proj.weight");
    defer gpa.free(up_name);
    const routed = try gpa.alloc(f32, c.hidden);
    defer gpa.free(routed);
    try ops.matvec(routed, try f32s(bag, up_name), acc, c.hidden, lat);

    const sg = try name(gpa, prefix, "shared_experts.gate_proj.weight");
    defer gpa.free(sg);
    const su = try name(gpa, prefix, "shared_experts.up_proj.weight");
    defer gpa.free(su);
    const sd = try name(gpa, prefix, "shared_experts.down_proj.weight");
    defer gpa.free(sd);
    const sgv = try gpa.alloc(f32, lat);
    defer gpa.free(sgv);
    const suv = try gpa.alloc(f32, lat);
    defer gpa.free(suv);
    try ops.matvec(sgv, try f32s(bag, sg), x, lat, c.hidden);
    try ops.matvec(suv, try f32s(bag, su), x, lat, c.hidden);
    swiglu(sgv, suv, c.swiglu_limit);
    const shared = try gpa.alloc(f32, c.hidden);
    defer gpa.free(shared);
    try ops.matvec(shared, try f32s(bag, sd), sgv, c.hidden, lat);

    for (routed, shared) |*a, v| a.* += v;
    const norm_name = try name(gpa, prefix, "combine_output_norm.weight");
    defer gpa.free(norm_name);
    try ops.rmsNorm(out, routed, try f32s(bag, norm_name), c.eps);
}

test "hy4 host: the released router config selects 16 of 512 experts" {
    const c = try @import("config.zig").parse(std.testing.allocator, @embedFile("config_fixture.json"));
    try std.testing.expectEqual(@as(u32, 1), c.n_group);
    try std.testing.expectEqual(@as(u32, 1), c.topk_group);
    try std.testing.expectEqual(@as(u32, 16), c.topk);
    try std.testing.expect(c.norm_topk_prob);
    try std.testing.expectEqual(@as(f32, 10), c.swiglu_limit);
}

test "hy4 host: synthetic LatentMoE runs and stays finite" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try @import("config.zig").parse(gpa, @embedFile("config_synthetic_fixture.json"));
    const x = try gpa.alloc(f32, c.hidden);
    defer gpa.free(x);
    @memset(x, 0.1);
    const out = try gpa.alloc(f32, c.hidden);
    defer gpa.free(out);
    try latentMoe(gpa, &c, &bag, 1, x, out);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}
