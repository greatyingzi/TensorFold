//! Identity hyper-connections: four residual streams mixed by per-layer learned gates.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");

pub const Streams = []const []const f32;

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s, .affine4 => error.ExpectedF32Tensor,
    };
}

fn unweightedNorm(x: []const f32, eps: f32) f32 {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * @as(f64, v);
    return @floatCast(1.0 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + @as(f64, eps)));
}

pub const Gates = struct {
    pre: []f32,
    post: []f32,
};

/// Per-layer gate computation: f32 mixes times an unweighted RMS of the flattened streams.
pub fn gates(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, layer: usize, kind: []const u8, streams: Streams) !Gates {
    const hc = c.hc_mult;
    if (streams.len != hc or c.hidden == 0) return error.BadHcShape;
    const flat = try gpa.alloc(f32, hc * c.hidden);
    defer gpa.free(flat);
    for (streams, 0..) |s, i| {
        if (s.len != c.hidden) return error.BadHcShape;
        @memcpy(flat[i * c.hidden ..][0..c.hidden], s);
    }
    const prefix = try std.fmt.allocPrint(gpa, "layers.{d}.{s}.", .{ layer, kind });
    defer gpa.free(prefix);
    const fn_name = try std.mem.concat(gpa, u8, &.{ prefix, "fn" });
    defer gpa.free(fn_name);
    const base_name = try std.mem.concat(gpa, u8, &.{ prefix, "base" });
    defer gpa.free(base_name);
    const scale_name = try std.mem.concat(gpa, u8, &.{ prefix, "scale" });
    defer gpa.free(scale_name);
    const fn_w = try f32s(bag, fn_name);
    const base = try f32s(bag, base_name);
    const scale = try f32s(bag, scale_name);
    if (fn_w.len != 2 * hc * hc * c.hidden or base.len != 2 * hc or scale.len != 2) return error.BadHcShape;

    const mixes = try gpa.alloc(f32, 2 * hc);
    defer gpa.free(mixes);
    try ops.matvec(mixes, fn_w, flat, 2 * hc, hc * c.hidden);
    const norm = unweightedNorm(flat, c.hc_eps);
    const pre = try gpa.alloc(f32, hc);
    const post = try gpa.alloc(f32, hc);
    for (0..hc) |s| {
        const pre_logit = mixes[s] * norm;
        const post_logit = mixes[hc + s] * norm;
        pre[s] = ops.sigmoid(pre_logit * scale[0] + base[s]) + c.hc_eps;
        post[s] = c.hc_magnitude * ops.sigmoid(post_logit * scale[1] + base[hc + s]) + c.hc_eps;
    }
    return .{ .pre = pre, .post = post };
}

pub fn gatesDeinit(gpa: std.mem.Allocator, g: Gates) void {
    gpa.free(g.pre);
    gpa.free(g.post);
}

/// Collapse the pre-gated streams into one hidden vector before the mixer.
pub fn collapse(g: Gates, streams: Streams, out: []f32) !void {
    if (out.len == 0 or streams.len != g.pre.len) return error.BadHcShape;
    @memset(out, 0);
    for (streams, g.pre) |s, w| {
        if (s.len != out.len) return error.BadHcShape;
        for (s, out) |v, *o| o.* += w * v;
    }
}

/// The final stream collapse before the model's shared RMSNorm.
pub fn head(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, streams: Streams, out: []f32) !void {
    const hc = c.hc_mult;
    if (streams.len != hc or out.len != c.hidden) return error.BadHcShape;
    const flat = try gpa.alloc(f32, hc * c.hidden);
    defer gpa.free(flat);
    for (streams, 0..) |s, i| {
        if (s.len != c.hidden) return error.BadHcShape;
        @memcpy(flat[i * c.hidden ..][0..c.hidden], s);
    }
    const fn_w = try f32s(bag, "hc_head.hc_fn");
    const base = try f32s(bag, "hc_head.hc_base");
    const scale = try f32s(bag, "hc_head.hc_scale");
    if (fn_w.len != hc * hc * c.hidden or base.len != hc or scale.len != 1) return error.BadHcShape;
    const mixes = try gpa.alloc(f32, hc);
    defer gpa.free(mixes);
    try ops.matvec(mixes, fn_w, flat, hc, hc * c.hidden);
    const norm = unweightedNorm(flat, c.hc_eps);
    @memset(out, 0);
    for (streams, 0..) |s, i| {
        const pre = ops.sigmoid(mixes[i] * norm * scale[0] + base[i]) + c.hc_eps;
        for (s, out) |v, *o| o.* += pre * v;
    }
}

test "hy4 host: iHC gates and head collapse keep four streams finite" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try @import("config.zig").parse(gpa, @embedFile("config_synthetic_fixture.json"));
    const row = try gpa.alloc(f32, c.hidden);
    defer gpa.free(row);
    @memset(row, 1);
    var store: [4][]const f32 = undefined;
    for (&store) |*s| s.* = row;
    const g = try gates(gpa, &c, &bag, 0, "attn_hc", &store);
    defer gatesDeinit(gpa, g);
    try std.testing.expectEqual(@as(usize, 4), g.pre.len);
    for (g.pre) |v| try std.testing.expect(std.math.isFinite(v) and v >= 0);
    for (g.post) |v| try std.testing.expect(std.math.isFinite(v) and v >= 0);
    const mixed = try gpa.alloc(f32, c.hidden);
    defer gpa.free(mixed);
    try collapse(g, &store, mixed);
    const out = try gpa.alloc(f32, c.hidden);
    defer gpa.free(out);
    try head(gpa, &c, &bag, &store, out);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}
