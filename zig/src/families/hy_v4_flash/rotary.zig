//! Interleaved 3D M-RoPE: one frequency table per attention kind, folded into time/height/width lanes.
const std = @import("std");
const config = @import("config.zig");

const Kind = config.Kind;

/// Fill cos/sin for the rotary slice width (qk_rope_head_dim, e.g. 64).
pub fn cosSin(c: *const config.Config, kind: Kind, pos: i64, cos: []f32, sin: []f32) !void {
    const rope_dim: usize = if (kind == .sliding) c.swa_qk_rope else c.qk_rope;
    if (cos.len != rope_dim or sin.len != rope_dim or rope_dim % 2 != 0) return error.BadRotaryShape;
    const base: f64 = @floatCast(c.ropeTheta(kind));
    const half = rope_dim / 2;
    var plane: [3][]f32 = undefined;
    const scratch = try std.heap.page_allocator.alloc(f32, 3 * half);
    defer std.heap.page_allocator.free(scratch);
    for (0..3) |i| plane[i] = scratch[i * half ..][0..half];
    for (0..half) |i| {
        const inv = 1.0 / std.math.pow(f64, base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rope_dim)));
        const phase = @as(f64, @floatFromInt(pos)) * inv;
        for (0..3) |p| plane[p][i] = @floatCast(@cos(phase));
    }
    // The time plane owns indices where i%%3==0; height and width overwrite i%%3==1 and i%%3==2 lanes.
    for (1..3) |dim| {
        const count: usize = c.mrope_section[dim];
        const offset: usize = dim;
        var idx: usize = offset;
        var n: usize = 0;
        while (n < count and idx < half) : (n += 1) {
            plane[0][idx] = plane[dim][idx];
            idx += 3;
        }
    }
    for (0..half) |i| {
        cos[2 * i] = plane[0][i];
        cos[2 * i + 1] = plane[0][i];
    }
    const cscratch = try std.heap.page_allocator.alloc(f32, 3 * half);
    defer std.heap.page_allocator.free(cscratch);
    for (0..3) |i| plane[i] = cscratch[i * half ..][0..half];
    for (0..half) |i| {
        const inv = 1.0 / std.math.pow(f64, base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rope_dim)));
        const phase = @as(f64, @floatFromInt(pos)) * inv;
        for (0..3) |p| plane[p][i] = @floatCast(@sin(phase));
    }
    for (1..3) |dim| {
        const count: usize = c.mrope_section[dim];
        var idx: usize = dim;
        var n: usize = 0;
        while (n < count and idx < half) : (n += 1) {
            plane[0][idx] = plane[dim][idx];
            idx += 3;
        }
    }
    for (0..half) |i| {
        sin[2 * i] = plane[0][i];
        sin[2 * i + 1] = plane[0][i];
    }
}

/// Rotate the trailing `rope_dim` channels, GLM-style interleaved pairs; leading channels pass through.
pub fn apply(x: []f32, cos: []const f32, sin: []const f32, rope_dim: usize) !void {
    if (x.len < rope_dim or cos.len != rope_dim or sin.len != rope_dim) return error.BadRotaryShape;
    const start = x.len - rope_dim;
    for (0..rope_dim / 2) |i| {
        const a = x[start + 2 * i];
        const b = x[start + 2 * i + 1];
        x[start + 2 * i] = a * cos[2 * i] - b * sin[2 * i];
        x[start + 2 * i + 1] = b * cos[2 * i + 1] + a * sin[2 * i + 1];
    }
}

test "hy4 host: rotary at position zero is identity and later positions stay finite" {
    const c = try @import("config.zig").parse(std.testing.allocator, @embedFile("config_synthetic_fixture.json"));
    const dim = c.qk_rope;
    const cos = try std.testing.allocator.alloc(f32, dim);
    defer std.testing.allocator.free(cos);
    const sin = try std.testing.allocator.alloc(f32, dim);
    defer std.testing.allocator.free(sin);
    try cosSin(&c, .sparse, 0, cos, sin);
    for (cos) |v| try std.testing.expectEqual(@as(f32, 1), v);
    for (sin) |v| try std.testing.expectEqual(@as(f32, 0), v);
    const x = try std.testing.allocator.alloc(f32, dim);
    defer std.testing.allocator.free(x);
    for (x, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    const before = try std.testing.allocator.dupe(f32, x);
    defer std.testing.allocator.free(before);
    try apply(x, cos, sin, dim);
    try std.testing.expectEqualSlices(f32, before, x);
    try cosSin(&c, .sliding, 7, cos, sin);
    for (cos) |v| try std.testing.expect(std.math.isFinite(v));
}
