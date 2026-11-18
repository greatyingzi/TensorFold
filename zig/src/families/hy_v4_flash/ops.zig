//! Host-side reference arithmetic for the Hy4 Flash forward; f32 accumulation, no fused approximations.
const std = @import("std");

pub fn rmsNorm(out: []f32, x: []const f32, weight: []const f32, eps: f32) !void {
    if (out.len != x.len or weight.len != x.len) return error.BadNormShape;
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * @as(f64, v);
    const scale: f64 = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + @as(f64, eps));
    for (x, 0..) |v, i| out[i] = @floatCast(@as(f64, v) * scale * @as(f64, weight[i]));
}

/// Row-major y = W x, with W shaped [rows, cols].
pub fn matvec(y: []f32, w: []const f32, x: []const f32, rows: usize, cols: usize) !void {
    if (y.len != rows or w.len != rows * cols or x.len != cols) return error.BadMatvecShape;
    for (0..rows) |r| {
        var sum: f64 = 0;
        const base = r * cols;
        for (0..cols) |c| sum += @as(f64, w[base + c]) * @as(f64, x[c]);
        y[r] = @floatCast(sum);
    }
}

pub fn sigmoid(x: f32) f32 {
    return @floatCast(1.0 / (1.0 + @exp(-@as(f64, x))));
}

const SortCtx = struct {
    values: []const f32,

    pub fn lessThan(ctx: SortCtx, a: usize, b: usize) bool {
        if (ctx.values[a] != ctx.values[b]) return ctx.values[a] > ctx.values[b];
        return a < b;
    }
};

/// Indices of the largest `k` values, descending by value then index; NaN is refused.
pub fn topk(dst: []usize, values: []const f32, k: usize) !void {
    if (dst.len < k or k > values.len or values.len > 1024) return error.BadTopkShape;
    for (values) |v| if (std.math.isNan(v)) return error.NanInRouting;
    var idx: [1024]usize = undefined;
    for (0..values.len) |i| idx[i] = i;
    std.mem.sort(usize, idx[0..values.len], SortCtx{ .values = values }, SortCtx.lessThan);
    for (0..k) |i| dst[i] = idx[i];
}

test "hy4 host: rmsnorm matches a hand-computed f64 reference" {
    var out: [4]f32 = undefined;
    const x = [4]f32{ 1, 2, 3, 4 };
    const w = [4]f32{ 1, 1, 1, 1 };
    try rmsNorm(&out, &x, &w, 1e-5);
    const mean_sq: f64 = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    const s: f64 = 1.0 / @sqrt(mean_sq + 1e-5);
    try std.testing.expect(@abs(out[0] - @as(f32, @floatCast(s))) < 1e-6);
    try std.testing.expect(@abs(out[3] - @as(f32, @floatCast(4.0 * s))) < 1e-6);
    try std.testing.expectError(error.BadNormShape, rmsNorm(&out, &x, w[0..3], 1e-5));
}

test "hy4 host: matvec and sigmoid are exact on small known inputs" {
    var y: [2]f32 = undefined;
    const w = [6]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [3]f32{ 1, 2, 3 };
    try matvec(&y, &w, &x, 2, 3);
    try std.testing.expectEqual(@as(f32, 14), y[0]);
    try std.testing.expectEqual(@as(f32, 32), y[1]);
    try std.testing.expect(sigmoid(0) == 0.5);
    try std.testing.expectError(error.BadMatvecShape, matvec(&y, &w, &x, 2, 4));
}

test "hy4 host: topk picks the largest values in order and refuses NaN" {
    const v = [6]f32{ 1, 5, 3, 9, 2, 7 };
    var idx: [3]usize = undefined;
    try topk(&idx, &v, 3);
    try std.testing.expectEqual(@as(usize, 3), idx[0]);
    try std.testing.expectEqual(@as(usize, 5), idx[1]);
    try std.testing.expectEqual(@as(usize, 1), idx[2]);
    var bad = v;
    bad[2] = std.math.nan(f32);
    try std.testing.expectError(error.NanInRouting, topk(&idx, &bad, 3));
}
