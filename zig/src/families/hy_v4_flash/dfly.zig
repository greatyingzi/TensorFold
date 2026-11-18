//! Hy4 Flash DFly2 drafter: five dense blocks over the main model's target-layer hiddens.
//! CPU fp32 reference mirroring session.zig; the fusion variant is chosen by the real-weight probe.
const std = @import("std");
const cfg = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");

/// How block i mixes its target-layer hidden with the draft stream.
pub const Fusion = enum {
    /// block input = norm(ln1, h_target + draft_stream)
    seq_norm_sum,
    /// block input = h_target + draft_stream, no extra norm
    plain_sum,
};

pub const block_heads = 32;
pub const nope = 192;
pub const rope = 64;
pub const v_dim = 256;
pub const qk = nope + rope;

fn rmsNorm(w: []const f32, x: []const f32, out: []f32) void {
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + 1e-6);
    for (x, 0..) |v, i| out[i] = v * w[i] * @as(f32, @floatCast(inv));
}

fn ropeNeox(x: []f32, t: usize, theta: f64, out: []f32) void {
    const half = rope / 2;
    for (0..half) |j| {
        const freq = std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * j)) / rope);
        const ang = freq * @as(f64, @floatFromInt(t));
        const c: f32 = @floatCast(@cos(ang));
        const s: f32 = @floatCast(@sin(ang));
        out[j] = x[j] * c - x[half + j] * s;
        out[half + j] = x[j] * s + x[half + j] * c;
    }
}

/// One dfly block over a [T, 3072] stream plus its target hidden; weights via bag lookups.
pub fn blockForward(
    gpa: std.mem.Allocator,
    bag: *const weights.Bag,
    li: usize,
    x: []const f32, // [T, 3072] the draft stream so far (fp32)
    out: []f32, // [T, 3072]
    theta: f64,
) !void {
    _ = gpa;
    _ = bag;
    _ = li;
    _ = x;
    _ = out;
    _ = theta;
    return error.DflyBlockUnimplemented;
}

/// The five-block chain: embed anchor -> block(target, stream) x5.
/// h_t[i] is the main model's residual into target layer 25 + 4i, at every position.
pub fn draftForward(
    gpa: std.mem.Allocator,
    c: *const cfg.Config,
    bag: *const weights.Bag,
    embed: []const f32, // [T, 3072] token embeddings at the anchor positions
    h_t: []const []const f32, // 5 x [T, 3072] target-layer hiddens
    out: []f32, // [T, 3072] final draft hidden
    fusion: Fusion,
) !void {
    _ = gpa;
    _ = c;
    _ = bag;
    _ = embed;
    _ = h_t;
    _ = out;
    _ = fusion;
    return error.DflyForwardUnimplemented;
}
