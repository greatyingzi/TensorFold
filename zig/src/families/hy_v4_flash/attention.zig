//! Multi-token MLA attention with causal, sliding-window and indexer-recalled key sets.
const std = @import("std");
const config = @import("config.zig");

pub const LayerKv = struct {
    heads: usize,
    nope: usize,
    rope: usize,
    vdim: usize,
    cap: usize,
    tokens: usize = 0,
    keys_nope: []f32,
    keys_rope: []f32,
    values: []f32,

    pub fn init(gpa: std.mem.Allocator, heads: usize, nope: usize, rope: usize, vdim: usize, cap: usize) !LayerKv {
        if (heads == 0 or nope == 0 or rope == 0 or vdim == 0 or cap == 0) return error.BadAttentionShape;
        return .{
            .heads = heads,
            .nope = nope,
            .rope = rope,
            .vdim = vdim,
            .cap = cap,
            .keys_nope = try gpa.alloc(f32, cap * heads * nope),
            .keys_rope = try gpa.alloc(f32, cap * heads * rope),
            .values = try gpa.alloc(f32, cap * heads * vdim),
        };
    }

    pub fn deinit(self: *LayerKv, gpa: std.mem.Allocator) void {
        gpa.free(self.keys_nope);
        gpa.free(self.keys_rope);
        gpa.free(self.values);
        self.* = undefined;
    }

    pub fn append(self: *LayerKv, k_nope: []const f32, k_rope: []const f32, v: []const f32) !void {
        if (self.tokens >= self.cap) return error.AttentionCacheFull;
        if (k_nope.len != self.heads * self.nope or k_rope.len != self.heads * self.rope or v.len != self.heads * self.vdim) return error.BadAttentionShape;
        const t = self.tokens;
        @memcpy(self.keys_nope[t * self.heads * self.nope ..][0 .. self.heads * self.nope], k_nope);
        @memcpy(self.keys_rope[t * self.heads * self.rope ..][0 .. self.heads * self.rope], k_rope);
        @memcpy(self.values[t * self.heads * self.vdim ..][0 .. self.heads * self.vdim], v);
        self.tokens += 1;
    }

    fn allowed(self: *const LayerKv, t: usize, kind: config.Kind, window: usize, selected: ?[]const usize, ratio: usize) bool {
        if (t >= self.tokens) return false;
        if (kind == .sliding and window > 0 and t + window <= self.tokens) return false;
        if (selected) |groups| {
            const g = t / ratio;
            for (groups) |s| if (s == g) return true;
            return false;
        }
        return true;
    }

    /// Attend the last appended token against all allowed positions and write [heads, vdim].
    pub fn attend(self: *const LayerKv, gpa: std.mem.Allocator, kind: config.Kind, window: usize, selected: ?[]const usize, ratio: usize, q_nope: []const f32, q_rope: []const f32, out: []f32) !void {
        if (self.tokens == 0) return error.EmptyAttentionCache;
        if (q_nope.len != self.heads * self.nope or q_rope.len != self.heads * self.rope or out.len != self.heads * self.vdim) return error.BadAttentionShape;
        const scores = try gpa.alloc(f32, self.tokens);
        defer gpa.free(scores);
        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(self.nope + self.rope)));
        @memset(out, 0);
        for (0..self.heads) |h| {
            var best: f64 = -std.math.inf(f64);
            for (0..self.tokens) |t| {
                if (!self.allowed(t, kind, window, selected, ratio)) {
                    scores[t] = -std.math.inf(f32);
                    continue;
                }
                var dot: f64 = 0;
                const qn = q_nope[h * self.nope ..][0..self.nope];
                const kn = self.keys_nope[t * self.heads * self.nope + h * self.nope ..][0..self.nope];
                for (qn, kn) |a, b| dot += @as(f64, a) * @as(f64, b);
                const qr = q_rope[h * self.rope ..][0..self.rope];
                const kr = self.keys_rope[t * self.heads * self.rope + h * self.rope ..][0..self.rope];
                for (qr, kr) |a, b| dot += @as(f64, a) * @as(f64, b);
                scores[t] = @floatCast(dot * scale);
                if (scores[t] > best) best = scores[t];
            }
            var total: f64 = 0;
            for (scores) |s| {
                if (std.math.isFinite(s)) total += @exp(@as(f64, s) - best);
            }
            for (0..self.tokens) |t| {
                if (!std.math.isFinite(scores[t])) continue;
                const w = @exp(@as(f64, scores[t]) - best) / total;
                const vv = self.values[t * self.heads * self.vdim + h * self.vdim ..][0..self.vdim];
                for (vv, 0..) |v, i| out[h * self.vdim + i] += @floatCast(w * @as(f64, v));
            }
        }
    }
};

test "hy4 host: causal attention keeps only the current token at position zero" {
    const gpa = std.testing.allocator;
    var kv = try LayerKv.init(gpa, 2, 4, 2, 3, 4);
    defer kv.deinit(gpa);
    const kn: [8]f32 = @splat(1);
    const kr: [4]f32 = @splat(1);
    const v: [6]f32 = @splat(2);
    try kv.append(&kn, &kr, &v);
    const out = try gpa.alloc(f32, 6);
    defer gpa.free(out);
    const qn: [8]f32 = @splat(1);
    const qr: [4]f32 = @splat(1);
    try kv.attend(gpa, .sparse, 0, null, 1, &qn, &qr, out);
    for (out) |x| try std.testing.expect(@abs(x - 2) < 1e-4);

    const kn2: [8]f32 = @splat(3);
    const kr2: [4]f32 = @splat(3);
    const v2: [6]f32 = @splat(9);
    try kv.append(&kn2, &kr2, &v2);
    try kv.attend(gpa, .sparse, 0, null, 1, &qn, &qr, out);
    var max: f32 = 0;
    for (out) |x| max = @max(max, x);
    try std.testing.expect(max > 2 and max <= 9);
}
