//! Compressed-indexer group recall cache: pool token keys, score them, choose recalled groups.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");
const indexer = @import("indexer.zig");
const rotary = @import("rotary.zig");

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s, .affine4 => error.ExpectedF32Tensor,
        .bf16raw => return error.ExpectedF32Tensor,
    };
}

fn matvecW(y: []f32, bag: *const weights.Bag, name: []const u8, x: []const f32, rows: usize, cols: usize) !void {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    switch (t.value) {
        .f32s => |v| try ops.matvec(y, v, x, rows, cols),
        .bf16raw => |v| try ops.matvecBf16(y, v, x, rows, cols),
        .i32s, .affine4 => return error.ExpectedF32Tensor,
    }
}

pub const Recall = struct {
    gpa: std.mem.Allocator,
    dim: usize,
    heads: usize,
    ratio: usize,
    topk: usize,
    cap: usize,
    tokens: usize = 0,
    pending_n: usize = 0,
    pending: []f32,
    keys: []f32,

    pub fn init(gpa: std.mem.Allocator, c: *const config.Config, cap_tokens: usize) !Recall {
        if (c.index_dim == 0 or c.index_heads == 0 or c.index_ratio == 0 or c.index_group_topk == 0) return error.BadRecallShape;
        const groups = (cap_tokens + c.index_ratio - 1) / c.index_ratio;
        return .{
            .gpa = gpa,
            .dim = c.index_dim,
            .heads = c.index_heads,
            .ratio = c.index_ratio,
            .topk = c.index_group_topk,
            .cap = groups,
            .pending = try gpa.alloc(f32, c.index_ratio * c.index_dim),
            .keys = try gpa.alloc(f32, groups * c.index_dim),
        };
    }

    pub fn deinit(self: *Recall) void {
        self.gpa.free(self.pending);
        self.gpa.free(self.keys);
        self.* = undefined;
    }

    /// Pool one completed group from `ratio` normalized token keys.
    pub fn addToken(self: *Recall, bag: *const weights.Bag, layer: usize, token_key: []const f32) !void {
        if (token_key.len != self.dim or self.tokens / self.ratio >= self.cap) return error.BadRecallShape;
        @memcpy(self.pending[self.pending_n * self.dim ..][0..self.dim], token_key);
        self.pending_n += 1;
        self.tokens += 1;
        if (self.pending_n < self.ratio) return;
        var slots: [16][]const f32 = undefined;
        for (0..self.ratio) |i| slots[i] = self.pending[i * self.dim ..][0..self.dim];
        const group = self.tokens / self.ratio - 1;
        try indexer.poolOne(self.gpa, bag, layer, slots[0..self.ratio], self.keys[group * self.dim ..][0..self.dim]);
        self.pending_n = 0;
    }

    /// Return the groups a query recalls; the caller passes the current token position.
    pub fn select(self: *Recall, c: *const config.Config, bag: *const weights.Bag, layer: usize, pos: usize, q_resid: []const f32, hidden: []const f32) ![]usize {
        const groups = self.tokens / self.ratio;
        if (groups == 0) {
            const own = try self.gpa.alloc(usize, 1);
            own[0] = 0;
            return own;
        }
        if (q_resid.len != c.q_lora or hidden.len != c.hidden) return error.BadRecallShape;
        const qn = try std.fmt.allocPrint(self.gpa, "layers.{d}.self_attn.compressed_indexer.query_projection.weight", .{layer});
        defer self.gpa.free(qn);
        const hn = try std.fmt.allocPrint(self.gpa, "layers.{d}.self_attn.compressed_indexer.head_weight_projection.weight", .{layer});
        defer self.gpa.free(hn);
        const q = try self.gpa.alloc(f32, self.heads * self.dim);
        defer self.gpa.free(q);
        try matvecW(q, bag, qn, q_resid, self.heads * self.dim, c.q_lora);
        const cos = try self.gpa.alloc(f32, c.qk_rope);
        defer self.gpa.free(cos);
        const sin = try self.gpa.alloc(f32, c.qk_rope);
        defer self.gpa.free(sin);
        try rotary.cosSin(c, .sparse, @intCast(pos), cos, sin);
        for (0..self.heads) |h| try rotary.apply(q[h * self.dim ..][0..self.dim], cos, sin, c.qk_rope);
        const key_views = try self.gpa.alloc([]const f32, groups);
        defer self.gpa.free(key_views);
        for (0..groups) |g| {
            const buf = try self.gpa.alloc(f32, self.dim);
            @memcpy(buf, self.keys[g * self.dim ..][0..self.dim]);
            try rotary.cosSin(c, .sparse, @intCast(g * self.ratio), cos, sin);
            try rotary.apply(buf, cos, sin, c.qk_rope);
            key_views[g] = buf;
        }
        defer for (key_views) |v| self.gpa.free(v);
        const q_views = try self.gpa.alloc([]const f32, self.heads);
        defer self.gpa.free(q_views);
        for (0..self.heads) |h| q_views[h] = q[h * self.dim ..][0..self.dim];
        const hw = try self.gpa.alloc(f32, self.heads);
        defer self.gpa.free(hw);
        try matvecW(hw, bag, hn, hidden, self.heads, c.hidden);
        const norm = 1.0 / @sqrt(@as(f64, @floatFromInt(self.heads * self.dim)));
        for (hw) |*w| w.* *= @floatCast(norm);
        const scores = try self.gpa.alloc(f32, self.heads * groups);
        defer self.gpa.free(scores);
        try indexer.score(q_views, key_views, hw, scores);
        const dst = try self.gpa.alloc(usize, self.topk);
        errdefer self.gpa.free(dst);
        const qg = @min((self.tokens - 1) / self.ratio, groups - 1);
        try indexer.select(scores, groups, self.heads, self.topk, qg, dst);
        return dst;
    }

    pub fn deinitResult(self: *Recall, r: []usize) void {
        self.gpa.free(r);
    }
};

test "hy4 host: recall pools groups and returns the query group" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try config.parse(gpa, @embedFile("config_synthetic_fixture.json"));
    var recall = try Recall.init(gpa, &c, 8);
    defer recall.deinit();
    const key = try gpa.alloc(f32, c.index_dim);
    defer gpa.free(key);
    @memset(key, 0.1);
    for (0..8) |_| try recall.addToken(&bag, 0, key);
    const q = try gpa.alloc(f32, c.q_lora);
    defer gpa.free(q);
    @memset(q, 0);
    const h = try gpa.alloc(f32, c.hidden);
    defer gpa.free(h);
    @memset(h, 0);
    const picked = try recall.select(&c, &bag, 0, 7, q, h);
    defer recall.deinitResult(picked);
    try std.testing.expectEqual(@as(usize, c.index_group_topk), picked.len);
    try std.testing.expectEqual(@as(usize, 3), picked[picked.len - 1]);
}
