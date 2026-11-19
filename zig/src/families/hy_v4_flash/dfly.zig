//! DFly2 drafter over the checkpoint dfly_layers tensors: gated MLA blocks with sinks.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");
const rotary = @import("rotary.zig");
const moe_unused = @import("moe.zig");
const ops_mod = @import("ops.zig");

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s, .affine4, .bf16raw => error.ExpectedF32Tensor,
    };
}

fn matvecW(y: []f32, bag: *const weights.Bag, name: []const u8, x: []const f32, rows: usize, cols: usize) !void {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    switch (t.value) {
        .f32s => |v| try ops_mod.matvec(y, v, x, rows, cols),
        .bf16raw => |v| try ops_mod.matvecBf16(y, v, x, rows, cols),
        .i32s, .affine4 => return error.ExpectedF32Tensor,
    }
}

/// The draft's attention target set: sum the target hidden into the stream, or targets only.
pub const Fusion = enum { sum_chain, target_only };

/// Per-layer causal KV: k_nope [cap][heads*nope], k_rope and v likewise (rope MQA-expanded).
const Kv = struct {
    k_nope: []f32 = &.{},
    k_rope: []f32 = &.{},
    v: []f32 = &.{},
    len: usize = 0,

    fn init(gpa: std.mem.Allocator, cap: usize, heads: usize, nope: usize, rope: usize, vdim: usize) !Kv {
        const n = try gpa.alloc(f32, cap * heads * nope);
        const r = try gpa.alloc(f32, cap * heads * rope);
        const v = try gpa.alloc(f32, cap * heads * vdim);
        return .{ .k_nope = n, .k_rope = r, .v = v };
    }
    fn deinit(self: *Kv, gpa: std.mem.Allocator) void {
        gpa.free(self.k_nope);
        gpa.free(self.k_rope);
        gpa.free(self.v);
    }

    fn append(self: *Kv, k_nope: []const f32, k_rope: []const f32, v: []const f32) void {
        const heads_rows = k_nope.len;
        @memcpy(self.k_nope[self.len * heads_rows ..][0..heads_rows], k_nope);
        @memcpy(self.k_rope[self.len * k_rope.len ..][0..k_rope.len], k_rope);
        @memcpy(self.v[self.len * v.len ..][0..v.len], v);
        self.len += 1;
    }
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    c: *const config.Config,
    bag: *const weights.Bag,
    kv: [8]Kv = @splat(.{}),
    len: usize = 0,

    pub fn init(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, cap: usize) !Drafter {
        var d: Drafter = .{ .gpa = gpa, .c = c, .bag = bag };
        for (0..c.dfly.layers) |li| {
            d.kv[li] = try Kv.init(gpa, cap, c.heads, c.qk_nope, c.qk_rope, c.v_dim);
        }
        return d;
    }

    pub fn deinit(self: *Drafter) void {
        for (0..self.c.dfly.layers) |li| self.kv[li].deinit(self.gpa);
    }

    fn nm(self: *const Drafter, a: std.mem.Allocator, li: usize, suffix: []const u8) ![]const u8 {
        _ = self;
        return std.fmt.allocPrint(a, "dfly_layers.{d}.{s}", .{ li, suffix });
    }

    /// One position: h_targets are the collapsed hiddens entering the target layers here.
    pub fn step(self: *Drafter, h_targets: []const []const f32, embed_tok: []const f32, pos: usize, fusion: Fusion, out: []f32) !void {
        const c = self.c;
        const gpa = self.gpa;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const h = c.hidden;
        const heads = c.heads;
        const nope = c.qk_nope;
        const rope = c.qk_rope;
        const vdim = c.v_dim;
        const q_lora = c.q_lora;
        const inter = 3 * c.hidden;
        if (out.len != h or embed_tok.len != h or h_targets.len != 0 and h_targets[0].len != h) return error.BadDraftShape;

        var stream = try a.alloc(f32, h);
        @memcpy(stream, embed_tok);

        const tmp = try a.alloc(f32, h);
        const fused = try a.alloc(f32, h);
        const qa = try a.alloc(f32, q_lora);
        const qn = try a.alloc(f32, q_lora);
        const qbuf = try a.alloc(f32, heads * (nope + rope));
        const q_nope = try a.alloc(f32, heads * nope);
        const q_rope = try a.alloc(f32, heads * rope);
        const kva = try a.alloc(f32, c.kv_lora + rope);
        const kvn = try a.alloc(f32, c.kv_lora);
        const kvb = try a.alloc(f32, heads * (nope + vdim));
        const k_rope = try a.alloc(f32, heads * rope);
        const gate = try a.alloc(f32, heads * vdim);
        const ctx = try a.alloc(f32, heads * vdim);
        const cos = try a.alloc(f32, rope);
        const sin = try a.alloc(f32, rope);
        const scores = try a.alloc(f32, self.len + 2);
        const gu = try a.alloc(f32, inter);
        const up = try a.alloc(f32, inter);

        for (0..c.dfly.layers) |li| {
            const target = if (li < h_targets.len) h_targets[li] else stream;
            switch (fusion) {
                .sum_chain => {
                    for (0..h) |i| fused[i] = target[i] + stream[i];
                },
                .target_only => {
                    @memcpy(fused, target);
                },
            }
            const ln1 = try self.nm(a, li, "input_layernorm.weight");
            try ops.rmsNorm(tmp, fused, try f32s(self.bag, ln1), c.eps);

            try matvecW(qa, self.bag, try self.nm(a, li, "self_attn.q_a_proj.weight"), tmp, q_lora, h);
            try ops.rmsNorm(qn, qa, try f32s(self.bag, try self.nm(a, li, "self_attn.q_a_layernorm.weight")), c.eps);
            try matvecW(qbuf, self.bag, try self.nm(a, li, "self_attn.q_b_proj.weight"), qn, heads * (nope + rope), q_lora);
            try rotary.cosSin(c, .sliding, @intCast(pos), cos, sin);
            for (0..heads) |hh| {
                const qh = qbuf[hh * (nope + rope) ..][0 .. nope + rope];
                try rotary.apply(qh, cos, sin, rope);
                @memcpy(q_nope[hh * nope ..][0..nope], qh[0..nope]);
                @memcpy(q_rope[hh * rope ..][0..rope], qh[nope..][0..rope]);
            }

            try matvecW(kva, self.bag, try self.nm(a, li, "self_attn.kv_a_proj_with_mqa.weight"), tmp, c.kv_lora + rope, h);
            try ops.rmsNorm(kvn, kva[0..c.kv_lora], try f32s(self.bag, try self.nm(a, li, "self_attn.kv_a_layernorm.weight")), c.eps);
            try matvecW(kvb, self.bag, try self.nm(a, li, "self_attn.kv_b_proj.weight"), kvn, heads * (nope + vdim), c.kv_lora);
            for (0..heads) |hh| @memcpy(k_rope[hh * rope ..][0..rope], kva[c.kv_lora..][0..rope]);
            const kn_all = try a.alloc(f32, heads * nope);
            const v_all = try a.alloc(f32, heads * vdim);
            for (0..heads) |hh| {
                @memcpy(kn_all[hh * nope ..][0..nope], kvb[hh * (nope + vdim) ..][0..nope]);
                @memcpy(v_all[hh * vdim ..][0..vdim], kvb[hh * (nope + vdim) + nope ..][0..vdim]);
            }
            self.kv[li].append(kn_all, k_rope, v_all);

// scores with the learned sink as an extra softmax column, as the main attention
            const sink = try f32s(self.bag, try self.nm(a, li, "self_attn.sinks"));
            try matvecW(gate, self.bag, try self.nm(a, li, "self_attn.gate_proj.weight"), tmp, heads * vdim, h);
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(nope + rope)));
// attend causal over all keys incl. this step's
            const kv = &self.kv[li];
            const total = kv.len;
            for (0..heads) |hh| {
                var mx: f32 = -std.math.inf(f32);
                for (0..total) |t| {
                    var s: f32 = 0;
                    const kn = kv.k_nope[t * heads * nope + hh * nope ..][0..nope];
                    for (0..nope) |i| s += q_nope[hh * nope + i] * kn[i];
                    const kr = kv.k_rope[t * heads * rope + hh * rope ..][0..rope];
                    for (0..rope) |i| s += q_rope[hh * rope + i] * kr[i];
                    s *= scale;
                    scores[t] = s;
                    mx = @max(mx, s);
                }
                scores[total] = sink[hh];
                mx = @max(mx, sink[hh]);
                var denom: f32 = 0;
                for (0..total + 1) |t| {
                    scores[t] = @exp(scores[t] - mx);
                    denom += scores[t];
                }
                const inv = 1.0 / denom;
                for (0..vdim) |i| {
                    var acc: f32 = 0;
                    for (0..total) |t| {
                        const vv = kv.v[t * heads * vdim + hh * vdim ..][0..vdim];
                        acc += scores[t] * inv * vv[i];
                    }
                    ctx[hh * vdim + i] = acc;
                }
            }
            for (0..heads * vdim) |i| ctx[i] *= ops.sigmoid(gate[i]);

            const attn_out = try a.alloc(f32, h);
            try matvecW(attn_out, self.bag, try self.nm(a, li, "self_attn.o_proj.weight"), ctx, h, heads * vdim);
            const pn = try f32s(self.bag, try self.nm(a, li, "self_attn.post_proj_norm.weight"));
            try ops.rmsNorm(tmp, attn_out, pn, c.eps);
            for (0..h) |i| stream[i] += tmp[i];

            try ops.rmsNorm(tmp, stream, try f32s(self.bag, try self.nm(a, li, "post_attention_layernorm.weight")), c.eps);
            try matvecW(gu, self.bag, try self.nm(a, li, "mlp.gate_proj.weight"), tmp, inter, h);
            try matvecW(up, self.bag, try self.nm(a, li, "mlp.up_proj.weight"), tmp, inter, h);
            for (gu, up) |*g, *u| g.* = g.* * ops.sigmoid(g.*) * u.*;
            try matvecW(tmp, self.bag, try self.nm(a, li, "mlp.down_proj.weight"), gu, h, inter);
            for (0..h) |i| stream[i] += tmp[i];
        }
        @memcpy(out, stream);
        self.len += 1;
    }
};
