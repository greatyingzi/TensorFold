//! Token-by-token session: multi-token attention attached to the full layer stack.

// Scope: indexer recall is not wired yet; sparse layers use causal attention with all prior tokens.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");
const ihc = @import("ihc.zig");
const engram = @import("engram.zig");
const moe = @import("moe.zig");
const rotary = @import("rotary.zig");
const graph = @import("graph.zig");
const attention = @import("attention.zig");
const recall_mod = @import("recall.zig");

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s, .affine4 => error.ExpectedF32Tensor,
        .bf16raw => {
            std.debug.print("F32S_BF16RAW {s}\n", .{name});
            return error.ExpectedF32Tensor;
        },
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

fn named(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, fmt, args);
}

pub const Session = struct {
    gpa: std.mem.Allocator,
    c: *const config.Config,
    bag: *const weights.Bag,
    kv: [64]attention.LayerKv = undefined,
    ready: [64]bool = @splat(false),
    recall: [64]recall_mod.Recall = undefined,
    recall_ready: [64]bool = @splat(false),
    streams: [16][]f32 = undefined,
    next: [16][]f32 = undefined,
    logits: []f32,
    last_hidden: []f32,
    pos: usize = 0,
    /// Optional per-step capture of the collapsed hidden entering these layers (the dfly targets).
    dump_layers: [42]u32 = @splat(0),
    dump_n: usize = 0,
    dump_buf: [42][]f32 = @splat(&.{}),
    dump_hit: [42]bool = @splat(false),

    pub fn init(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, cap: usize) !Session {
        try graph.validate(c);
        var self = Session{
            .gpa = gpa,
            .c = c,
            .bag = bag,
            .logits = try gpa.alloc(f32, c.vocab),
            .last_hidden = try gpa.alloc(f32, c.hidden),
        };
        errdefer gpa.free(self.logits);
        errdefer gpa.free(self.last_hidden);
        var ok_layers: usize = 0;
        errdefer for (0..ok_layers) |i| self.kv[i].deinit(gpa);
        for (0..c.layers) |i| {
            const plan = try graph.layerPlan(c, i);
            const nope = if (plan.kind == .sliding) c.swa_qk_nope else c.qk_nope;
            const rope = if (plan.kind == .sliding) c.swa_qk_rope else c.qk_rope;
            const vdim = if (plan.kind == .sliding) c.swa_v_dim else c.v_dim;
            self.kv[i] = try attention.LayerKv.init(gpa, c.layer[i].heads, nope, rope, vdim, cap);
            self.ready[i] = true;
            if (plan.kind == .sparse) {
                self.recall[i] = try recall_mod.Recall.init(gpa, c, cap);
                self.recall_ready[i] = true;
            }
            ok_layers += 1;
        }
        const emb_t = bag.get("embed_tokens.weight") orelse return error.MissingHy4Tensor;
        const emb_elems: usize = emb_t.shape[0] * emb_t.shape[1];
        if (emb_t.rank != 2 or emb_elems != c.vocab * c.hidden) return error.BadForwardShape;
        for (0..c.hc_mult) |s| {
            self.streams[s] = try gpa.alloc(f32, c.hidden);
            self.next[s] = try gpa.alloc(f32, c.hidden);
            @memset(self.streams[s], 0);
        }
        return self;
    }

    pub fn deinit(self: *Session) void {
        for (0..self.c.layers) |i| {
            if (self.ready[i]) self.kv[i].deinit(self.gpa);
            if (self.recall_ready[i]) self.recall[i].deinit();
        }
        for (0..self.c.hc_mult) |s| {
            self.gpa.free(self.streams[s]);
            self.gpa.free(self.next[s]);
        }
        self.gpa.free(self.logits);
        self.gpa.free(self.last_hidden);
        self.* = undefined;
    }

    fn expandRope(self: *Session, src: []const f32, heads: usize, rope: usize, dst: []f32) !void {
        const kv_heads = self.c.kv_heads;
        if (dst.len != heads * rope) return error.BadAttentionShape;
        if (src.len == rope) {
            for (0..heads) |h| @memcpy(dst[h * rope ..][0..rope], src);
            return;
        }
        if (src.len != kv_heads * rope or heads % kv_heads != 0) return error.BadAttentionShape;
        const rep = heads / kv_heads;
        for (0..heads) |h| @memcpy(dst[h * rope ..][0..rope], src[(h / rep) * rope ..][0..rope]);
    }

    /// Consume one token and return the logits for the next-token position.
    pub fn step(self: *Session, token: usize) ![]const f32 {
        if (token >= self.c.vocab or self.ready[0] == false) return error.BadForwardShape;
        const gpa = self.gpa;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const c = self.c;
        const h = c.hidden;
        const hc = c.hc_mult;
        const emb_t = self.bag.get("embed_tokens.weight") orelse return error.MissingHy4Tensor;
        switch (emb_t.value) {
            .f32s => |v| {
                for (0..hc) |s| @memcpy(self.streams[s], v[token * h ..][0..h]);
            },
            .bf16raw => |v| {
                const base = token * h * 2;
                for (0..h) |i| {
                    const bits = std.mem.readInt(u16, v[base + i * 2 ..][0..2], .little);
                    const decoded: f32 = @bitCast(@as(u32, bits) << 16);
                    for (0..hc) |s| self.streams[s][i] = decoded;
                }
            },
            .i32s, .affine4 => return error.ExpectedF32Tensor,
        }

        const mixed = try a.alloc(f32, h);
        const ffn = try a.alloc(f32, h);
        for (0..c.layers) |li| {
            const plan = try graph.layerPlan(c, li);
            const nope = if (plan.kind == .sliding) c.swa_qk_nope else c.qk_nope;
            const rope = if (plan.kind == .sliding) c.swa_qk_rope else c.qk_rope;
            const vdim = if (plan.kind == .sliding) c.swa_v_dim else c.v_dim;
            const heads = c.layer[li].heads;
            const q_out = heads * (nope + rope);
            const kv_a_len = c.kv_lora + rope;
            const kv_b_len = heads * (nope + vdim);

            if (plan.engram) |ei| {
                _ = ei;
                const ngram = [3]u32{ @intCast(token), @intCast(token), @intCast(token) };
                var residual = self.next;
                try engram.inject(gpa, c, self.bag, li, ngram, self.streams[0..hc], residual[0..hc]);
                for (0..hc) |s| {
                    for (0..h) |i| self.streams[s][i] += residual[s][i];
                }
            }
            if (self.dump_n > 0 and li == 2) {
                for (0..hc) |ss| {
                    std.debug.print("CPUDUMP L2 stream{d} ", .{ss});
                    for (self.streams[ss][0..4]) |v| std.debug.print("{d:.3} ", .{v});
                    std.debug.print("\n", .{});
                }
            }
            const ag = try ihc.gates(gpa, c, self.bag, li, "attn_hc", self.streams[0..hc]);
            if (self.dump_n > 0 and li == 2) std.debug.print("CPUDUMP L2 ATTNGATES pres {d:.3} {d:.3} {d:.3} {d:.3} posts {d:.3} {d:.3}\n", .{ ag.pre[0], ag.pre[1], ag.pre[2], ag.pre[3], ag.post[0], ag.post[1] });
            defer ihc.gatesDeinit(gpa, ag);
            try ihc.collapse(ag, self.streams[0..hc], mixed);
            if (self.dump_n > 0) {
                for (0..self.dump_n) |di| {
                    if (self.dump_layers[di] == li and self.dump_buf[di].len == h) {
                        @memcpy(self.dump_buf[di], mixed);
                        self.dump_hit[di] = true;
                    }
                }
            }
            const an = try named(a, "layers.{d}.input_layernorm.weight", .{li});
            try ops.rmsNorm(mixed, mixed, try f32s(self.bag, an), c.eps);

            const qa = try a.alloc(f32, c.q_lora);
            const qan = try named(a, "layers.{d}.self_attn.q_a_proj.weight", .{li});
            try matvecW(qa, self.bag, qan, mixed, c.q_lora, h);
            const qn = try a.alloc(f32, c.q_lora);
            const qnl = try named(a, "layers.{d}.self_attn.q_a_layernorm.weight", .{li});
            try ops.rmsNorm(qn, qa, try f32s(self.bag, qnl), c.eps);
            const q = try a.alloc(f32, q_out);
            const qbn = try named(a, "layers.{d}.self_attn.q_b_proj.weight", .{li});
            try matvecW(q, self.bag, qbn, qn, q_out, c.q_lora);
            const cos = try a.alloc(f32, rope);
            const sin = try a.alloc(f32, rope);
            try rotary.cosSin(c, plan.kind, @intCast(self.pos), cos, sin);
            for (0..heads) |hh| try rotary.apply(q[hh * (nope + rope) ..][0 .. nope + rope], cos, sin, rope);
            const q_nope = try a.alloc(f32, heads * nope);
            const q_rope = try a.alloc(f32, heads * rope);
            for (0..heads) |hh| {
                @memcpy(q_nope[hh * nope ..][0..nope], q[hh * (nope + rope) ..][0..nope]);
                @memcpy(q_rope[hh * rope ..][0..rope], q[hh * (nope + rope) + nope ..][0..rope]);
            }

            const kva = try a.alloc(f32, kv_a_len);
            const kan = try named(a, "layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{li});
            try matvecW(kva, self.bag, kan, mixed, kv_a_len, h);
            const kvn = try a.alloc(f32, c.kv_lora);
            const kvnl = try named(a, "layers.{d}.self_attn.kv_a_layernorm.weight", .{li});
            try ops.rmsNorm(kvn, kva[0..c.kv_lora], try f32s(self.bag, kvnl), c.eps);
            const kvb = try a.alloc(f32, kv_b_len);
            const kbn = try named(a, "layers.{d}.self_attn.kv_b_proj.weight", .{li});
            try matvecW(kvb, self.bag, kbn, kvn, kv_b_len, c.kv_lora);
            const k_nope = try a.alloc(f32, heads * nope);
            const vbuf = try a.alloc(f32, heads * vdim);
            for (0..heads) |hh| {
                @memcpy(k_nope[hh * nope ..][0..nope], kvb[hh * (nope + vdim) ..][0..nope]);
                @memcpy(vbuf[hh * vdim ..][0..vdim], kvb[hh * (nope + vdim) + nope ..][0..vdim]);
            }
            const k_rope = try a.alloc(f32, heads * rope);
            try self.expandRope(kva[c.kv_lora..], heads, rope, k_rope);
            try self.kv[li].append(k_nope, k_rope, vbuf);
            var selected: ?[]usize = null;
            if (plan.kind == .sparse) {
                const tkn = try named(a, "layers.{d}.self_attn.compressed_indexer.token_key_projection.weight", .{li});
                const tknn = try named(a, "layers.{d}.self_attn.compressed_indexer.token_key_norm.weight", .{li});
                const tk = try a.alloc(f32, c.index_dim);
                try matvecW(tk, self.bag, tkn, mixed, c.index_dim, h);
                try ops.rmsNorm(tk, tk, try f32s(self.bag, tknn), c.eps);
                try self.recall[li].addToken(self.bag, li, tk);
                selected = try self.recall[li].select(c, self.bag, li, self.pos, qa, mixed);
                defer if (selected) |s| self.recall[li].deinitResult(s);
            }
            const ctx = try a.alloc(f32, heads * vdim);
            const sinkn = try named(a, "layers.{d}.self_attn.sinks", .{li});
            const sinks: ?[]const f32 = if (self.bag.get(sinkn)) |t| switch (t.value) {
                .f32s => |v| v,
                else => null,
            } else null;
            try self.kv[li].attendSink(gpa, plan.kind, c.window, selected, c.index_ratio, q_nope, q_rope, ctx, sinks);
            const on = try named(a, "layers.{d}.self_attn.o_proj.weight", .{li});
            try matvecW(ffn, self.bag, on, ctx, h, heads * vdim);
            // Official HYV4FlashAttention applies post_proj_norm after o_proj (every layer).
            const ppn = try named(a, "layers.{d}.self_attn.post_proj_norm.weight", .{li});
            defer a.free(ppn);
            try ops.rmsNorm(ffn, ffn, try f32s(self.bag, ppn), c.eps);
            if (self.dump_n > 0 and li == 1) {
                var fs: f64 = 0;
                for (ffn) |v| fs += @as(f64, v) * v;
                var cs: f64 = 0;
                for (ctx) |v| cs += @as(f64, v) * v;
                std.debug.print("CPUDUMP L1 BRANCH rms {d:.4} ctx_rms {d:.4}\n", .{ @sqrt(fs / @as(f64, @floatFromInt(ffn.len))), @sqrt(cs / @as(f64, @floatFromInt(ctx.len))) });
            }
            for (0..hc) |s| {
                for (0..h) |i| self.next[s][i] = ag.post[s] * ffn[i] + self.streams[s][i];
            }
            for (0..hc) |s| @memcpy(self.streams[s], self.next[s]);

            const fg = try ihc.gates(gpa, c, self.bag, li, "ffn_hc", self.streams[0..hc]);
            if (self.dump_n > 0 and li == 1) std.debug.print("CPUDUMP L1 FFNGATES posts {d:.3} {d:.3} {d:.3} {d:.3}\n", .{ fg.post[0], fg.post[1], fg.post[2], fg.post[3] });
            if (self.dump_n > 0 and li == 1) {
                std.debug.print("CPUDUMP L1 FFN8 ", .{});
                for (ffn[0..8]) |v| std.debug.print("{d:.3} ", .{v});
                std.debug.print("\n", .{});
            }
            defer ihc.gatesDeinit(gpa, fg);
            try ihc.collapse(fg, self.streams[0..hc], mixed);
            const pn = try named(a, "layers.{d}.post_attention_layernorm.weight", .{li});
            try ops.rmsNorm(mixed, mixed, try f32s(self.bag, pn), c.eps);
            if (plan.moe) try moe.latentMoe(gpa, c, self.bag, li, mixed, ffn) else try denseFfn(gpa, c, self.bag, li, mixed, ffn);
            if (self.dump_n > 0 and li == 1) {
                std.debug.print("CPUDUMP L1 FFN_AFTER ", .{});
                for (ffn[0..8]) |v| std.debug.print("{d:.3} ", .{v});
                std.debug.print("\n", .{});
            }
            if (self.dump_n > 0 and li == 1) {
                var fss: f64 = 0;
                for (ffn) |v| fss += @as(f64, v) * v;
                var mss: f64 = 0;
                for (mixed) |v| mss += @as(f64, v) * v;
                std.debug.print("CPUDUMP L1 ffn_rms {d:.4} moe_in_rms {d:.4} ffn0 {d:.4}\n", .{ @sqrt(fss / @as(f64, @floatFromInt(ffn.len))), @sqrt(mss / @as(f64, @floatFromInt(mixed.len))), ffn[0] });
            }
            for (0..hc) |s| {
                for (0..h) |i| self.next[s][i] = fg.post[s] * ffn[i] + self.streams[s][i];
            }
            for (0..hc) |s| @memcpy(self.streams[s], self.next[s]);
        }
        const final = try a.alloc(f32, h);
        try ihc.head(gpa, c, self.bag, self.streams[0..hc], final);
        try ops.rmsNorm(final, final, try f32s(self.bag, "norm.weight"), c.eps);
        @memcpy(self.last_hidden, final);
        try matvecW(self.logits, self.bag, "lm_head.weight", final, c.vocab, h);
        self.pos += 1;
        return self.logits;
    }

    fn denseFfn(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, layer: usize, x: []const f32, out: []f32) !void {
        const gate = try gpa.alloc(f32, c.intermediate);
        defer gpa.free(gate);
        const up = try gpa.alloc(f32, c.intermediate);
        defer gpa.free(up);
        const gn = try named(gpa, "layers.{d}.mlp.gate_proj.weight", .{layer});
        defer gpa.free(gn);
        const un = try named(gpa, "layers.{d}.mlp.up_proj.weight", .{layer});
        defer gpa.free(un);
        const dn = try named(gpa, "layers.{d}.mlp.down_proj.weight", .{layer});
        defer gpa.free(dn);
        try matvecW(gate, bag, gn, x, c.intermediate, c.hidden);
        try matvecW(up, bag, un, x, c.intermediate, c.hidden);
        for (gate, up) |*g, u| g.* = g.* * ops.sigmoid(g.*) * u;
        try matvecW(out, bag, dn, gate, c.hidden, c.intermediate);
    }
};

test "hy4 host: a token session steps through the synthetic stack" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try config.parse(gpa, @embedFile("config_synthetic_fixture.json"));
    var session = try Session.init(gpa, &c, &bag, 4);
    defer session.deinit();
    for ([_]usize{ 1, 2, 3 }) |tok| {
        const logits = try session.step(tok);
        try std.testing.expectEqual(@as(usize, c.vocab), logits.len);
        for (logits) |v| try std.testing.expect(std.math.isFinite(v));
    }
}

test "hy4 host: identical token sequences give identical logits, different ones do not" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try config.parse(gpa, @embedFile("config_synthetic_fixture.json"));
    const seq = [_]usize{ 3, 5, 7 };
    const other = [_]usize{ 3, 5, 9 };
    var first = try Session.init(gpa, &c, &bag, 4);
    defer first.deinit();
    var second = try Session.init(gpa, &c, &bag, 4);
    defer second.deinit();
    var third = try Session.init(gpa, &c, &bag, 4);
    defer third.deinit();
    for (0..seq.len) |i| {
        const a = try first.step(seq[i]);
        const b = try second.step(seq[i]);
        const d = try third.step(other[i]);
        const a_copy = try gpa.dupe(f32, a);
        defer gpa.free(a_copy);
        try std.testing.expectEqualSlices(f32, a_copy, b);
        if (i == seq.len - 1) try std.testing.expect(!std.mem.eql(f32, a_copy, d));
    }
}
