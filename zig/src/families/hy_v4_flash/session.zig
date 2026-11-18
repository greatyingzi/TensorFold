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
        .i32s => error.ExpectedF32Tensor,
    };
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
    pos: usize = 0,

    pub fn init(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, cap: usize) !Session {
        try graph.validate(c);
        var self = Session{
            .gpa = gpa,
            .c = c,
            .bag = bag,
            .logits = try gpa.alloc(f32, c.vocab),
        };
        errdefer gpa.free(self.logits);
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
        const embed = try f32s(bag, "embed_tokens.weight");
        if (embed.len != c.vocab * c.hidden) return error.BadForwardShape;
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
        self.* = undefined;
    }

    fn expandRope(self: *Session, src: []const f32, heads: usize, rope: usize, dst: []f32) !void {
        const kv_heads = self.c.kv_heads;
        if (src.len != kv_heads * rope or heads % kv_heads != 0 or dst.len != heads * rope) return error.BadAttentionShape;
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
        const embed = try f32s(self.bag, "embed_tokens.weight");
        for (0..hc) |s| @memcpy(self.streams[s], embed[token * h ..][0..h]);

        const mixed = try a.alloc(f32, h);
        const ffn = try a.alloc(f32, h);
        for (0..c.layers) |li| {
            const plan = try graph.layerPlan(c, li);
            const nope = if (plan.kind == .sliding) c.swa_qk_nope else c.qk_nope;
            const rope = if (plan.kind == .sliding) c.swa_qk_rope else c.qk_rope;
            const vdim = if (plan.kind == .sliding) c.swa_v_dim else c.v_dim;
            const heads = c.layer[li].heads;
            const q_out = heads * (2 * nope + rope);
            const kv_a_len = c.kv_lora + c.kv_heads * rope;
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
            const ag = try ihc.gates(gpa, c, self.bag, li, "attn_hc", self.streams[0..hc]);
            defer ihc.gatesDeinit(gpa, ag);
            try ihc.collapse(ag, self.streams[0..hc], mixed);
            const an = try named(a, "layers.{d}.input_layernorm.weight", .{li});
            try ops.rmsNorm(mixed, mixed, try f32s(self.bag, an), c.eps);

            const qa = try a.alloc(f32, c.q_lora);
            const qan = try named(a, "layers.{d}.self_attn.q_a_proj.weight", .{li});
            try ops.matvec(qa, try f32s(self.bag, qan), mixed, c.q_lora, h);
            const qn = try a.alloc(f32, c.q_lora);
            const qnl = try named(a, "layers.{d}.self_attn.q_a_layernorm.weight", .{li});
            try ops.rmsNorm(qn, qa, try f32s(self.bag, qnl), c.eps);
            const q = try a.alloc(f32, q_out);
            const qbn = try named(a, "layers.{d}.self_attn.q_b_proj.weight", .{li});
            try ops.matvec(q, try f32s(self.bag, qbn), qn, q_out, c.q_lora);
            const cos = try a.alloc(f32, rope);
            const sin = try a.alloc(f32, rope);
            try rotary.cosSin(c, plan.kind, @intCast(self.pos), cos, sin);
            for (0..heads) |hh| try rotary.apply(q[hh * (2 * nope + rope) ..][0 .. 2 * nope + rope], cos, sin, rope);
            const q_nope = try a.alloc(f32, heads * nope);
            const q_rope = try a.alloc(f32, heads * rope);
            for (0..heads) |hh| {
                @memcpy(q_nope[hh * nope ..][0..nope], q[hh * (2 * nope + rope) ..][0..nope]);
                @memcpy(q_rope[hh * rope ..][0..rope], q[hh * (2 * nope + rope) + 2 * nope ..][0..rope]);
            }

            const kva = try a.alloc(f32, kv_a_len);
            const kan = try named(a, "layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{li});
            try ops.matvec(kva, try f32s(self.bag, kan), mixed, kv_a_len, h);
            const kvn = try a.alloc(f32, c.kv_lora);
            const kvnl = try named(a, "layers.{d}.self_attn.kv_a_layernorm.weight", .{li});
            try ops.rmsNorm(kvn, kva[0..c.kv_lora], try f32s(self.bag, kvnl), c.eps);
            const kvb = try a.alloc(f32, kv_b_len);
            const kbn = try named(a, "layers.{d}.self_attn.kv_b_proj.weight", .{li});
            try ops.matvec(kvb, try f32s(self.bag, kbn), kvn, kv_b_len, c.kv_lora);
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
                try ops.matvec(tk, try f32s(self.bag, tkn), mixed, c.index_dim, h);
                try ops.rmsNorm(tk, tk, try f32s(self.bag, tknn), c.eps);
                try self.recall[li].addToken(self.bag, li, tk);
                selected = try self.recall[li].select(c, self.bag, li, self.pos, qa, mixed);
                defer if (selected) |s| self.recall[li].deinitResult(s);
            }
            const ctx = try a.alloc(f32, heads * vdim);
            try self.kv[li].attend(gpa, plan.kind, c.window, selected, c.index_ratio, q_nope, q_rope, ctx);
            const on = try named(a, "layers.{d}.self_attn.o_proj.weight", .{li});
            try ops.matvec(ffn, try f32s(self.bag, on), ctx, h, heads * vdim);
            for (0..hc) |s| {
                for (0..h) |i| self.next[s][i] = ag.post[s] * ffn[i] + self.streams[s][i];
            }
            for (0..hc) |s| @memcpy(self.streams[s], self.next[s]);

            const fg = try ihc.gates(gpa, c, self.bag, li, "ffn_hc", self.streams[0..hc]);
            defer ihc.gatesDeinit(gpa, fg);
            try ihc.collapse(fg, self.streams[0..hc], mixed);
            const pn = try named(a, "layers.{d}.post_attention_layernorm.weight", .{li});
            try ops.rmsNorm(mixed, mixed, try f32s(self.bag, pn), c.eps);
            if (plan.moe) try moe.latentMoe(gpa, c, self.bag, li, mixed, ffn) else try denseFfn(gpa, c, self.bag, li, mixed, ffn);
            for (0..hc) |s| {
                for (0..h) |i| self.next[s][i] = fg.post[s] * ffn[i] + self.streams[s][i];
            }
            for (0..hc) |s| @memcpy(self.streams[s], self.next[s]);
        }
        const final = try a.alloc(f32, h);
        try ihc.head(gpa, c, self.bag, self.streams[0..hc], final);
        try ops.rmsNorm(final, final, try f32s(self.bag, "norm.weight"), c.eps);
        try ops.matvec(self.logits, try f32s(self.bag, "lm_head.weight"), final, c.vocab, h);
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
        try ops.matvec(gate, try f32s(bag, gn), x, c.intermediate, c.hidden);
        try ops.matvec(up, try f32s(bag, un), x, c.intermediate, c.hidden);
        for (gate, up) |*g, u| g.* = g.* * ops.sigmoid(g.*) * u;
        try ops.matvec(out, try f32s(bag, dn), gate, c.hidden, c.intermediate);
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
