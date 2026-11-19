//! Reusable 3bpw Hy4 Flash decode step over a device-resident pack.
//!
//! One Runner owns its stream, per-layer latent KV caches, scratch buffers and
//! logits; step() is synchronous. The absorb/expand weights are recovered from
//! the EXL3 kv_b projection once at init (the pack stores no BF16 copy).
const std = @import("std");
const cuda = @import("cuda");
const pack = @import("hy4pack");

const cache_rows: usize = 4096;
const chunk: usize = 512;
const KW: usize = 576; // 512 latent + 64 roped columns, shared across heads

fn bf16(f: f32) u16 {
    const bits: u32 = @bitCast(f);
    return @truncate(bits >> 16);
}

pub const Runner = struct {
    const rows: usize = 1;

    a: std.mem.Allocator,
    d: *const cuda.Driver,
    p: *pack.Pack,
    s: cuda.Stream = undefined,
    ihc: ?cuda.hy4_ihc.Ihc = null,
    norms: ?cuda.norms.Norms = null,
    dense: ?cuda.exl3.Dense = null,
    drows: ?cuda.dense_rows.DenseRows = null,
    act: ?cuda.act_rows.ActRows = null,
    mla: ?cuda.mla.Mla = null,
    bufs: [200]cuda.memory.DeviceBuffer = undefined,
    nbuf: usize = 0,
    pdev: DevPtrs = undefined,
    streams: cuda.memory.DeviceBuffer = undefined,
    logits_dev: cuda.memory.DeviceBuffer = undefined,
    inv_freq: cuda.memory.DeviceBuffer = undefined,
    pos_dev: cuda.memory.DeviceBuffer = undefined,
    kvabf_dev: cuda.memory.DeviceBuffer = undefined,
    ol_dev: cuda.memory.DeviceBuffer = undefined,
    cache: [42]cuda.memory.DeviceBuffer = undefined,
    wk: [42]u64 = @splat(0),
    wv: [42]u64 = @splat(0),
    parts: cuda.mla.Partials = undefined,
    merged: cuda.mla.Partials = undefined,
    host_streams: []u16 = &.{},
    host_logits: []f32 = &.{},
    host_kva_f32: []f32 = &.{},
    host_kva_bf: []u16 = &.{},
    host_ol_f32: []f32 = &.{},
    host_ol_bf: []u16 = &.{},
    host_l: []f32 = &.{},
    pos: usize = 0,
    gate_logits: u64 = 0,

    const DevPtrs = struct {
        mixed: u64,
        xbuf: u64,
        qn: u64,
        qfull: u64,
        kva: u64,
        kvn: u64,
        kvb: u64,
        qa: u64,
        attn_out: u64,
        branch: u64,
        gu: u64,
        gu2: u64,
        gu_out: u64,
        ffn: u64,
        gates: u64,
        final: u64,
    };

    fn buffer(r: *Runner, len: usize) !cuda.memory.DeviceBuffer {
        if (r.nbuf >= r.bufs.len) return error.TooManyBuffers;
        const b = try cuda.memory.DeviceBuffer.alloc(r.d, len);
        r.bufs[r.nbuf] = b;
        r.nbuf += 1;
        return b;
    }

    fn ptr(r: *Runner, len: usize) !u64 {
        return (try r.buffer(len)).ptr;
    }

    fn fromHost(r: *Runner, bytes: []const u8) !cuda.memory.DeviceBuffer {
        if (r.nbuf >= r.bufs.len) return error.TooManyBuffers;
        const b = try cuda.memory.DeviceBuffer.fromHost(r.d, bytes);
        r.bufs[r.nbuf] = b;
        r.nbuf += 1;
        return b;
    }
    /// Rebuild the absorb/expand matrices from the pack's original BF16 kv_b weight.
    fn absorbWeights(r: *Runner) !void {
        const cfg = r.p.cfg;
        const k = cfg.kv_lora;
        const nope = cfg.nope;
        const vdim = cfg.vdim;
        const per = nope + vdim;
        const kv_out = cfg.heads * per;
        const host_full = try r.a.alloc(u16, k * kv_out);
        defer r.a.free(host_full);
        const wk_host = try r.a.alloc(u16, cfg.heads * nope * k);
        defer r.a.free(wk_host);
        const wv_host = try r.a.alloc(u16, cfg.heads * vdim * k);
        defer r.a.free(wv_host);
        for (0..cfg.layers) |li| {
            if (li % 8 == 0) std.debug.print("hy4 runner init: absorb layer {d}\n", .{li});
            var src = cuda.memory.DeviceBuffer{ .d = r.d, .ptr = r.p.w[li].kv_b_w, .len = k * kv_out * 2 };
            try src.download(0, std.mem.sliceAsBytes(host_full));
            // kv_b.weight is row-major [kv_out, k]: element (o, c) sits at o*k + c.
            for (0..cfg.heads) |h| {
                for (0..nope) |i| {
                    const o = h * per + i;
                    for (0..k) |c| wk_host[(h * nope + i) * k + c] = host_full[o * k + c];
                }
                for (0..vdim) |i| {
                    const o = h * per + nope + i;
                    for (0..k) |c| wv_host[(h * vdim + i) * k + c] = host_full[o * k + c];
                }
            }
            r.wk[li] = (try r.fromHost(std.mem.sliceAsBytes(wk_host))).ptr;
            r.wv[li] = (try r.fromHost(std.mem.sliceAsBytes(wv_host))).ptr;
        }
        try r.s.synchronize();
        std.debug.print("hy4 runner init: absorb weights ready ({d} layers)\n", .{cfg.layers});
    }

    /// Loads the kernels and allocates all per-stream scratch and KV caches once.
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, p: *pack.Pack) !Runner {
        try ctx.makeCurrent();
        std.debug.print("hy4 runner init: kernels loading\n", .{});
        var r: Runner = .{ .a = a, .d = d, .p = p };
        r.s = .{ .d = d, .handle = null };
        errdefer r.deinit();
        r.s = try cuda.Stream.init(d, false); // blocking: host uploads on the default stream order before launches
        r.ihc = try cuda.hy4_ihc.Ihc.load(d);
        r.norms = try cuda.norms.Norms.load(d);
        r.dense = try cuda.exl3.Dense.load(d);
        r.drows = try cuda.dense_rows.DenseRows.load(d);
        r.act = try cuda.act_rows.ActRows.load(d);
        r.mla = try cuda.mla.Mla.load(d);
        std.debug.print("hy4 runner init: kernels loaded\n", .{});
        const H = p.cfg.hidden;
        const HC = p.cfg.hc;
        r.host_streams = try a.alloc(u16, H * HC);
        r.host_logits = try a.alloc(f32, p.cfg.vocab);
        r.gate_logits = try r.ptr(rows * p.cfg.heads * p.cfg.vdim * 4);
        r.host_kva_f32 = try a.alloc(f32, KW);
        r.host_kva_bf = try a.alloc(u16, KW);
        r.host_ol_f32 = try a.alloc(f32, p.cfg.heads * p.cfg.kv_lora);
        r.host_ol_bf = try a.alloc(u16, p.cfg.heads * p.cfg.kv_lora);
        r.host_l = try a.alloc(f32, p.cfg.heads);
        r.streams = try r.buffer(rows * H * HC * 2);
        r.pdev = .{
            .mixed = try r.ptr(rows * H * 2),
            .xbuf = try r.ptr(rows * H * 2),
            .qn = try r.ptr(rows * p.cfg.q_lora * 2),
            .qfull = try r.ptr(rows * p.cfg.heads * (p.cfg.nope + p.cfg.rope) * 2),
            .kva = try r.ptr(rows * (p.cfg.kv_lora + p.cfg.rope) * 4),
            .kvn = try r.ptr(rows * p.cfg.kv_lora * 2),
            .kvb = try r.ptr(rows * p.cfg.heads * (p.cfg.nope + p.cfg.vdim) * 2),
            .qa = try r.ptr(rows * p.cfg.heads * KW * 2),
            .attn_out = try r.ptr(rows * p.cfg.heads * p.cfg.vdim * 2),
            .branch = try r.ptr(rows * H * 2),
            .gu = try r.ptr(rows * 1536 * 2),
            .gu2 = try r.ptr(rows * 1536 * 2),
            .gu_out = try r.ptr(rows * 1536 * 2),
            .ffn = try r.ptr(rows * H * 2),
            .gates = try r.ptr(rows * HC * 2 * 4),
            .final = try r.ptr(rows * H * 2),
        };
        r.logits_dev = try r.buffer(rows * p.cfg.vocab * 4);
        r.inv_freq = try r.fromHost(std.mem.sliceAsBytes(&cuda.mla.ropeFreqs(6e6)));
        r.pos_dev = try r.fromHost(std.mem.asBytes(&@as(i32, 0)));
        r.kvabf_dev = try r.buffer(KW * 2);
        r.ol_dev = try r.buffer(rows * p.cfg.heads * p.cfg.kv_lora * 2);
        const nch_max = cache_rows / chunk;
        r.parts = .{
            .po = try r.ptr(nch_max * rows * p.cfg.heads * p.cfg.kv_lora * 4),
            .pm = try r.ptr(nch_max * rows * p.cfg.heads * 4),
            .pl = try r.ptr(nch_max * rows * p.cfg.heads * 4),
        };
        r.merged = .{
            .po = try r.ptr(rows * p.cfg.heads * p.cfg.kv_lora * 4),
            .pm = try r.ptr(rows * p.cfg.heads * 4),
            .pl = try r.ptr(rows * p.cfg.heads * 4),
        };
        std.debug.print("hy4 runner init: scratch allocated\n", .{});
        for (0..p.cfg.layers) |li| {
            r.cache[li] = try r.buffer(cache_rows * KW * 2);
            try r.cache[li].fill8(0, r.s.handle);
        }
        try r.s.synchronize();
        std.debug.print("hy4 runner init: {d} KV caches allocated\n", .{p.cfg.layers});
        // Scratch for every projection is reserved for 128 rows; the identity decode below
        // uses that same 128-row maximum.
        const dense = &r.dense.?;
        for (0..p.cfg.layers) |li| {
            const L = p.w[li];
            try dense.reserve(r.s, L.q_a);
            try dense.reserve(r.s, L.q_b);
            try dense.reserve(r.s, L.kv_b);
            try dense.reserve(r.s, L.o);
            if (li > 0) {
                try dense.reserve(r.s, L.up);
                try dense.reserve(r.s, L.gate);
                try dense.reserve(r.s, L.down);
            }
        }
        try r.s.synchronize();
        std.debug.print("hy4 runner init: scratch reserved\n", .{});
        try r.absorbWeights();
        std.debug.print("hy4 runner init: ready\n", .{});
        return r;
    }

    pub fn deinit(self: *Runner) void {
        if (self.s.handle != null) self.s.deinit();
        if (self.ihc) |*k| k.unload();
        if (self.norms) |*k| k.unload();
        if (self.dense) |*k| k.unload();
        if (self.drows) |*k| k.unload();
        if (self.act) |*k| k.unload();
        if (self.mla) |*k| k.unload();
        for (self.bufs[0..self.nbuf]) |*b| b.free();
        for ([_][]u8{ std.mem.sliceAsBytes(self.host_streams), std.mem.sliceAsBytes(self.host_logits), std.mem.sliceAsBytes(self.host_kva_f32), std.mem.sliceAsBytes(self.host_kva_bf), std.mem.sliceAsBytes(self.host_ol_f32), std.mem.sliceAsBytes(self.host_ol_bf), std.mem.sliceAsBytes(self.host_l) }) |h| if (h.len > 0) self.a.free(h);
        self.* = undefined;
    }

    /// Device scratch and cache bytes this runner holds (the pack weights belong to the pack).
    pub fn deviceBytes(self: *const Runner) usize {
        var n: usize = 0;
        for (self.bufs[0..self.nbuf]) |b| n += b.len;
        return n;
    }

    /// One token in, host logits out. Synchronous, so lanes sharing a Runner cannot overlap.
    pub fn step(self: *Runner, token: usize) ![]const f32 {
        const p = self.p;
        const cfg = p.cfg;
        const H = cfg.hidden;
        const HC = cfg.hc;
        const heads = cfg.heads;
        const nope = cfg.nope;
        const rope = cfg.rope;
        if (self.pos >= cache_rows) return error.AttentionCacheFull;
        if (token * H * 2 + H * 2 > p.embed_host.len) return error.BadToken;
        try self.pos_dev.upload(0, std.mem.asBytes(&@as(i32, @intCast(self.pos))));
        const emb_src = p.embed_host[token * H * 2 ..][0 .. H * 2];
        const row16 = std.mem.bytesAsSlice(u16, emb_src);
        for (0..HC) |c| @memcpy(self.host_streams[c * H ..][0..H], row16);
        try self.streams.upload(0, std.mem.sliceAsBytes(self.host_streams));

        const ihc = self.ihc orelse return error.RunnerNotReady;
        const norms = self.norms orelse return error.RunnerNotReady;
        var dense = self.dense orelse return error.RunnerNotReady;
        const drows = self.drows orelse return error.RunnerNotReady;
        const act = self.act orelse return error.RunnerNotReady;
        const mla = self.mla orelse return error.RunnerNotReady;
        const P = self.pdev;
        const nch = (self.pos + 1 + chunk - 1) / chunk;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(nope + rope)));
        const diag = self.pos == 0;
        for (0..cfg.layers) |li| {
            const L = p.w[li];
            try ihc.collapse(self.s, self.streams.ptr, p.attn_fn[li], p.attn_base[li], p.attn_scale[li], P.mixed, rows, H, HC, cfg.hc_eps, cfg.hc_magnitude, false);
            try norms.rms(self.s, P.mixed, .bf16, p.input_norm[li], P.xbuf, .bf16, rows, H, cfg.eps);
            try dense.decode(self.s, L.q_a, P.xbuf, .bf16, rows, P.qn, .bf16);
            try norms.rms(self.s, P.qn, .bf16, p.q_a_norm[li], P.qn, .bf16, rows, cfg.q_lora, cfg.eps);
            try dense.decode(self.s, L.q_b, P.qn, .bf16, rows, P.qfull, .bf16);
            if (diag and li <= 1) {
                var pb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.qfull, .len = 8 };
                var ph: [4]u16 = undefined;
                try pb.download(0, std.mem.sliceAsBytes(&ph));
                std.debug.print("hy4 diag L{d} qfull v0=0x{x}\n", .{ li, ph[0] });
            }
            try mla.rope(self.s, .{
                .x = P.qfull + nope * 2,
                .x_row = heads * (nope + rope),
                .x_head = nope + rope,
                .out = P.qfull + nope * 2,
                .o_row = heads * (nope + rope),
                .o_head = nope + rope,
            }, rows, heads, self.pos_dev.ptr, self.inv_freq.ptr, 1.0);
            try drows.rows(self.s, P.xbuf, H, L.kv_a_w, P.kva, cfg.kv_lora + rope, rows, cfg.kv_lora + rope, H);
            try norms.rms(self.s, P.kva, .f32, p.kv_a_norm[li], P.kvn, .bf16, rows, cfg.kv_lora, cfg.eps);
            try dense.decode(self.s, L.kv_b, P.kvn, .bf16, rows, P.kvb, .bf16);
            if (diag and li <= 1) {
                var pb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.kvb, .len = 8 };
                var ph: [4]u16 = undefined;
                try pb.download(0, std.mem.sliceAsBytes(&ph));
                std.debug.print("hy4 diag L{d} kvb v0=0x{x}\n", .{ li, ph[0] });
            }
            // Cache row = normalized latent (kvn) plus kv_a's roped tail, all BF16.
            var latent_copy = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.kvabf_dev.ptr, .len = cfg.kv_lora * 2 };
            try latent_copy.copyFrom(0, P.kvn, cfg.kv_lora * 2, self.s.handle);
            var tail_f32 = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.kva + cfg.kv_lora * 4, .len = rope * 4 };
            try tail_f32.download(0, std.mem.sliceAsBytes(self.host_kva_f32[0..rope]));
            for (0..rope) |i| self.host_kva_bf[cfg.kv_lora + i] = bf16(self.host_kva_f32[i]);
            var tail_bf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.kvabf_dev.ptr + cfg.kv_lora * 2, .len = rope * 2 };
            try tail_bf.upload(0, std.mem.sliceAsBytes(self.host_kva_bf[cfg.kv_lora ..][0..rope]));
            try mla.rope(self.s, .{
                .x = self.kvabf_dev.ptr + cfg.kv_lora * 2,
                .x_row = KW,
                .x_head = rope,
                .out = self.kvabf_dev.ptr + cfg.kv_lora * 2,
                .o_row = KW,
                .o_head = rope,
            }, 1, 1, self.pos_dev.ptr, self.inv_freq.ptr, 1.0);
            var slot = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.cache[li].ptr + self.pos * KW * 2, .len = KW * 2 };
            try slot.copyFrom(0, self.kvabf_dev.ptr, KW * 2, self.s.handle);
            // Absorb q_nope into the latent columns, then place the roped q tail in 512..575.
            try mla.absorbQ(self.s, P.qfull, nope + rope, self.wk[li], P.qa, KW, rows, heads, nope, cfg.kv_lora);
            for (0..heads) |h| {
                var tail = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.qa + (h * KW + cfg.kv_lora) * 2, .len = rope * 2 };
                try tail.copyFrom(0, P.qfull + (h * (nope + rope) + nope) * 2, rope * 2, self.s.handle);
            }
            try mla.chunks(self.s, P.qa, self.cache[li].ptr, self.pos_dev.ptr, self.parts, rows, heads, nch, scale);
            // p.sinks[li] is FP32 [heads] in the pack: no dtype conversion.
            try mla.mergePartSink(self.s, self.parts, self.merged, rows, heads, nch, p.sinks[li]);
            var merged_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.merged.po, .len = rows * heads * cfg.kv_lora * 4 };
            try merged_buf.download(0, std.mem.sliceAsBytes(self.host_ol_f32));
            var l_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.merged.pl, .len = rows * heads * 4 };
            try l_buf.download(0, std.mem.sliceAsBytes(self.host_l));
            for (self.host_ol_bf, self.host_ol_f32, 0..) |*b, v, i| {
                const l = self.host_l[i / cfg.kv_lora];
                b.* = bf16(if (l > 1e-6 and std.math.isFinite(l)) v / l else v);
            }
            if (diag and li <= 1) std.debug.print("hy4 diag L{d} merge l0={d:.4} ol0={d:.4}\n", .{ li, self.host_l[0], self.host_ol_f32[0] });
            try self.ol_dev.upload(0, std.mem.sliceAsBytes(self.host_ol_bf));
            try mla.expandV(self.s, self.ol_dev.ptr, self.wv[li], P.attn_out, rows, heads, cfg.vdim, cfg.kv_lora);
            if (diag and li <= 1) {
                var pb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.attn_out, .len = 8 };
                var ph: [4]u16 = undefined;
                try pb.download(0, std.mem.sliceAsBytes(&ph));
                std.debug.print("hy4 diag L{d} attnout v0=0x{x}\n", .{ li, ph[0] });
            }
            const gate_logits = self.gate_logits;
            try self.drows.?.rows(self.s, P.xbuf, cfg.hidden, L.attn_gate_w, gate_logits, heads * cfg.vdim, rows, heads * cfg.vdim, cfg.hidden);
            try ihc.gateSigmoid(self.s, P.attn_out, gate_logits, rows * heads * cfg.vdim);
            try dense.decode(self.s, L.o, P.attn_out, .bf16, rows, P.branch, .bf16);
            if (diag and li < 4) {
                var pb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.branch, .len = 8 };
                var ph: [4]u16 = undefined;
                try pb.download(0, std.mem.sliceAsBytes(&ph));
                const bad = for (ph) |v| {
                    if (std.math.isNan(@as(f32, @bitCast(@as(u32, v) << 16)))) break true;
                } else false;
                std.debug.print("hy4 diag L{d} branch nan={} v0=0x{x}\n", .{ li, bad, ph[0] });
            }
            try ihc.collapse(self.s, self.streams.ptr, p.ffn_fn[li], p.ffn_base[li], p.ffn_scale[li], P.mixed, rows, H, HC, cfg.hc_eps, cfg.hc_magnitude, false);
            try norms.rms(self.s, P.mixed, .bf16, p.post_attn_norm[li], P.xbuf, .bf16, rows, H, cfg.eps);
            if (li > 0) {
                try dense.decode(self.s, L.up, P.xbuf, .bf16, rows, P.gu, .bf16);
                try dense.decode(self.s, L.gate, P.xbuf, .bf16, rows, P.gu2, .bf16);
                try act.swiglu(self.s, P.gu, 1536, P.gu2, 1536, P.gu_out, null, rows, 1536);
                try dense.decode(self.s, L.down, P.gu_out, .bf16, rows, P.ffn, .bf16);
            } else {
                var ffn_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.ffn, .len = rows * H * 2 };
                try ffn_buf.fill8(0, self.s.handle);
            }
            if (diag and li < 4) {
                var pf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.ffn, .len = 8 };
                var fh: [4]u16 = undefined;
                try pf.download(0, std.mem.sliceAsBytes(&fh));
                const bad = for (fh) |v| {
                    if (std.math.isNan(@as(f32, @bitCast(@as(u32, v) << 16)))) break true;
                } else false;
                std.debug.print("hy4 diag L{d} ffn nan={} v0=0x{x}\n", .{ li, bad, fh[0] });
            }
            try ihc.gates(self.s, self.streams.ptr, p.ffn_fn[li], p.ffn_base[li], p.ffn_scale[li], P.gates, P.gates + rows * HC * 4, rows, H, HC, cfg.hc_eps, cfg.hc_magnitude);
            try ihc.update(self.s, self.streams.ptr, P.ffn, P.gates + rows * HC * 4, rows, H, HC);
        }
        if (diag) {
            var ps = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.streams.ptr, .len = 8 };
            var sh: [4]u16 = undefined;
            try ps.download(0, std.mem.sliceAsBytes(&sh));
            std.debug.print("hy4 diag streams end v0=0x{x}\n", .{sh[0]});
        }
        try ihc.head(self.s, self.streams.ptr, p.head_fn, p.head_base, p.head_scale, P.final, rows, H, HC, cfg.hc_eps);
        try norms.rms(self.s, P.final, .bf16, p.final_norm, P.final, .bf16, rows, H, cfg.eps);
        try drows.rows(self.s, P.final, H, p.lm_head_w, self.logits_dev.ptr, cfg.vocab, rows, cfg.vocab, H);
        try self.logits_dev.downloadAsync(0, std.mem.sliceAsBytes(self.host_logits), self.s.handle);
        try self.s.synchronize();
        self.pos += 1;
        return self.host_logits;
    }
};
