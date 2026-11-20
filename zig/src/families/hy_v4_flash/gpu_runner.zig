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
    ex: ?cuda.exl3_experts.Experts = null,
    bufs: [200]cuda.memory.DeviceBuffer = undefined,
    nbuf: usize = 0,
    pdev: DevPtrs = undefined,
    streams: cuda.memory.DeviceBuffer = undefined,
    logits_dev: cuda.memory.DeviceBuffer = undefined,
    inv_freq: cuda.memory.DeviceBuffer = undefined,
    inv_freq_swa: cuda.memory.DeviceBuffer = undefined,
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
    expert_scratch: u64 = 0,
    expert_scratch_len: usize = 0,
    router_logits: u64 = 0,
    pick_dev: u64 = 0,
    wts_dev: u64 = 0,
    latent_in: u64 = 0,
    expert_out: u64 = 0,
    routed_out: u64 = 0,
    d0_gate: u64 = 0,
    d0_up: u64 = 0,
    d0_gate_bf: u64 = 0,
    d0_up_bf: u64 = 0,
    d0_act: u64 = 0,
    d0_out: u64 = 0,
    d0_out_bf: u64 = 0,
    host_router: []f32 = &.{},
    host_router_bias: []f32 = &.{},
    host_pick: []i32 = &.{},
    host_wts: []f32 = &.{},
    host_combine: []f32 = &.{},
    host_combine_bf: []u16 = &.{},

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
        const kv_out_max = cfg.max_heads * per;
        const host_full = try r.a.alloc(u16, k * kv_out_max);
        defer r.a.free(host_full);
        const wk_host = try r.a.alloc(u16, cfg.max_heads * nope * k);
        defer r.a.free(wk_host);
        const wv_host = try r.a.alloc(u16, cfg.max_heads * vdim * k);
        defer r.a.free(wv_host);
        for (0..cfg.layers) |li| {
            if (li % 8 == 0) std.debug.print("hy4 runner init: absorb layer {d}\n", .{li});
            const hd = r.p.w[li].heads;
            const kv_out = hd * per;
            var src = cuda.memory.DeviceBuffer{ .d = r.d, .ptr = r.p.w[li].kv_b_w, .len = k * kv_out * 2 };
            // the source is this layer's kv_out rows; host_full is sized for the max.
            try src.download(0, std.mem.sliceAsBytes(host_full)[0 .. k * kv_out * 2]);
            // kv_b.weight is row-major [kv_out, k]: element (o, c) sits at o*k + c.
            for (0..hd) |h| {
                for (0..nope) |i| {
                    const o = h * per + i;
                    for (0..k) |c| wk_host[(h * nope + i) * k + c] = host_full[o * k + c];
                }
                for (0..vdim) |i| {
                    const o = h * per + nope + i;
                    for (0..k) |c| wv_host[(h * vdim + i) * k + c] = host_full[o * k + c];
                }
            }
            const wk_l = try r.a.alloc(u16, hd * nope * k);
            const wv_l = try r.a.alloc(u16, hd * vdim * k);
            @memcpy(wk_l, wk_host[0..hd * nope * k]);
            @memcpy(wv_l, wv_host[0..hd * vdim * k]);
            r.wk[li] = (try r.fromHost(std.mem.sliceAsBytes(wk_l))).ptr;
            r.wv[li] = (try r.fromHost(std.mem.sliceAsBytes(wv_l))).ptr;
            // release each pinned staging copy before the next: 84 large uploads
            // otherwise exhaust the pinned pool and fault the copies.
            try r.s.synchronize();
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
        r.ex = try cuda.exl3_experts.Experts.load(d);
        std.debug.print("hy4 runner init: kernels loaded\n", .{});
        const H = p.cfg.hidden;
        const HC = p.cfg.hc;
        r.host_streams = try a.alloc(u16, H * HC);
        r.host_logits = try a.alloc(f32, p.cfg.vocab);
        r.gate_logits = try r.ptr(rows * p.cfg.max_heads * p.cfg.vdim * 4);
        r.router_logits = try r.ptr(rows * 512 * 4);
        r.pick_dev = try r.ptr(rows * p.cfg.topk * 4);
        r.wts_dev = try r.ptr(rows * p.cfg.topk * 4);
        r.latent_in = try r.ptr(rows * 1536 * 2);
        r.expert_out = try r.ptr(rows * 1536 * 4);
        r.routed_out = try r.ptr(rows * H * 2);
        r.d0_gate = try r.ptr(rows * 13824 * 4);
        r.d0_up = try r.ptr(rows * 13824 * 4);
        r.d0_gate_bf = try r.ptr(rows * 13824 * 2);
        r.d0_up_bf = try r.ptr(rows * 13824 * 2);
        r.d0_act = try r.ptr(rows * 13824 * 2);
        r.d0_out = try r.ptr(rows * H * 4);
        r.d0_out_bf = try r.ptr(rows * H * 2);
        r.host_router = try a.alloc(f32, 512);
        r.host_router_bias = try a.alloc(f32, 512);
        r.host_pick = try a.alloc(i32, p.cfg.topk);
        r.host_wts = try a.alloc(f32, p.cfg.topk);
        r.host_combine = try a.alloc(f32, H);
        r.host_combine_bf = try a.alloc(u16, H);
        const worst: cuda.exl3_experts.Layer = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = 512, .dims = 1536, .width = 1536, .codebook = .mul1 };
        r.expert_scratch_len = try cuda.exl3_experts.scratchBytes(worst, rows, p.cfg.topk);
        r.expert_scratch = try r.ptr(r.expert_scratch_len);
        r.host_kva_f32 = try a.alloc(f32, KW);
        r.host_kva_bf = try a.alloc(u16, KW);
        r.host_ol_f32 = try a.alloc(f32, p.cfg.max_heads * p.cfg.kv_lora);
        r.host_ol_bf = try a.alloc(u16, p.cfg.max_heads * p.cfg.kv_lora);
        r.host_l = try a.alloc(f32, p.cfg.max_heads);
        r.streams = try r.buffer(rows * H * HC * 2);
        r.pdev = .{
            .mixed = try r.ptr(rows * H * 2),
            .xbuf = try r.ptr(rows * H * 2),
            .qn = try r.ptr(rows * p.cfg.q_lora * 2),
            .qfull = try r.ptr(rows * p.cfg.max_heads * (p.cfg.nope + p.cfg.rope) * 2),
            .kva = try r.ptr(rows * (p.cfg.kv_lora + p.cfg.rope) * 4),
            .kvn = try r.ptr(rows * p.cfg.kv_lora * 2),
            .kvb = try r.ptr(rows * p.cfg.max_heads * (p.cfg.nope + p.cfg.vdim) * 2),
            .qa = try r.ptr(rows * p.cfg.max_heads * KW * 2),
            .attn_out = try r.ptr(rows * p.cfg.max_heads * p.cfg.vdim * 2),
            .branch = try r.ptr(rows * H * 2),
            .gu = try r.ptr(rows * 1536 * 2),
            .gu2 = try r.ptr(rows * 1536 * 2),
            .gu_out = try r.ptr(rows * 1536 * 2),
            .ffn = try r.ptr(rows * H * 2),
            .gates = try r.ptr(rows * HC * 2 * 4),
            .final = try r.ptr(rows * H * 2),
        };
        r.logits_dev = try r.buffer(rows * p.cfg.vocab * 4);
        r.inv_freq = try r.fromHost(std.mem.sliceAsBytes(&cuda.mla.ropeFreqs(1e7)));
        r.inv_freq_swa = try r.fromHost(std.mem.sliceAsBytes(&cuda.mla.ropeFreqs(1e4)));
        r.pos_dev = try r.fromHost(std.mem.asBytes(&@as(i32, 0)));
        r.kvabf_dev = try r.buffer(KW * 2);
        r.ol_dev = try r.buffer(rows * p.cfg.max_heads * p.cfg.kv_lora * 2);
        const nch_max = cache_rows / chunk;
        r.parts = .{
            .po = try r.ptr(nch_max * rows * p.cfg.max_heads * p.cfg.kv_lora * 4),
            .pm = try r.ptr(nch_max * rows * p.cfg.max_heads * 4),
            .pl = try r.ptr(nch_max * rows * p.cfg.max_heads * 4),
        };
        r.merged = .{
            .po = try r.ptr(rows * p.cfg.max_heads * p.cfg.kv_lora * 4),
            .pm = try r.ptr(rows * p.cfg.max_heads * 4),
            .pl = try r.ptr(rows * p.cfg.max_heads * 4),
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
        // free each host buffer at its declared element type: a bytes view would
        // free a u16/f32 allocation at align 1 and panic the allocator.
        if (self.host_streams.len > 0) self.a.free(self.host_streams);
        if (self.host_logits.len > 0) self.a.free(self.host_logits);
        if (self.host_kva_f32.len > 0) self.a.free(self.host_kva_f32);
        if (self.host_kva_bf.len > 0) self.a.free(self.host_kva_bf);
        if (self.host_ol_f32.len > 0) self.a.free(self.host_ol_f32);
        if (self.host_ol_bf.len > 0) self.a.free(self.host_ol_bf);
        if (self.host_l.len > 0) self.a.free(self.host_l);
        self.* = undefined;
    }

    /// Device scratch and cache bytes this runner holds (the pack weights belong to the pack).
    pub fn deviceBytes(self: *const Runner) usize {
        var n: usize = 0;
        for (self.bufs[0..self.nbuf]) |b| n += b.len;
        return n;
    }

    /// One token in, host logits out. Synchronous, so lanes sharing a Runner cannot overlap.

    /// Sigmoid + bias, top-k (n_group = 1), normalized and scaled: the released router.
    fn hostRoute(self: *Runner) !void {
        const c = self.p.cfg;
        const n = 512;
        var scores: [512]f32 = undefined;
        for (0..n) |i| {
            const lg = self.host_router[i];
            scores[i] = 1.0 / (1.0 + @exp(-lg));
        }
        var choice: [512]f32 = undefined;
        for (0..n) |i| choice[i] = scores[i] + self.host_router_bias[i];
        var idx: [512]usize = undefined;
        for (0..n) |i| idx[i] = i;
        const SortCtx = struct {
            ch: *const [512]f32,
            fn lt(s: @This(), a: usize, b: usize) bool {
                if (s.ch[a] != s.ch[b]) return s.ch[a] > s.ch[b];
                return a < b;
            }
        };
        std.mem.sort(usize, idx[0..n], SortCtx{ .ch = &choice }, SortCtx.lt);
        var total: f64 = 0;
        for (0..c.topk) |i| {
            self.host_pick[i] = @intCast(idx[i]);
            self.host_wts[i] = scores[idx[i]];
            total += self.host_wts[i];
        }
        const den = total + 1e-20;
        for (0..c.topk) |i| self.host_wts[i] = @floatCast(self.host_wts[i] / den * c.routed_scale);
    }

    pub fn step(self: *Runner, token: usize) ![]const f32 {
        const p = self.p;
        const cfg = p.cfg;
        const H = cfg.hidden;
        const HC = cfg.hc;
        const heads_max = cfg.max_heads;
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
        _ = act;
        const mla = self.mla orelse return error.RunnerNotReady;
        const P = self.pdev;
        const nch = (self.pos + 1 + chunk - 1) / chunk;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(nope + rope)));
        const diag = self.pos == 0;
        for (0..cfg.layers) |li| {
            const L = p.w[li];
            const heads = L.heads; // 32 on DSA layers, 64 on sliding
            _ = heads_max;
            try ihc.collapse(self.s, self.streams.ptr, p.attn_fn[li], p.attn_base[li], p.attn_scale[li], P.mixed, rows, H, HC, cfg.hc_eps, cfg.hc_magnitude, false);
            if (self.pos == 0 and li < 3) {
                const mh = try self.a.alloc(u16, 8);
                defer self.a.free(mh);
                var mb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.mixed, .len = 16 };
                try mb.download(0, std.mem.sliceAsBytes(mh));
                std.debug.print("GPUDUMP L{d} mixed", .{li});
                for (mh) |v| std.debug.print(" {d}", .{@as(f32, @bitCast(@as(u32, v) << 16))});
                std.debug.print("\n", .{});
            }
            try norms.rms(self.s, P.mixed, .bf16, p.input_norm[li], P.xbuf, .bf16, rows, H, cfg.eps);
            try dense.decode(self.s, L.q_a, P.xbuf, .bf16, rows, P.qn, .bf16);
            try norms.rms(self.s, P.qn, .bf16, p.q_a_norm[li], P.qn, .bf16, rows, cfg.q_lora, cfg.eps);
            try dense.decode(self.s, L.q_b, P.qn, .bf16, rows, P.qfull, .bf16);
            if (diag and li <= 2) {
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
            }, rows, heads, self.pos_dev.ptr, (if (heads > 32) self.inv_freq_swa.ptr else self.inv_freq.ptr), 1.0);
            try drows.rows(self.s, P.xbuf, H, L.kv_a_w, P.kva, cfg.kv_lora + rope, rows, cfg.kv_lora + rope, H);
            try norms.rms(self.s, P.kva, .f32, p.kv_a_norm[li], P.kvn, .bf16, rows, cfg.kv_lora, cfg.eps);
            try dense.decode(self.s, L.kv_b, P.kvn, .bf16, rows, P.kvb, .bf16);
            if (diag and li <= 2) {
                var pb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.kvb, .len = 8 };
                var ph: [4]u16 = undefined;
                try pb.download(0, std.mem.sliceAsBytes(&ph));
                std.debug.print("hy4 diag L{d} kvb v0=0x{x}\n", .{ li, ph[0] });
            }
            // Cache row = normalized latent (kvn) plus kv_a's roped tail, all BF16.
            if (diag and li == 0) {
                try self.s.synchronize();
                std.debug.print("hy4 diag L0 pre-cachewrite\n", .{});
            }
            var latent_copy = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.kvabf_dev.ptr, .len = cfg.kv_lora * 2 };
            try latent_copy.copyFrom(0, P.kvn, cfg.kv_lora * 2, self.s.handle);
            if (diag and li == 0) {
                try self.s.synchronize();
                std.debug.print("hy4 diag L0 latent done\n", .{});
            }
            var tail_f32 = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.kva + cfg.kv_lora * 4, .len = rope * 4 };
            try tail_f32.download(0, std.mem.sliceAsBytes(self.host_kva_f32[0..rope]));
            if (diag and li == 0) std.debug.print("hy4 diag L0 tail dl done\n", .{});
            for (0..rope) |i| self.host_kva_bf[cfg.kv_lora + i] = bf16(self.host_kva_f32[i]);
            var tail_bf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.kvabf_dev.ptr + cfg.kv_lora * 2, .len = rope * 2 };
            try tail_bf.upload(0, std.mem.sliceAsBytes(self.host_kva_bf[cfg.kv_lora ..][0..rope]));
            if (diag and li == 0) {
                try self.s.synchronize();
                std.debug.print("hy4 diag L0 tail ul done\n", .{});
            }
            try mla.rope(self.s, .{
                .x = self.kvabf_dev.ptr + cfg.kv_lora * 2,
                .x_row = KW,
                .x_head = rope,
                .out = self.kvabf_dev.ptr + cfg.kv_lora * 2,
                .o_row = KW,
                .o_head = rope,
            }, 1, 1, self.pos_dev.ptr, (if (L.heads > 32) self.inv_freq_swa.ptr else self.inv_freq.ptr), 1.0);
            if (diag and li == 0) {
                try self.s.synchronize();
                std.debug.print("hy4 diag L0 k-rope done\n", .{});
            }
            var slot = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.cache[li].ptr + self.pos * KW * 2, .len = KW * 2 };
            try slot.copyFrom(0, self.kvabf_dev.ptr, KW * 2, self.s.handle);
            if (diag and li == 0) {
                try self.s.synchronize();
                std.debug.print("hy4 diag L0 slot done\n", .{});
            }
            if (diag and li == 2) std.debug.print("hy4 diag L2 cachewrite done\n", .{});
            // Absorb q_nope into the latent columns, then place the roped q tail in 512..575.
            try mla.absorbQ(self.s, P.qfull, nope + rope, self.wk[li], P.qa, KW, rows, heads, nope, cfg.kv_lora);
            if (diag and li == 0) {
                try self.s.synchronize();
                std.debug.print("hy4 diag L0 absorb done\n", .{});
            }
            if (diag and li == 2) std.debug.print("hy4 diag L2 absorb done\n", .{});
            for (0..heads) |h| {
                var tail = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.qa + (h * KW + cfg.kv_lora) * 2, .len = rope * 2 };
                try tail.copyFrom(0, P.qfull + (h * (nope + rope) + nope) * 2, rope * 2, self.s.handle);
            }
            if (diag and li == 2) {
                var pb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.qa, .len = 8 };
                var ph: [4]u16 = undefined;
                try pb.download(0, std.mem.sliceAsBytes(&ph));
                std.debug.print("hy4 diag L2 qa v0=0x{x}\n", .{ph[0]});
            }
            if (diag and li == 0) {
                try self.s.synchronize();
                std.debug.print("hy4 diag L0 tailcopies done\n", .{});
            }
            try mla.chunks(self.s, P.qa, self.cache[li].ptr, self.pos_dev.ptr, self.parts, rows, heads, nch, scale);
            if (diag and li == 0) {
                try self.s.synchronize();
                std.debug.print("hy4 diag L0 chunks done\n", .{});
            }
            // p.sinks[li] is FP32 [heads] in the pack: no dtype conversion.
            if (diag and li == 2) {
                var pb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.parts.po, .len = 8 };
                var ph: [2]f32 = undefined;
                try pb.download(0, std.mem.sliceAsBytes(&ph));
                std.debug.print("hy4 diag L2 parts po0={d:.4}\n", .{ph[0]});
            }
            try mla.mergePartSink(self.s, self.parts, self.merged, rows, heads, nch, p.sinks[li]);
            // Download only this layer's heads: the host arrays are max-heads sized.
            const nout = rows * heads * cfg.kv_lora;
            var merged_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.merged.po, .len = nout * 4 };
            try merged_buf.download(0, std.mem.sliceAsBytes(self.host_ol_f32[0..nout]));
            var l_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.merged.pl, .len = rows * heads * 4 };
            try l_buf.download(0, std.mem.sliceAsBytes(self.host_l[0 .. rows * heads]));
            for (0..nout) |i| {
                const l = self.host_l[i / cfg.kv_lora];
                self.host_ol_bf[i] = bf16(if (l > 1e-6 and std.math.isFinite(l)) self.host_ol_f32[i] / l else self.host_ol_f32[i]);
            }
            if (diag and li <= 2) std.debug.print("hy4 diag L{d} merge l0={d:.4} ol0={d:.4}\n", .{ li, self.host_l[0], self.host_ol_f32[0] });
            try self.ol_dev.upload(0, std.mem.sliceAsBytes(self.host_ol_bf[0..nout]));
            try mla.expandV(self.s, self.ol_dev.ptr, self.wv[li], P.attn_out, rows, heads, cfg.vdim, cfg.kv_lora);
            if (diag and li <= 2) {
                var pb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.attn_out, .len = 8 };
                var ph: [4]u16 = undefined;
                try pb.download(0, std.mem.sliceAsBytes(&ph));
                std.debug.print("hy4 diag L{d} attnout v0=0x{x}\n", .{ li, ph[0] });
            }
            const gate_logits = self.gate_logits;
            try self.drows.?.rows(self.s, P.xbuf, cfg.hidden, L.attn_gate_w, gate_logits, heads * cfg.vdim, rows, heads * cfg.vdim, cfg.hidden);
            try ihc.gateSigmoid(self.s, P.attn_out, gate_logits, rows * heads * cfg.vdim);
            try dense.decode(self.s, L.o, P.attn_out, .bf16, rows, P.branch, .bf16);
            // Attention-side stream update: streams += attn_post_gates * branch (the CPU session does this).
            try ihc.gates(self.s, self.streams.ptr, p.attn_fn[li], p.attn_base[li], p.attn_scale[li], P.gates, P.gates + rows * HC * 4, rows, H, HC, cfg.hc_eps, cfg.hc_magnitude);
            try ihc.update(self.s, self.streams.ptr, P.branch, P.gates + rows * HC * 4, rows, H, HC);
            if (self.pos == 0 and li == 1) {
                const g4 = try self.a.alloc(f32, 8);
                defer self.a.free(g4);
                var gb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.gates, .len = 32 };
                try gb.download(0, std.mem.sliceAsBytes(g4));
                std.debug.print("GPUDUMP L1 ATTNGATES pre0 {d:.4} post0 {d:.4}\n", .{ g4[0], g4[4] });
            }
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
                // Router: gate.weight @ x on device, sigmoid+bias top-k on host.
                try self.drows.?.rows(self.s, P.xbuf, H, L.router_w, self.router_logits, 512, rows, 512, H);
                if (self.pos == 0 and li == 1) {
                    const el = L.experts orelse return error.RunnerNotReady;
                    // read back expert 223's first suh scales (fp16) from the stacked table
                    const sh = try self.a.alloc(u8, 8);
                    defer self.a.free(sh);
                    var sb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = el.suh_g + 223 * 1536 * 2, .len = 8 };
                    try sb.download(0, sh);
                    const s0: f16 = @bitCast(std.mem.readInt(u16, sh[0..2], .little));
                    const s1: f16 = @bitCast(std.mem.readInt(u16, sh[2..4], .little));
                    // read back the first trellis words via the pointer table
                    const pt = try self.a.alloc(u64, 1);
                    defer self.a.free(pt);
                    var pb2 = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = el.gate, .len = 8 };
                    try pb2.download(0, std.mem.sliceAsBytes(pt));
                    var tw = try self.a.alloc(u8, 8);
                    defer self.a.free(tw);
                    var tb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = pt[0], .len = 8 };
                    try tb.download(0, tw);
                    std.debug.print("GPUDUMP L1 exp223 suh {d:.3} {d:.3} trellis_ptr 0x{x} w0=0x{x}\n", .{ s0, s1, pt[0], std.mem.readInt(u32, tw[0..4], .little) });
                }
                var rl = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.router_logits, .len = rows * 512 * 4 };
                try rl.download(0, std.mem.sliceAsBytes(self.host_router));
                var bias_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = L.router_bias, .len = 512 * 4 };
                try bias_buf.download(0, std.mem.sliceAsBytes(self.host_router_bias));
                try self.hostRoute();
                var pick_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.pick_dev, .len = rows * p.cfg.topk * 4 };
                try pick_buf.upload(0, std.mem.sliceAsBytes(self.host_pick));
                // the expert kernel reads fp32 routing weights ([rows, slots])
                var wts_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.wts_dev, .len = rows * p.cfg.topk * 4 };
                try wts_buf.upload(0, std.mem.sliceAsBytes(self.host_wts));
                // Routed: latent_down -> experts -> latent_up.
                try dense.decode(self.s, L.routed_down, P.xbuf, .bf16, rows, self.latent_in, .bf16);
                if (self.pos == 0 and li == 1) {
                    const xb = try self.a.alloc(u16, 3072);
                    defer self.a.free(xb);
                    var xbv = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.xbuf, .len = 3072 * 2 };
                    try xbv.download(0, std.mem.sliceAsBytes(xb));
                    var xs: f64 = 0;
                    for (xb) |v| {
                        const fv = @as(f32, @bitCast(@as(u32, v) << 16));
                        xs += @as(f64, fv) * fv;
                    }
                    const lfull = try self.a.alloc(u16, 1536);
                    defer self.a.free(lfull);
                    var lbv = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.latent_in, .len = 1536 * 2 };
                    try lbv.download(0, std.mem.sliceAsBytes(lfull));
                    std.debug.print("GPUDUMP L1 ORACLE xbuf_rms {d:.4} latent_rms {d:.4} latent0 {d:.4} latent1 {d:.4} xbuf0 {d:.4}\n", .{ @sqrt(xs / 3072.0), blk: {
                        var s: f64 = 0;
                        for (lfull) |v| {
                            const fv = @as(f32, @bitCast(@as(u32, v) << 16));
                            s += fv * fv;
                        }
                        break :blk @sqrt(s / 1536.0);
                    }, @as(f32, @bitCast(@as(u32, lfull[0]) << 16)), @as(f32, @bitCast(@as(u32, lfull[1]) << 16)), @as(f32, @bitCast(@as(u32, xb[0]) << 16)) });
                }
                if (false) {
                    // known-input substitution (diagnostic, disabled)
                    const known = try self.a.alloc(u16, 1536);
                    defer self.a.free(known);
                    var kprng = std.Random.DefaultPrng.init(9);
                    for (known) |*v| {
                        const fv: f32 = kprng.random().float(f32) * 2.0 - 1.0;
                        const bits: u32 = @bitCast(fv);
                        v.* = @truncate(bits >> 16);
                    }
                    var kb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.latent_in, .len = 1536 * 2 };
                    try kb.upload(0, std.mem.sliceAsBytes(known));
                    std.debug.print("GPUDUMP L1 substituted known x\n", .{});
                }
                if (self.pos == 0 and li == 1) {
                    const li4 = try self.a.alloc(u16, 1536);
                    defer self.a.free(li4);
                    var lb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.latent_in, .len = 1536 * 2 };
                    try lb.download(0, std.mem.sliceAsBytes(li4));
                    var lmx: f32 = 0;
                    var lss: f64 = 0;
                    for (li4) |v| {
                        const fv = @as(f32, @bitCast(@as(u32, v) << 16));
                        lmx = @max(lmx, @abs(fv));
                        lss += @as(f64, fv) * fv;
                    }
                    std.debug.print("GPUDUMP L1 latent_in max {d:.4} rms {d:.4}\n", .{ lmx, @sqrt(lss / 1536.0) });
                }
                if (self.pos == 0 and li == 1) {
                    const el0 = L.experts orelse return error.RunnerNotReady;
                    std.debug.print("GPUDUMP L1 LAYER experts={d} dims={d} width={d} cb={d} uk2={d} gus={d} limit={d:.1} act={d}\n", .{ el0.experts, el0.dims, el0.width, @backingInt(el0.codebook), el0.uniform_k2, el0.gu_stride, el0.limit, el0.act_mode });
                    std.debug.print("GPUDUMP L1 PTRS gate=0x{x} k2=0x{x} suhg=0x{x} svhg=0x{x} suhd=0x{x} svhd=0x{x}\n", .{ el0.gate, el0.gate_k2, el0.suh_g, el0.svh_g, el0.suh_d, el0.svh_d });
                    std.debug.print("GPUDUMP L1 SCRATCH base=0x{x} len={d} pick=0x{x} wts=0x{x} x=0x{x} out=0x{x}\n", .{ self.expert_scratch, self.expert_scratch_len, self.pick_dev, self.wts_dev, self.latent_in, self.expert_out });
                }
                const ex = self.ex orelse return error.RunnerNotReady;
                const el = L.experts orelse return error.RunnerNotReady;
                var scb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.expert_scratch, .len = self.expert_scratch_len };
                try scb.fill8(0, self.s.handle); // group()'s counters must start zero every call
                const sc = try cuda.exl3_experts.scratch(self.expert_scratch, el, rows, p.cfg.topk);
                try ex.run(self.s, el, self.latent_in, 1536, .bf16, self.pick_dev, self.wts_dev, rows, self.expert_out, sc);
                try dense.decode(self.s, L.routed_up, self.expert_out, .f32, rows, self.routed_out, .bf16);
                if (self.pos == 0 and li == 1) {
                    const eo = try self.a.alloc(f32, 256);
                    defer self.a.free(eo);
                    var eb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.expert_out, .len = 1024 };
                    try eb.download(0, std.mem.sliceAsBytes(eo));
                    var emx: f32 = 0;
                    for (eo) |v| emx = @max(emx, @abs(v));
                    std.debug.print("GPUDUMP L1 expert_out max256 {d:.4} first {d:.4} picks", .{ emx, eo[0] });
                    for (self.host_pick[0..4]) |pk| std.debug.print(" {d}", .{pk});
                    for (self.host_wts[0..4]) |w| std.debug.print(" w={d:.3}", .{w});
                    std.debug.print("\n", .{});
                }
                // Shared expert.
                try dense.decode(self.s, L.up, P.xbuf, .bf16, rows, P.gu, .bf16);
                try dense.decode(self.s, L.gate, P.xbuf, .bf16, rows, P.gu2, .bf16);
                try ihc.swigluLimit(self.s, P.gu, P.gu2, P.gu_out, rows * 1536, cfg.swiglu_limit);
                try dense.decode(self.s, L.down, P.gu_out, .bf16, rows, P.ffn, .bf16);
                // combine = rmsNorm(routed + shared) with the layer's combine norm.
                var rb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.routed_out, .len = rows * H * 2 };
                const routed_host = try self.a.alloc(u16, H);
                defer self.a.free(routed_host);
                try rb.download(0, std.mem.sliceAsBytes(routed_host));
                var fb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.ffn, .len = rows * H * 2 };
                const shared_host = try self.a.alloc(u16, H);
                defer self.a.free(shared_host);
                try fb.download(0, std.mem.sliceAsBytes(shared_host));
                var cn = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = L.combine_norm, .len = H * 4 };
                const cn_host = try self.a.alloc(f32, H);
                defer self.a.free(cn_host);
                try cn.download(0, std.mem.sliceAsBytes(cn_host));
                var ss: f64 = 0;
                for (0..H) |i| {
                    const rt = @as(f32, @bitCast(@as(u32, routed_host[i]) << 16));
                    const sh = @as(f32, @bitCast(@as(u32, shared_host[i]) << 16));
                    self.host_combine[i] = rt + sh;
                    ss += @as(f64, self.host_combine[i]) * self.host_combine[i];
                }
                const hf: f64 = @floatFromInt(H);
                const inv: f32 = @floatCast(1.0 / @sqrt(ss / hf + cfg.eps));
                for (0..H) |i| self.host_combine_bf[i] = bf16(self.host_combine[i] * inv * cn_host[i]);
                if (self.pos == 0 and li == 1) std.debug.print("GPUDUMP L1 COMBINE ss {d:.2} inv {d:.2} cn0 {d:.4} out0 {d:.4} out1 {d:.4}\n", .{ ss, inv, cn_host[0], @as(f32, @bitCast(@as(u32, self.host_combine_bf[0]) << 16)), @as(f32, @bitCast(@as(u32, self.host_combine_bf[1]) << 16)) });
                try fb.upload(0, std.mem.sliceAsBytes(self.host_combine_bf));
            } else {
                // Layer 0's dense MLP (BF16 weights): drows gives fp32, swiglu wants bf16.
                try self.drows.?.rows(self.s, P.xbuf, H, L.dense_gate, self.d0_gate, 13824, rows, 13824, H);
                try self.drows.?.rows(self.s, P.xbuf, H, L.dense_up, self.d0_up, 13824, rows, 13824, H);
                try ihc.castBf16(self.s, self.d0_gate, self.d0_gate_bf, rows * 13824);
                try ihc.castBf16(self.s, self.d0_up, self.d0_up_bf, rows * 13824);
                try ihc.swigluLimit(self.s, self.d0_gate_bf, self.d0_up_bf, self.d0_act, rows * 13824, cfg.swiglu_limit);
                try self.drows.?.rows(self.s, self.d0_act, 13824, L.dense_down, self.d0_out, H, rows, H, 13824);
                try ihc.castBf16(self.s, self.d0_out, self.d0_out_bf, rows * H);
                var fbn = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.ffn, .len = rows * H * 2 };
                const srcn = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = self.d0_out_bf, .len = rows * H * 2 };
                try fbn.copyFrom(0, srcn.ptr, rows * H * 2, self.s.handle);
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
            if (self.pos == 0 and li == 1) {
                const g4 = try self.a.alloc(f32, 8);
                defer self.a.free(g4);
                var gb = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.gates, .len = 32 };
                try gb.download(0, std.mem.sliceAsBytes(g4));
                std.debug.print("GPUDUMP L1 FFNGATES pre0 {d:.4} post0 {d:.4}\n", .{ g4[0], g4[4] });
            }
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
