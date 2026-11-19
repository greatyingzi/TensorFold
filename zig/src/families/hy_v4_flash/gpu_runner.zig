//! Reusable 3bpw Hy4 Flash decode step over a device-resident pack.
//!
//! One Runner owns its stream, scratch buffers and logits; step() is synchronous and
//! returns host-readable logits. No KV cache is wired yet: attention is zeroed, so
//! only the first token with an empty cache is meaningful.
const std = @import("std");
const cuda = @import("cuda");
const pack = @import("hy4pack");

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
    bufs: [64]cuda.memory.DeviceBuffer = undefined,
    nbuf: usize = 0,
    pdev: DevPtrs = undefined,
    streams: cuda.memory.DeviceBuffer = undefined,
    logits_dev: cuda.memory.DeviceBuffer = undefined,
    host_streams: []u16 = &.{},
    host_logits: []f32 = &.{},

    const DevPtrs = struct {
        mixed: u64,
        xbuf: u64,
        qn: u64,
        qfull: u64,
        kva: u64,
        kvn: u64,
        kvb: u64,
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

    /// Loads the kernels and allocates all per-stream scratch once; ctx must own the driver.
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, p: *pack.Pack) !Runner {
        try ctx.makeCurrent();
        var r: Runner = .{ .a = a, .d = d, .p = p };
        r.s = .{ .d = d, .handle = null };
        errdefer r.deinit();
        r.s = try cuda.Stream.init(d, false); // blocking: host uploads on the default stream order before launches
        r.ihc = try cuda.hy4_ihc.Ihc.load(d);
        r.norms = try cuda.norms.Norms.load(d);
        r.dense = try cuda.exl3.Dense.load(d);
        r.drows = try cuda.dense_rows.DenseRows.load(d);
        r.act = try cuda.act_rows.ActRows.load(d);
        const H = p.cfg.hidden;
        const HC = p.cfg.hc;
        r.host_streams = try a.alloc(u16, H * HC);
        r.host_logits = try a.alloc(f32, p.cfg.vocab);
        r.streams = try r.buffer(rows * H * HC * 2);
        r.pdev = .{
            .mixed = try r.ptr(rows * H * 2),
            .xbuf = try r.ptr(rows * H * 2),
            .qn = try r.ptr(rows * p.cfg.q_lora * 2),
            .qfull = try r.ptr(rows * p.cfg.heads * (p.cfg.nope + p.cfg.rope) * 2),
            .kva = try r.ptr(rows * (p.cfg.kv_lora + p.cfg.rope) * 4),
            .kvn = try r.ptr(rows * p.cfg.kv_lora * 2),
            .kvb = try r.ptr(rows * p.cfg.heads * (p.cfg.nope + p.cfg.vdim) * 2),
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
        // Grow every scratch buffer once before any launch: dense.reserve frees the old buffer.
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
        return r;
    }

    pub fn deinit(self: *Runner) void {
        if (self.s.handle != null) self.s.deinit();
        if (self.ihc) |*k| k.unload();
        if (self.norms) |*k| k.unload();
        if (self.dense) |*k| k.unload();
        if (self.drows) |*k| k.unload();
        if (self.act) |*k| k.unload();
        for (self.bufs[0..self.nbuf]) |*b| b.free();
        if (self.host_streams.len > 0) self.a.free(self.host_streams);
        if (self.host_logits.len > 0) self.a.free(self.host_logits);
        self.* = undefined;
    }

    /// Device scratch bytes this runner holds (the pack weights belong to the pack).
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
        if (token * H * 2 + H * 2 > p.embed_host.len) return error.BadToken;
        const emb_src = p.embed_host[token * H * 2 ..][0 .. H * 2];
        const row16 = std.mem.bytesAsSlice(u16, emb_src);
        for (0..HC) |c| @memcpy(self.host_streams[c * H ..][0..H], row16);
        try self.streams.upload(0, std.mem.sliceAsBytes(self.host_streams));

        const ihc = self.ihc orelse return error.RunnerNotReady;
        const norms = self.norms orelse return error.RunnerNotReady;
        var dense = self.dense orelse return error.RunnerNotReady;
        const drows = self.drows orelse return error.RunnerNotReady;
        const act = self.act orelse return error.RunnerNotReady;
        const P = self.pdev;
        for (0..cfg.layers) |li| {
            const L = p.w[li];
            try ihc.collapse(self.s, self.streams.ptr, p.attn_fn[li], p.attn_base[li], p.attn_scale[li], P.mixed, rows, H, HC, cfg.hc_eps, cfg.hc_magnitude, false);
            try norms.rms(self.s, P.mixed, .bf16, p.input_norm[li], P.xbuf, .bf16, rows, H, cfg.eps);
            try dense.decode(self.s, L.q_a, P.xbuf, .bf16, rows, P.qn, .bf16);
            try norms.rms(self.s, P.qn, .bf16, p.q_a_norm[li], P.qn, .bf16, rows, cfg.q_lora, cfg.eps);
            try dense.decode(self.s, L.q_b, P.qn, .bf16, rows, P.qfull, .bf16);
            try drows.rows(self.s, P.xbuf, H, L.kv_a_w, P.kva, cfg.kv_lora + cfg.rope, rows, cfg.kv_lora + cfg.rope, H);
            try norms.rms(self.s, P.kva, .f32, p.kv_a_norm[li], P.kvn, .bf16, rows, cfg.kv_lora, cfg.eps);
            try dense.decode(self.s, L.kv_b, P.kvn, .bf16, rows, P.kvb, .bf16);
            // No KV cache yet: zero the attention output so the rest of the stack runs on real weights.
            var attn_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.attn_out, .len = rows * cfg.heads * cfg.vdim * 2 };
            try attn_buf.fill8(0, self.s.handle);
            try act.headGate(self.s, P.attn_out, p.gate_proj[li], cfg.heads * cfg.vdim, P.attn_out, null, rows, cfg.heads, cfg.vdim);
            try dense.decode(self.s, L.o, P.attn_out, .bf16, rows, P.branch, .bf16);
            try ihc.collapse(self.s, self.streams.ptr, p.ffn_fn[li], p.ffn_base[li], p.ffn_scale[li], P.mixed, rows, H, HC, cfg.hc_eps, cfg.hc_magnitude, false);
            try norms.rms(self.s, P.mixed, .bf16, p.post_attn_norm[li], P.xbuf, .bf16, rows, H, cfg.eps);
            if (li > 0) {
                try dense.decode(self.s, L.up, P.xbuf, .bf16, rows, P.gu, .bf16);
                try dense.decode(self.s, L.gate, P.xbuf, .bf16, rows, P.gu2, .bf16);
                try act.swiglu(self.s, P.gu, 1536, P.gu2, 1536, P.gu_out, null, rows, 1536);
                try dense.decode(self.s, L.down, P.gu_out, .bf16, rows, P.ffn, .bf16);
            } else {
                // Layer 0 is dense: its MoE weights do not exist in the pack.
                var ffn_buf = cuda.memory.DeviceBuffer{ .d = self.d, .ptr = P.ffn, .len = rows * H * 2 };
                try ffn_buf.fill8(0, self.s.handle);
            }
            try ihc.gates(self.s, self.streams.ptr, p.ffn_fn[li], p.ffn_base[li], p.ffn_scale[li], P.gates, P.gates + rows * HC * 4, rows, H, HC, cfg.hc_eps, cfg.hc_magnitude);
            try ihc.update(self.s, self.streams.ptr, P.ffn, P.gates + rows * HC * 4, rows, H, HC);
        }
        try ihc.head(self.s, self.streams.ptr, p.head_fn, p.head_base, p.head_scale, P.final, rows, H, HC, cfg.hc_eps);
        try norms.rms(self.s, P.final, .bf16, p.final_norm, P.final, .bf16, rows, H, cfg.eps);
        try drows.rows(self.s, P.final, H, p.lm_head_w, self.logits_dev.ptr, cfg.vocab, rows, cfg.vocab, H);
        try self.logits_dev.downloadAsync(0, std.mem.sliceAsBytes(self.host_logits), self.s.handle);
        try self.s.synchronize();
        return self.host_logits;
    }
};
