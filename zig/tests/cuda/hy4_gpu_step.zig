//! One real-weight GPU decode step for Hy4 Flash from the 3bpw pack, with its latent KV cache.
const std = @import("std");
const cuda = @import("cuda");

const cache_rows: usize = 4096;
const chunk: usize = 512;
const KW: usize = 576; // 512 latent + 64 roped columns, shared across heads

fn bf16(f: f32) u16 {
    const bits: u32 = @bitCast(f);
    return @truncate(bits >> 16);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.BadArgs;
    const dir = args[1];
    const token: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 120000;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const a = init.arena.allocator();
    const io = init.io;
    const packmod = @import("hy4pack");
    var p = try packmod.load(a, io, &driver, dir);
    defer p.deinit();
    std.debug.print("pack loaded\n", .{});

    var s = try cuda.Stream.init(&driver, false);
    defer s.deinit();
    std.debug.print("load ihc\n", .{});
    var ihc = try cuda.hy4_ihc.Ihc.load(&driver);
    defer ihc.unload();
    std.debug.print("load norms\n", .{});
    var norms = try cuda.norms.Norms.load(&driver);
    defer norms.unload();
    std.debug.print("load dense\n", .{});
    var dense = try cuda.exl3.Dense.load(&driver);
    defer dense.unload();
    std.debug.print("load drows\n", .{});
    var drows = try cuda.dense_rows.DenseRows.load(&driver);
    defer drows.unload();
    std.debug.print("load act\n", .{});
    var act = try cuda.act_rows.ActRows.load(&driver);
    defer act.unload();
    std.debug.print("load experts\n", .{});
    var ex = try cuda.exl3_experts.Experts.load(&driver);
    defer ex.unload();
    std.debug.print("load mla\n", .{});
    var mla = try cuda.mla.Mla.load(&driver);
    defer mla.unload();

    std.debug.print("kernels loaded\n", .{});
    const H = p.cfg.hidden;
    const HC = p.cfg.hc;
    const heads_max = p.cfg.max_heads; // buffers take the max; the step loop uses each layer's own
    const nope = p.cfg.nope;
    const rope = p.cfg.rope;
    const rows: usize = 1;
    const Dev = struct {
        fn buf(al: std.mem.Allocator, d: *const cuda.Driver, n: usize) !u64 {
            const b = try cuda.memory.DeviceBuffer.alloc(d, n);
            const ptr = b.ptr;
            // keep the lifetime in the arena
            const keep = try al.create(cuda.memory.DeviceBuffer);
            keep.* = b;
            return ptr;
        }
        fn copy(al: std.mem.Allocator, d: *const cuda.Driver, bytes: []const u8) !u64 {
            const b = try cuda.memory.DeviceBuffer.fromHost(d, bytes);
            const keep = try al.create(cuda.memory.DeviceBuffer);
            keep.* = b;
            return b.ptr;
        }
    };
    const dev = Dev.buf;
    const copy = Dev.copy;

    const mixed = try dev(a, &driver, rows * H * 2);
    const xbuf = try dev(a, &driver, rows * H * 2);
    var inv_freq = try cuda.memory.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&cuda.mla.ropeFreqs(6e6)));
    defer inv_freq.free();
    var pos_buf = try cuda.memory.DeviceBuffer.fromHost(&driver, std.mem.asBytes(&@as(i32, 0)));
    defer pos_buf.free();
    const qn = try dev(a, &driver, rows * p.cfg.q_lora * 2);
    const qfull = try dev(a, &driver, rows * heads_max * (nope + rope) * 2);
    const kva = try dev(a, &driver, rows * (p.cfg.kv_lora + rope) * 4);
    const kvn = try dev(a, &driver, rows * p.cfg.kv_lora * 2);
    const kvb = try dev(a, &driver, rows * heads_max * (p.cfg.nope + p.cfg.vdim) * 2);
    const qa = try dev(a, &driver, rows * heads_max * KW * 2);
    const kvabf = try dev(a, &driver, KW * 2);
    const ol = try dev(a, &driver, rows * heads_max * p.cfg.kv_lora * 2);
    const nch_max = cache_rows / chunk;
    const pp = cuda.mla.Partials{
        .po = try dev(a, &driver, nch_max * rows * heads_max * p.cfg.kv_lora * 4),
        .pm = try dev(a, &driver, nch_max * rows * heads_max * 4),
        .pl = try dev(a, &driver, nch_max * rows * heads_max * 4),
    };
    const merged = cuda.mla.Partials{
        .po = try dev(a, &driver, rows * heads_max * p.cfg.kv_lora * 4),
        .pm = try dev(a, &driver, rows * heads_max * 4),
        .pl = try dev(a, &driver, rows * heads_max * 4),
    };
    const attn_out = try dev(a, &driver, rows * heads_max * p.cfg.vdim * 2);
    const branch = try dev(a, &driver, rows * H * 2);
    const gu = try dev(a, &driver, rows * 1536 * 2);
    const gu2 = try dev(a, &driver, rows * 1536 * 2);
    const gu_out = try dev(a, &driver, rows * 1536 * 2);
    const ffn = try dev(a, &driver, rows * H * 2);
    const gates = try dev(a, &driver, rows * HC * 2 * 4);
    const final = try dev(a, &driver, rows * H * 2);
    const logits = try dev(a, &driver, rows * p.cfg.vocab * 4);
    var cache: [42]u64 = undefined;
    var wk: [42]u64 = undefined;
    var wv: [42]u64 = undefined;
    const nlayers: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else p.cfg.layers;
    for (0..nlayers) |li| {
        cache[li] = try dev(a, &driver, cache_rows * KW * 2);
        var view = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = cache[li], .len = cache_rows * KW * 2 };
        try view.fill8(0, s.handle);
    }
    std.debug.print("buffers allocated\n", .{});

    // Grow every scratch buffer once before any launch. The identity decode below
    // uses 128-row blocks, the same maximum this reserve covers.
    for (0..nlayers) |li| {
        const L = p.w[li];
        try dense.reserve(s, L.q_a);
        try dense.reserve(s, L.q_b);
        try dense.reserve(s, L.kv_b);
        try dense.reserve(s, L.o);
        if (li > 0) {
            try dense.reserve(s, L.up);
            try dense.reserve(s, L.gate);
            try dense.reserve(s, L.down);
        }
    }
    try s.synchronize();
    std.debug.print("scratch reserved\n", .{});

    // Rebuild the absorb/expand weights from the pack's original BF16 kv_b weight.
    {
        const k = p.cfg.kv_lora;
        const per = nope + p.cfg.vdim;
        const kv_out_max = p.cfg.max_heads * per;
        const host_full = try a.alloc(u16, k * kv_out_max);
        const wk_host = try a.alloc(u16, p.cfg.max_heads * nope * k);
        const wv_host = try a.alloc(u16, p.cfg.max_heads * p.cfg.vdim * k);
        for (0..nlayers) |li| {
            const hd = p.w[li].heads;
            const kv_out = hd * per;
            var src = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = p.w[li].kv_b_w, .len = k * kv_out * 2 };
            try src.download(0, std.mem.sliceAsBytes(host_full)[0 .. k * kv_out * 2]);
            // kv_b.weight is row-major [kv_out, k]: element (o, c) sits at o*k + c.
            for (0..hd) |h| {
                for (0..nope) |i| {
                    const o = h * per + i;
                    for (0..k) |c| wk_host[(h * nope + i) * k + c] = host_full[o * k + c];
                }
                for (0..p.cfg.vdim) |i| {
                    const o = h * per + nope + i;
                    for (0..k) |c| wv_host[(h * p.cfg.vdim + i) * k + c] = host_full[o * k + c];
                }
            }
            const wk_l = try a.alloc(u16, hd * nope * k);
            const wv_l = try a.alloc(u16, hd * p.cfg.vdim * k);
            @memcpy(wk_l, wk_host[0..hd * nope * k]);
            @memcpy(wv_l, wv_host[0..hd * p.cfg.vdim * k]);
            wk[li] = try copy(a, &driver, std.mem.sliceAsBytes(wk_l));
            wv[li] = try copy(a, &driver, std.mem.sliceAsBytes(wv_l));
        }
        try s.synchronize();
    }
    std.debug.print("absorb weights ready\n", .{});

    // Embedding: four streams all hold the token row. Fill on host and upload once.
    std.debug.print("embed_host {d} bytes token {d} vocab {d}\n", .{ p.embed_host.len, token, p.cfg.vocab });
    const emb = try a.alloc(u16, H * HC);
    const emb_src = p.embed_host[token * H * 2 ..][0 .. H * 2];
    const row16 = std.mem.bytesAsSlice(u16, emb_src);
    for (0..HC) |c| @memcpy(emb[c * H ..][0..H], row16);
    var streams_buf = try cuda.memory.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(emb));
    defer streams_buf.free();
    const streams = streams_buf.ptr;
    const host_kva_f32 = try a.alloc(f32, KW);
    const host_kva_bf = try a.alloc(u16, KW);
    const host_ol_f32 = try a.alloc(f32, p.cfg.max_heads * p.cfg.kv_lora);
    const host_ol_bf = try a.alloc(u16, p.cfg.max_heads * p.cfg.kv_lora);
    const host_l = try a.alloc(f32, p.cfg.max_heads);
    std.debug.print("embed copied\n", .{});

    var t0 = try cuda.Event.init(&driver, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(&driver, true);
    defer t1.deinit();
    try t0.record(s);
    const pos: usize = 0; // this CLI runs one step; the serving Runner keeps a counter
    const nch = (pos + 1 + chunk - 1) / chunk;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(nope + rope)));
    for (0..nlayers) |li| {
        const L = p.w[li];
        const heads = L.heads;
        try ihc.collapse(s, streams, p.attn_fn[li], p.attn_base[li], p.attn_scale[li], mixed, rows, H, HC, p.cfg.hc_eps, p.cfg.hc_magnitude, false);
        {
            const mh = try a.alloc(u16, 8);
            defer a.free(mh);
            var mb = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = mixed, .len = 16 };
            try mb.download(0, std.mem.sliceAsBytes(mh[0..8]));
            std.debug.print("GPUDUMP L{d} mixed", .{li});
            for (mh) |v| std.debug.print(" {d}", .{@as(f32, @bitCast(@as(u32, v) << 16))});
            std.debug.print("\n", .{});
        }
        if (li == 0) std.debug.print("step pre norms1\n", .{});
        try norms.rms(s, mixed, .bf16, p.input_norm[li], xbuf, .bf16, rows, H, p.cfg.eps);
        if (li == 0) std.debug.print("step pre qa\n", .{});
        try dense.decode(s, L.q_a, xbuf, .bf16, rows, qn, .bf16);
        try norms.rms(s, qn, .bf16, p.q_a_norm[li], qn, .bf16, rows, p.cfg.q_lora, p.cfg.eps);
        if (li == 0) std.debug.print("step pre qb\n", .{});
        try dense.decode(s, L.q_b, qn, .bf16, rows, qfull, .bf16);
        try mla.rope(s, .{
            .x = qfull + nope * 2,
            .x_row = heads_max * (nope + rope),
            .x_head = nope + rope,
            .out = qfull + nope * 2,
            .o_row = heads_max * (nope + rope),
            .o_head = nope + rope,
        }, rows, heads, pos_buf.ptr, inv_freq.ptr, 1.0);
        if (li == 0) std.debug.print("step pre kva\n", .{});
        try drows.rows(s, xbuf, H, L.kv_a_w, kva, p.cfg.kv_lora + rope, rows, p.cfg.kv_lora + rope, H);
        try norms.rms(s, kva, .f32, p.kv_a_norm[li], kvn, .bf16, rows, p.cfg.kv_lora, p.cfg.eps);
        if (li == 0) std.debug.print("step pre kvb\n", .{});
        try dense.decode(s, L.kv_b, kvn, .bf16, rows, kvb, .bf16);
        // Cache row = normalized latent (kvn) plus kv_a's roped tail, all BF16.
        var latent_copy = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = kvabf, .len = p.cfg.kv_lora * 2 };
        try latent_copy.copyFrom(0, kvn, p.cfg.kv_lora * 2, s.handle);
        var tail_f32 = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = kva + p.cfg.kv_lora * 4, .len = rope * 4 };
        try tail_f32.download(0, std.mem.sliceAsBytes(host_kva_f32[0..rope]));
        for (0..rope) |i| host_kva_bf[p.cfg.kv_lora + i] = bf16(host_kva_f32[i]);
        var tail_bf = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = kvabf + p.cfg.kv_lora * 2, .len = rope * 2 };
        try tail_bf.upload(0, std.mem.sliceAsBytes(host_kva_bf[p.cfg.kv_lora ..][0..rope]));
        try mla.rope(s, .{
            .x = kvabf + p.cfg.kv_lora * 2,
            .x_row = KW,
            .x_head = rope,
            .out = kvabf + p.cfg.kv_lora * 2,
            .o_row = KW,
            .o_head = rope,
        }, 1, 1, pos_buf.ptr, inv_freq.ptr, 1.0);
        var slot = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = cache[li] + pos * KW * 2, .len = KW * 2 };
        try slot.copyFrom(0, kvabf, KW * 2, s.handle);
        if (li == 0) std.debug.print("step pre absorb\n", .{});
        try mla.absorbQ(s, qfull, nope + rope, wk[li], qa, KW, rows, heads, nope, p.cfg.kv_lora);
        for (0..heads) |h| {
            var tail = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = qa + (h * KW + p.cfg.kv_lora) * 2, .len = rope * 2 };
            try tail.copyFrom(0, qfull + (h * (nope + rope) + nope) * 2, rope * 2, s.handle);
        }
        try mla.chunks(s, qa, cache[li], pos_buf.ptr, pp, rows, heads, nch, scale);
        // p.sinks[li] is FP32 [heads] in the pack: no dtype conversion.
        try mla.mergePartSink(s, pp, merged, rows, heads, nch, p.sinks[li]);
        var merged_buf = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = merged.po, .len = rows * heads_max * p.cfg.kv_lora * 4 };
        try merged_buf.download(0, std.mem.sliceAsBytes(host_ol_f32));
        var l_buf = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = merged.pl, .len = rows * heads_max * 4 };
        try l_buf.download(0, std.mem.sliceAsBytes(host_l));
        for (host_ol_bf, host_ol_f32, 0..) |*b, v, i| b.* = bf16(v / host_l[i / p.cfg.kv_lora]);
        var ol_buf = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = ol, .len = rows * heads_max * p.cfg.kv_lora * 2 };
        try ol_buf.upload(0, std.mem.sliceAsBytes(host_ol_bf));
        try mla.expandV(s, ol, wv[li], attn_out, rows, heads, p.cfg.vdim, p.cfg.kv_lora);
        if (li == 0) std.debug.print("step pre gate\n", .{});
        try act.headGate(s, attn_out, p.gate_proj[li], heads * p.cfg.vdim, attn_out, null, rows, heads, p.cfg.vdim);
        if (li == 0) std.debug.print("step pre oproj\n", .{});
        try dense.decode(s, L.o, attn_out, .bf16, rows, branch, .bf16);
        try ihc.collapse(s, streams, p.ffn_fn[li], p.ffn_base[li], p.ffn_scale[li], mixed, rows, H, HC, p.cfg.hc_eps, p.cfg.hc_magnitude, false);
        try norms.rms(s, mixed, .bf16, p.post_attn_norm[li], xbuf, .bf16, rows, H, p.cfg.eps);
        if (li > 0) {
            try dense.decode(s, L.up, xbuf, .bf16, rows, gu, .bf16);
            try dense.decode(s, L.gate, xbuf, .bf16, rows, gu2, .bf16);
            try act.swiglu(s, gu, 1536, gu2, 1536, gu_out, null, rows, 1536);
            try dense.decode(s, L.down, gu_out, .bf16, rows, ffn, .bf16);
        } else {
            // Layer 0 is the dense MLP layer; its MoE weights do not exist in the pack.
            var ffn_buf = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = ffn, .len = rows * H * 2 };
            try ffn_buf.fill8(0, s.handle);
        }
        if (li == 0) std.debug.print("step pre update\n", .{});
        try ihc.gates(s, streams, p.ffn_fn[li], p.ffn_base[li], p.ffn_scale[li], gates, gates + rows * HC * 4, rows, H, HC, p.cfg.hc_eps, p.cfg.hc_magnitude);
        try ihc.update(s, streams, ffn, gates + rows * HC * 4, rows, H, HC);
    }
    try ihc.head(s, streams, p.head_fn, p.head_base, p.head_scale, final, rows, H, HC, p.cfg.hc_eps);
    try norms.rms(s, final, .bf16, p.final_norm, final, .bf16, rows, H, p.cfg.eps);
    try drows.rows(s, final, H, p.lm_head_w, logits, p.cfg.vocab, rows, p.cfg.vocab, H);
    try s.synchronize();
    try t1.record(s);
    try t1.synchronize();
    const ms = try cuda.Event.elapsedMs(t0, t1);
    const host = try a.alloc(f32, p.cfg.vocab);
    var logits_buf = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = logits, .len = p.cfg.vocab * 4 };
    try logits_buf.downloadAsync(0, std.mem.sliceAsBytes(host), s.handle);
    try s.synchronize();
    var best: usize = 0;
    var lo: f32 = std.math.inf(f32);
    var hi: f32 = -std.math.inf(f32);
    for (host, 0..) |v, i| {
        if (v > host[best]) best = i;
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    std.debug.print("hy4 gpu step token={d} -> {d} logits[{d}]={d:.4} min={d:.4} max={d:.4}  {d:.2} ms  {d:.1} tok/s\n", .{ token, best, best, host[best], lo, hi, ms, 1000.0 / ms });
}
