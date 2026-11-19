//! One real-weight GPU decode step for Hy4 Flash from the 3bpw pack.
const std = @import("std");
const cuda = @import("cuda");

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
    const rows: usize = 1;
    const dev = struct {
        fn buf(al: std.mem.Allocator, d: *const cuda.Driver, n: usize) !u64 {
            const b = try cuda.memory.DeviceBuffer.alloc(d, n);
            const ptr = b.ptr;
            // keep the lifetime in the arena
            const keep = try al.create(cuda.memory.DeviceBuffer);
            keep.* = b;
            return ptr;
        }
    }.buf;

    const mixed = try dev(a, &driver, rows * H * 2);
    const xbuf = try dev(a, &driver, rows * H * 2);
    const qn = try dev(a, &driver, rows * p.cfg.q_lora * 2);
    const qfull = try dev(a, &driver, rows * p.cfg.heads * (p.cfg.nope + p.cfg.rope) * 2);
    const kva = try dev(a, &driver, rows * (p.cfg.kv_lora + p.cfg.rope) * 4);
    const kvn = try dev(a, &driver, rows * p.cfg.kv_lora * 2);
    const kvb = try dev(a, &driver, rows * p.cfg.heads * (p.cfg.nope + p.cfg.vdim) * 2);
    const attn_out = try dev(a, &driver, rows * p.cfg.heads * p.cfg.vdim * 2);
    const branch = try dev(a, &driver, rows * H * 2);
    const gu = try dev(a, &driver, rows * 1536 * 2);
    const gu2 = try dev(a, &driver, rows * 1536 * 2);
    const gu_out = try dev(a, &driver, rows * 1536 * 2);
    const ffn = try dev(a, &driver, rows * H * 2);
    const final = try dev(a, &driver, rows * H * 2);
    const logits = try dev(a, &driver, rows * p.cfg.vocab * 4);

    std.debug.print("buffers allocated\n", .{});
    // Embedding: four streams all hold the token row. Use dense_rows as a copy-free embed
    // by treating embed_tokens as a [vocab, hidden] bf16 weight and one-hot x is not possible;
    // instead fill the row on host and upload once.
    std.debug.print("embed_host {d} bytes token {d} vocab {d}\n", .{ p.embed_host.len, token, p.cfg.vocab });
    const emb = try a.alloc(u16, H * HC);
    const emb_src = p.embed_host[token * H * 2 ..][0 .. H * 2];
    const row16 = std.mem.bytesAsSlice(u16, emb_src);
    for (0..HC) |c| @memcpy(emb[c * H ..][0..H], row16);
    var streams_buf = try cuda.memory.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(emb));
    defer streams_buf.free();
    const streams = streams_buf.ptr;

    std.debug.print("embed copied\n", .{});
    // Grow every scratch buffer once before any launch: scratch() frees the old buffer,
    // which is unsafe while earlier async decodes are still in flight.
    for (0..p.cfg.layers) |li| {
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
    var t0 = try cuda.Event.init(&driver, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(&driver, true);
    defer t1.deinit();
    try t0.record(s);
    for (0..p.cfg.layers) |li| {
        const L = p.w[li];
        try ihc.collapse(s, streams, p.attn_fn[li], p.attn_base[li], p.attn_scale[li], mixed, rows, H, HC, p.cfg.hc_eps, p.cfg.hc_magnitude, false);
        if (li == 0) std.debug.print("step pre norms1\n", .{});
        try norms.rms(s, mixed, .bf16, p.input_norm[li], xbuf, .bf16, rows, H, p.cfg.eps);
        if (li == 0) std.debug.print("step pre qa\n", .{});
        try dense.decode(s, L.q_a, xbuf, .bf16, rows, qn, .bf16);
        try norms.rms(s, qn, .bf16, p.q_a_norm[li], qn, .bf16, rows, p.cfg.q_lora, p.cfg.eps);
        if (li == 0) std.debug.print("step pre qb\n", .{});
        try dense.decode(s, L.q_b, qn, .bf16, rows, qfull, .bf16);
        if (li == 0) std.debug.print("step pre kva\n", .{});
        try drows.rows(s, xbuf, H, L.kv_a_w, kva, p.cfg.kv_lora + p.cfg.rope, rows, p.cfg.kv_lora + p.cfg.rope, H);
        try norms.rms(s, kva, .f32, p.kv_a_norm[li], kvn, .bf16, rows, p.cfg.kv_lora, p.cfg.eps);
        if (li == 0) std.debug.print("step pre kvb\n", .{});
        try dense.decode(s, L.kv_b, kvn, .bf16, rows, kvb, .bf16);
        if (li == 0) std.debug.print("step pre absorb\n", .{});
        // The real KV cache path is not wired yet: zero the attention output so the rest
        // of the step can be validated on real weights first.
        var attn_buf = cuda.memory.DeviceBuffer{ .d = &driver, .ptr = attn_out, .len = rows * p.cfg.heads * p.cfg.vdim * 2 };
        try attn_buf.fill8(0, s.handle);
        if (li == 0) std.debug.print("step pre gate\n", .{});
        try act.headGate(s, attn_out, p.gate_proj[li], p.cfg.heads * p.cfg.vdim, attn_out, null, rows, p.cfg.heads, p.cfg.vdim);
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
        try ihc.update(s, streams, ffn, p.ffn_base[li], rows, H, HC);
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
