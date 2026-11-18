// ! Hy4 Flash latent-MoE decode on the GPU: what the CPU reference cannot reach. ! ! One decode token touches 41 MoE layers x 16 of 512 experts x two latent projections, ! 4.64 G params of affine-4 weights. This measures those launches on real kernels.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");

const qlinear = cuda.qlinear;
const Gpu = check.Gpu;
const Rig = quant_ref.RigT;

/// Hy4 Flash: hidden 3072, latent 1536, expert inter 1536, 512 experts top 16, 41 MoE layers.
const hidden = 3072;
const latent = 1536;
const inter = 1536;
const top_k = 16;
const moe_layers = 41;
/// mla.cuh chunks are 512 keys.
const chunk = 512;

/// tf affine-4 wants the 64-group sums of x, fp32, in group order.
fn groupSums(a: std.mem.Allocator, x: []const u16, rows: usize, k: usize) ![]f32 {
    const out = try a.alloc(f32, rows * k / 64);
    for (0..rows) |row| {
        for (0..k / 64) |g| {
            var s: f32 = 0;
            for (0..64) |j| s += @floatCast(quant_ref.helpers.bfOf(x[row * k + g * 64 + j]));
            out[row * (k / 64) + g] = s;
        }
    }
    return out;
}

const Proj = struct { q: qlinear.Affine4, x: u64, xs: u64, out: u64, name: []const u8 };

fn make(r: *Rig, name: []const u8, n: usize, k: usize, rows: usize) !Proj {
    const lay = qlinear.Affine4.layout(n, k);
    const q: qlinear.Affine4 = .{
        .w = try r.zeros(lay.words),
        .s = try r.zeros(lay.scales),
        .b = try r.zeros(lay.scales),
        .n = n,
        .k = k,
        .npad = lay.npad,
    };
    const x = try r.bfs(rows * k, 1.0);
    const sums_host = try groupSums(r.a, x, rows, k);
    return .{
        .q = q,
        .x = try r.dev(std.mem.sliceAsBytes(x)),
        .xs = try r.dev(std.mem.sliceAsBytes(sums_host)),
        .out = try r.zeros(rows * n * 4),
        .name = name,
    };
}

fn timeIt(gpu: Gpu, a4: *const qlinear.Affine4Kernels, s: cuda.Stream, ps: []const Proj, rows: usize, reps: usize) !f32 {
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    for (ps) |p| try a4.decode(s, p.x, p.xs, p.q, p.out, rows);
    try s.synchronize();
    try t0.record(s);
    for (0..reps) |_| {
        for (ps) |p| try a4.decode(s, p.x, p.xs, p.q, p.out, rows);
    }
    try t1.record(s);
    try t1.synchronize();
    return cuda.Event.elapsedMs(t0, t1);
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const kk = cuda.kernels;
    const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack };
    var mods: [images.len]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*md| md.unload();
    for (images, &mods) |img, *md| {
        md.* = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var a4 = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, true);
    defer a4.deinit();

// 16 distinct experts so a token streams its weights instead of re-reading one pair from L2: gate_up (latent -> 2*inter) then down (inter -> latent) per expert, as LatentMoE runs them.
    var stems: [top_k]Proj = undefined;
    var downs: [top_k]Proj = undefined;
    for (&stems) |*e| e.* = try make(&r, "expert gate_up", 2 * inter, latent, 16);
    for (&downs) |*e| e.* = try make(&r, "expert down", latent, inter, 16);
    for ([_]usize{ 1, 16 }) |rows| {
        for (stems) |e| try a4.decode(s, e.x, e.xs, e.q, e.out, rows);
        for (downs) |e| try a4.decode(s, e.x, e.xs, e.q, e.out, rows);
        try s.synchronize();
        var t0 = try cuda.Event.init(gpu.d, true);
        defer t0.deinit();
        var t1 = try cuda.Event.init(gpu.d, true);
        defer t1.deinit();
        try t0.record(s);
        for (0..3) |_| {
            for (0..moe_layers) |_| {
                for (stems) |e| try a4.decode(s, e.x, e.xs, e.q, e.out, rows);
                for (downs) |e| try a4.decode(s, e.x, e.xs, e.q, e.out, rows);
            }
        }
        try t1.record(s);
        try t1.synchronize();
        const ms = try cuda.Event.elapsedMs(t0, t1);
        const token = ms / 3.0;
        std.debug.print("hy4 experts rows={d}  {d:.2} ms/token  {d:.1} tok/s  (streaming {d} experts x {d} layers)\n", .{ rows, token, 1000.0 / token, top_k, moe_layers });
    }
// The same 16 experts through the segmented launch: 2 launches a layer instead of 32.
    const Out = qlinear.Affine4Kernels.Out;
    var stem_outs: [top_k]Out = undefined;
    var down_outs: [top_k]Out = undefined;
    for (stems, 0..) |e, i| stem_outs[i] = .{ .q = e.q, .out = e.out };
    for (downs, 0..) |e, i| down_outs[i] = .{ .q = e.q, .out = e.out };
    for ([_]usize{ 1, 16 }) |rows| {
        for (0..4) |g| try a4.gemvMany(s, stems[0].x, stems[0].xs, stem_outs[g * 4 ..][0..4], rows);
        for (0..4) |g| try a4.gemvMany(s, downs[0].x, downs[0].xs, down_outs[g * 4 ..][0..4], rows);
        try s.synchronize();
        var g0 = try cuda.Event.init(gpu.d, true);
        defer g0.deinit();
        var g1 = try cuda.Event.init(gpu.d, true);
        defer g1.deinit();
        try g0.record(s);
        for (0..3) |_| {
            for (0..moe_layers) |_| {
                for (0..4) |g| try a4.gemvMany(s, stems[0].x, stems[0].xs, stem_outs[g * 4 ..][0..4], rows);
                for (0..4) |g| try a4.gemvMany(s, downs[0].x, downs[0].xs, down_outs[g * 4 ..][0..4], rows);
            }
        }
        try g1.record(s);
        try g1.synchronize();
        const gms = try cuda.Event.elapsedMs(g0, g1);
        const gtok = gms / 3.0;
        std.debug.print("hy4 experts grouped rows={d}  {d:.2} ms/token  {d:.1} tok/s  (8 launches a layer)\n", .{ rows, gtok, 1000.0 / gtok });
    }
// Hy4's MLA attention, the math around the projections: absorb 192, chunked latent attention over a 4096-key cache, merge, expand 256, output gate.
    const heads = 64;
    const lw = 512;
    const cache_rows = 4096;
    var mla = try cuda.mla.Mla.load(gpu.d);
    defer mla.unload();
    const q_nope = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * 192, 1.0)));
    const wk = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * 192 * lw, 1.0)));
    const qa = try r.zeros(heads * (lw + 64) * 2);
    const cache = try r.zeros(cache_rows * (lw + 64) * 2);
    const ol = try r.zeros(heads * lw * 2);
    const wv = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * 256 * lw, 1.0)));
    const attn_out = try r.zeros(heads * 256 * 2);
    const gate_w = try r.zeros(heads * 2);
    const pos = try r.zeros(4);
    const nch = (cache_rows + chunk - 1) / chunk;
    const po = try r.zeros(nch * heads * lw * 4);
    const pm = try r.zeros(nch * heads * 4);
    const pl = try r.zeros(nch * heads * 4);
    const parts: cuda.mla.Partials = .{ .po = po, .pm = pm, .pl = pl };
    const merged: cuda.mla.Partials = .{ .po = ol, .pm = pm, .pl = pl };
    for ([_]usize{1}) |rows| {
        try mla.absorbQ(s, q_nope, 192, wk, qa, lw + 64, rows, heads, 192, lw);
        try mla.chunks(s, qa, cache, pos, parts, rows, heads, nch, 0.0417);
        try mla.mergePart(s, parts, merged, rows, heads, nch);
        try mla.expandV(s, ol, wv, attn_out, rows, heads, 256, lw);
        try mla.gate(s, attn_out, gate_w, 256, rows, heads, 256);
        try s.synchronize();
        var a0 = try cuda.Event.init(gpu.d, true);
        defer a0.deinit();
        var a1 = try cuda.Event.init(gpu.d, true);
        defer a1.deinit();
        try a0.record(s);
        for (0..100) |_| {
            try mla.absorbQ(s, q_nope, 192, wk, qa, lw + 64, rows, heads, 192, lw);
            try mla.chunks(s, qa, cache, pos, parts, rows, heads, nch, 0.0417);
            try mla.mergePart(s, parts, merged, rows, heads, nch);
            try mla.expandV(s, ol, wv, attn_out, rows, heads, 256, lw);
            try mla.gate(s, attn_out, gate_w, 256, rows, heads, 256);
        }
        try a1.record(s);
        try a1.synchronize();
        const ams = try cuda.Event.elapsedMs(a0, a1);
        std.debug.print("hy4 mla math rows={d}  {d:.3} ms/layer  {d:.2} ms/token ({d} layers)  cache {d}\n", .{ rows, ams / 100.0, ams / 100.0 * 42.0, 42, cache_rows });
    }
// Attention and dense projections of one token, at Hy4 widths.
    const attn = [_]Proj{
        try make(&r, "q_b", 16384, 1024, 16),
        try make(&r, "kv_b", 28672, 512, 16),
        try make(&r, "o_proj", hidden, 16384, 16),
        try make(&r, "kv_a", 576, hidden, 16),
        try make(&r, "shared gate_up", 2 * inter, hidden, 16),
        try make(&r, "shared down", hidden, inter, 16),
    };
    for ([_]usize{ 1, 16 }) |rows| {
        const ms_total = try timeIt(gpu, &a4, s, &attn, rows, 100);
        const per_layer = ms_total / 100.0 / @as(f32, @floatFromInt(attn.len));
        const token = per_layer * @as(f32, @floatFromInt(attn.len * 42));
        std.debug.print("hy4 attn/dense rows={d}  {d:.1} us/proj  {d:.2} ms/token  {d:.1} tok/s\n", .{ rows, per_layer * 1000.0, token, 1000.0 / token });
    }
}
