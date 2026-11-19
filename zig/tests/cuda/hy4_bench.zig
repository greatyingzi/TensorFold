//! Hy4 Flash projection, expert and MLA timings on the GPU.
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
const q_lora = 1024;
const nope = 192;
const vdim = 256;
const n_experts = 512;
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


/// Full 42-layer decode step at Hy4 geometry: random affine-4 weights (kernels are memory-bound).
pub fn forwardBench(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(13);
    var r: Rig = .{ .gpu = gpu, .a = a, .s = s, .rng = prng.random() };
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
    var mla = try cuda.mla.Mla.load(gpu.d);
    defer mla.unload();

    const layers = 42;
    const cache_rows = 4096;
    const heads: usize = 32;
    const lw = 512;
    const rw = 64;

    const P = struct { n: usize, k: usize };
    const proj_specs = [_]P{
        .{ .n = q_lora, .k = hidden },           .{ .n = heads * (nope + rw), .k = q_lora },
        .{ .n = lw + rw, .k = hidden },          .{ .n = heads * (nope + vdim), .k = lw },
        .{ .n = hidden, .k = heads * vdim },     .{ .n = 2 * inter, .k = hidden },
        .{ .n = hidden, .k = inter },            .{ .n = latent, .k = hidden },
        .{ .n = hidden, .k = latent },           .{ .n = n_experts, .k = hidden },
        .{ .n = 2 * inter, .k = latent },        .{ .n = latent, .k = inter },
    };
    var lay: [proj_specs.len][layers]qlinear.Affine4 = undefined;
    var x_in: [proj_specs.len][layers]u64 = undefined;
    var xs_in: [proj_specs.len][layers]u64 = undefined;
    for (proj_specs, 0..) |sp, pi| {
        for (0..layers) |li| {
            const lay0 = qlinear.Affine4.layout(sp.n, sp.k);
            lay[pi][li] = .{ .w = try r.zeros(lay0.words), .s = try r.zeros(lay0.scales), .b = try r.zeros(lay0.scales), .n = sp.n, .k = sp.k, .npad = lay0.npad };
            const xb = try r.bfs(16 * sp.k, 1.0);
            x_in[pi][li] = try r.dev(std.mem.sliceAsBytes(xb));
            const sums = try a.alloc(f32, 16 * sp.k / 64);
            for (0..16 * sp.k / 64) |g| {
                var t: f32 = 0;
                for (0..64) |j| t += bfOfHost(xb[g * 64 + j]);
                sums[g] = t;
            }
            xs_in[pi][li] = try r.dev(std.mem.sliceAsBytes(sums));
        }
    }
    var outs: [proj_specs.len][layers]u64 = undefined;
    for (proj_specs, 0..) |sp, pi| {
        for (0..layers) |li| outs[pi][li] = try r.zeros(16 * sp.n * 2);
    }

    var caches: [layers]u64 = undefined;
    var poss: [layers]u64 = undefined;
    for (0..layers) |li| {
        caches[li] = try r.zeros(cache_rows * (lw + rw) * 2);
        poss[li] = try r.zeros(4);
    }
    const qa = try r.zeros(16 * heads * (lw + rw) * 2);
    const po = try r.zeros(8 * heads * lw * 4);
    const pm = try r.zeros(8 * heads * 4);
    const pl = try r.zeros(8 * heads * 4);
    const merged: cuda.mla.Partials = .{ .po = po, .pm = pm, .pl = pl };
    const ol = try r.zeros(16 * heads * lw * 2);
    const attn_out = try r.zeros(16 * heads * vdim * 2);
    const wv = try r.zeros(heads * vdim * lw * 2);
    const wk = try r.zeros(heads * nope * lw * 2);
    const nch = (cache_rows + 511) / 512;
    const sinks = try r.zeros(heads * 4);
    var expert_out: [2][layers][top_k]u64 = undefined;
    for (0..layers) |li| {
        for (0..top_k) |e| {
            expert_out[0][li][e] = try r.zeros(16 * (2 * inter) * 2);
            expert_out[1][li][e] = try r.zeros(16 * latent * 2);
        }
    }

    for ([_]usize{1}) |rows| {
        var t0 = try cuda.Event.init(gpu.d, true);
        defer t0.deinit();
        var t1 = try cuda.Event.init(gpu.d, true);
        defer t1.deinit();
        try s.synchronize();
        try t0.record(s);
        for (0..10) |_| {
            for (0..layers) |li| {
                try a4.decode(s, x_in[0][li], xs_in[0][li], lay[0][li], outs[0][li], rows);
                try a4.decode(s, x_in[1][li], xs_in[1][li], lay[1][li], outs[1][li], rows);
                try a4.decode(s, x_in[2][li], xs_in[2][li], lay[2][li], outs[2][li], rows);
                try a4.decode(s, x_in[3][li], xs_in[3][li], lay[3][li], outs[3][li], rows);
                try a4.decode(s, x_in[4][li], xs_in[4][li], lay[4][li], outs[4][li], rows);
                try a4.decode(s, x_in[5][li], xs_in[5][li], lay[5][li], outs[5][li], rows);
                try a4.decode(s, x_in[6][li], xs_in[6][li], lay[6][li], outs[6][li], rows);
                try a4.decode(s, x_in[7][li], xs_in[7][li], lay[7][li], outs[7][li], rows);
                try a4.decode(s, x_in[8][li], xs_in[8][li], lay[8][li], outs[8][li], rows);
                try a4.decode(s, x_in[9][li], xs_in[9][li], lay[9][li], outs[9][li], rows);
                const Out = qlinear.Affine4Kernels.Out;
                var stem_outs: [top_k]Out = undefined;
                var down_outs: [top_k]Out = undefined;
                for (0..top_k) |e| {
                    stem_outs[e] = .{ .q = lay[10][li], .out = expert_out[0][li][e] };
                    down_outs[e] = .{ .q = lay[11][li], .out = expert_out[1][li][e] };
                }
                for (0..4) |g| {
                    try a4.gemvMany(s, x_in[10][li], xs_in[10][li], stem_outs[g * 4 ..][0..4], rows);
                    try a4.gemvMany(s, x_in[11][li], xs_in[11][li], down_outs[g * 4 ..][0..4], rows);
                }
                try mla.absorbQ(s, qa, nope, wk, qa, lw + rw, rows, heads, nope, lw);
                try mla.chunks(s, qa, caches[li], poss[li], .{ .po = po, .pm = pm, .pl = pl }, rows, heads, nch, 0.0625);
                try mla.mergePartSink(s, .{ .po = po, .pm = pm, .pl = pl }, merged, rows, heads, nch, sinks);
                try mla.expandV(s, ol, wv, attn_out, rows, heads, vdim, lw);
            }
        }
        try t1.record(s);
        try t1.synchronize();
        const ms = try cuda.Event.elapsedMs(t0, t1);
        std.debug.print("hy4 full-step rows={d}: {d:.2} ms/token  {d:.1} tok/s  (42L, all projections + mla, random affine4)\n", .{ rows, ms / 10.0, 10000.0 / ms });
    }
}


/// Upload one real EXL3 projection from the finished pack and time the shared decode kernel.
/// Usage: hy4-gpu-bench exl3real PACK_DIR PROJECTION (e.g. model.layers.1.self_attn.q_a_proj)
pub fn exl3Real(gpu: Gpu, dir: []const u8, proj: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = gpu.io;
    const sf = @import("core").safetensors;

    const idx_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const idx_text = try std.Io.Dir.cwd().readFileAlloc(io, idx_path, a, .limited(64 << 20));
    var doc = try std.json.parseFromSlice(std.json.Value, a, idx_text, .{});
    const wm = doc.value.object.get("weight_map").?.object;

    const tr_name = try std.fmt.allocPrint(a, "{s}.trellis", .{proj});
    const shard_name = wm.get(tr_name) orelse return error.MissingExl3Tensor;
    if (shard_name != .string) return error.MissingExl3Tensor;
    const shard_path = try std.fs.path.join(a, &.{ dir, shard_name.string });
    var file = try sf.File.open(a, io, shard_path);

    const tr = file.get(tr_name) orelse return error.MissingExl3Tensor;
    const su = file.get(try std.fmt.allocPrint(a, "{s}.suh", .{proj})) orelse return error.MissingExl3Tensor;
    const sv = file.get(try std.fmt.allocPrint(a, "{s}.svh", .{proj})) orelse return error.MissingExl3Tensor;
    // The pack's config declares the codebook; the marker name is only a footprint tag.
    const cb: cuda.exl3.Codebook = .mul1;
    if (file.get(try std.fmt.allocPrint(a, "{s}.mcg", .{proj})) == null and
        file.get(try std.fmt.allocPrint(a, "{s}.mul1", .{proj})) == null and
        file.get(try std.fmt.allocPrint(a, "{s}.inst3", .{proj})) == null) return error.MissingExl3Tensor;

    const k = su.numel();
    const n = sv.numel();
    if (tr.rank != 3 or tr.shape[0] != k / 16 or tr.shape[1] != n / 16 or tr.shape[2] % 8 != 0)
        return error.BadExl3Shape;
    const halves: u32 = @intCast(tr.shape[2] / 8);
    const st = cuda.exl3.strides(k, n, halves, false);
    std.debug.print("exl3real {s}: k={d} n={d} halves={d} codebook={s} words={d} trellis_bytes={d}\n", .{ proj, k, n, halves, @tagName(cb), tr.shape[2], tr.bytes.len });

    var up = cuda.exl3_upload.Uploader{ .d = gpu.d, .gpa = a, .max_buffers = 64 };
    defer up.deinit();
    const w: cuda.exl3.Weight = .{
        .trellis = try up.dev(tr.bytes),
        .suh = try up.dev(su.bytes),
        .svh = try up.dev(sv.bytes),
        .k = @intCast(k),
        .n = @intCast(n),
        .halves = halves,
        .codebook = cb,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };

    const rows: usize = 1;
    const xh = try a.alloc(u16, rows * k);
    @memset(xh, 0x3c00);
    const x = try up.dev(std.mem.sliceAsBytes(xh));
    const y = try up.dev(try a.alloc(u8, rows * n * 2));

    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var dense = try cuda.exl3.Dense.load(gpu.d);
    defer dense.unload();
    try dense.reserve(s, w);
    for (0..3) |_| try dense.decode(s, w, x, .f16, rows, y, .f16);
    try s.synchronize();

    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    try t0.record(s);
    for (0..50) |_| try dense.decode(s, w, x, .f16, rows, y, .f16);
    try t1.record(s);
    try t1.synchronize();
    const ms = try cuda.Event.elapsedMs(t0, t1);
    std.debug.print("exl3real decode: {d:.4} ms/row  {d:.1} tok/s  (real 3bpw {s})\n", .{ ms / 50.0, 50000.0 / ms, proj });
    const yhost = try a.alloc(u16, n);
    var dev_y = try cuda.memory.DeviceBuffer.alloc(gpu.d, n * 2);
    defer dev_y.free();
    try dev_y.copyFrom(0, y, n * 2, s.handle);
    try dev_y.downloadAsync(0, std.mem.sliceAsBytes(yhost), s.handle);
    try s.synchronize();
    // Two CPU references: dequantized EXL3 matrix (loader orientation) and the pack's original BF16.
    const fmt = @import("core").exl3_format;
    const tree = try a.alloc(i16, (k / 16) * (n / 16) * tr.shape[2]);
    @memcpy(std.mem.sliceAsBytes(tree), tr.bytes);
    const suh = try a.alloc(f16, k);
    @memcpy(std.mem.sliceAsBytes(suh), su.bytes);
    const svh = try a.alloc(f16, n);
    @memcpy(std.mem.sliceAsBytes(svh), sv.bytes);
    const ref = try a.alloc(f64, k * n);
    const scratch = try a.alloc(f64, 2 * k * n);
    const fcb: fmt.Codebook = switch (cb) { .inst3 => .inst3, .mcg => .mcg, .mul1 => .mul1 };
    fmt.dequantize(.{ .halves = @intCast(halves) }, fcb, tree, k / 16, n / 16, suh, svh, ref, scratch);
    const wname = try std.fmt.allocPrint(a, "{s}.weight", .{proj});
    const wshard = wm.get(wname) orelse return error.MissingExl3Tensor;
    const wpath = try std.fs.path.join(a, &.{ dir, wshard.string });
    var wfile = try sf.File.open(a, io, wpath);
    const wt = wfile.get(wname) orelse return error.MissingExl3Tensor;
    const wb = std.mem.bytesAsSlice(u16, wt.bytes);
    var worst_q: f64 = 0;
    var worst_b: f64 = 0;
    var finite: usize = 0;
    for (0..n) |j| {
        const got: f64 = @floatCast(@as(f16, @bitCast(yhost[j])));
        var wq: f64 = 0;
        var wbv: f64 = 0;
        for (0..k) |kk| {
            wq += ref[kk * n + j];
            const bits = wb[j * k + kk];
            wbv += @as(f64, @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16))));
        }
        if (std.math.isFinite(got)) finite += 1;
        worst_q = @max(worst_q, @abs(got - wq) / @max(1e-6, @abs(wq)));
        worst_b = @max(worst_b, @abs(got - wbv) / @max(1e-6, @abs(wbv)));
    }
    std.debug.print("exl3real numerics: finite {d}/{d} worst_vs_deq {e:.3} worst_vs_bf16 {e:.3}\n", .{ finite, n, worst_q, worst_b });
}


/// Time every dense EXL3 projection of one real layer from the pack.
pub fn exl3Layer(gpu: Gpu, dir: []const u8, layer: usize, count: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = gpu.io;
    const sf = @import("core").safetensors;
    const P = struct { name: []const u8, k: usize, n: usize };
    const list = [_]P{
        .{ .name = "self_attn.q_a_proj", .k = 3072, .n = 1024 },
        .{ .name = "self_attn.q_b_proj", .k = 1024, .n = 8192 },
        .{ .name = "self_attn.kv_a_proj_with_mqa", .k = 3072, .n = 576 },
        .{ .name = "self_attn.kv_b_proj", .k = 512, .n = 14336 },
        .{ .name = "self_attn.o_proj", .k = 8192, .n = 3072 },
        .{ .name = "mlp.shared_experts.up_proj", .k = 3072, .n = 1536 },
        .{ .name = "mlp.shared_experts.gate_proj", .k = 3072, .n = 1536 },
        .{ .name = "mlp.shared_experts.down_proj", .k = 1536, .n = 3072 },
        .{ .name = "mlp.routed_expert_down_proj", .k = 3072, .n = 1536 },
        .{ .name = "mlp.routed_expert_up_proj", .k = 1536, .n = 3072 },
    };
    const idx_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const idx_text = try std.Io.Dir.cwd().readFileAlloc(io, idx_path, a, .limited(64 << 20));
    var doc = try std.json.parseFromSlice(std.json.Value, a, idx_text, .{});
    const wm = doc.value.object.get("weight_map").?.object;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var dense = try cuda.exl3.Dense.load(gpu.d);
    defer dense.unload();
    var total_ms: f32 = 0;
    for (0..count) |li| {
    for (list) |p| {
        if (p.n % 128 != 0 or (layer + li == 0 and std.mem.startsWith(u8, p.name, "mlp."))) {
            std.debug.print("  L{d} {s}: skipped\n", .{ layer + li, p.name });
            continue;
        }
        const proj = try std.fmt.allocPrint(a, "model.layers.{d}.{s}", .{ layer + li, p.name });
        const tr_name = try std.fmt.allocPrint(a, "{s}.trellis", .{proj});
        const shard = wm.get(tr_name) orelse return error.MissingExl3Tensor;
        const path = try std.fs.path.join(a, &.{ dir, shard.string });
        var file = try sf.File.open(a, io, path);
        const tr = file.get(tr_name) orelse return error.MissingExl3Tensor;
        const su = file.get(try std.fmt.allocPrint(a, "{s}.suh", .{proj})) orelse return error.MissingExl3Tensor;
        const sv = file.get(try std.fmt.allocPrint(a, "{s}.svh", .{proj})) orelse return error.MissingExl3Tensor;
        const cb: cuda.exl3.Codebook = if (file.get(try std.fmt.allocPrint(a, "{s}.mcg", .{proj})) != null) .mcg else .mul1;
        const halves: u32 = @intCast(tr.shape[2] / 8);
        const st = cuda.exl3.strides(p.k, p.n, halves, false);
        var up = cuda.exl3_upload.Uploader{ .d = gpu.d, .gpa = a, .max_buffers = 64 };
        const w: cuda.exl3.Weight = .{
            .trellis = try up.dev(tr.bytes), .suh = try up.dev(su.bytes), .svh = try up.dev(sv.bytes),
            .k = @intCast(p.k), .n = @intCast(p.n), .halves = halves, .codebook = cb,
            .stride_k = st.k, .stride_nb = st.nb,
        };
        const rows: usize = 1;
        const xh = try a.alloc(u16, rows * p.k);
        @memset(xh, 0x3c00);
        const x = try up.dev(std.mem.sliceAsBytes(xh));
        const y = try up.dev(try a.alloc(u8, rows * p.n * 2));
        try dense.reserve(s, w);
        for (0..3) |_| try dense.decode(s, w, x, .f16, rows, y, .f16);
        try s.synchronize();
        var p0 = try cuda.Event.init(gpu.d, true);
        defer p0.deinit();
        var p1 = try cuda.Event.init(gpu.d, true);
        defer p1.deinit();
        try p0.record(s);
        for (0..20) |_| try dense.decode(s, w, x, .f16, rows, y, .f16);
        try p1.record(s);
        try p1.synchronize();
        const ms = try cuda.Event.elapsedMs(p0, p1) / 20.0;
        total_ms += ms;
        std.debug.print("  {s}: {d:.4} ms\n", .{ proj, ms });
    }
    }
    std.debug.print("exl3layer {d}..{d}: {d:.3} ms/token dense only  {d:.1} tok/s  ({d} real 3bpw projections)\n", .{ layer, layer + count - 1, total_ms, 1000.0 / total_ms, list.len * count });
}

fn bfOfHost(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}


/// The real pack through the full GPU step: load, pack, run the assembled forward.
pub fn realForward(gpu: Gpu, dir: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
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
    var mla = try cuda.mla.Mla.load(gpu.d);
    defer mla.unload();

    _ = dir;
    _ = a;
    std.debug.print("realforward: pack loading not yet wired\n", .{});
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

// 16 distinct experts: a token streams its weights instead of re-reading one pair from L2.
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
// The segmented launch: 2 launches a layer instead of 32.
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
// MLA math: absorb 192, chunks over a 4096-key cache, merge, expand 256, gate.
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
