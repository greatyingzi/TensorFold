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


/// Load the whole real pack into device weights and report the totals.
pub fn packCheck(gpu: Gpu, dir: []const u8) !void {
    const pack = @import("hy4pack");
    var p = try pack.load(gpu.gpa, gpu.io, gpu.d, dir);
    defer p.deinit();
    std.debug.print("packcheck: layers={d} dense={d} device_bytes={d} embed={d} lm_head={d}\n", .{ p.cfg.layers, p.w.len, p.up.bytes, p.embed, p.lm_head });
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


/// Upload one real layer's 512 fused gate/up + down experts and time the shared kernel.
pub fn exl3Experts(gpu: Gpu, dir: []const u8, layer: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = gpu.io;
    const sf = @import("core").safetensors;
    const E: usize = 512;
    const dims: usize = 1536; // expert input = latent
    const width: usize = 1536;
    const slots: usize = 16;
    const idx_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const idx_text = try std.Io.Dir.cwd().readFileAlloc(io, idx_path, a, .limited(64 << 20));
    var doc = try std.json.parseFromSlice(std.json.Value, a, idx_text, .{});
    const wm = doc.value.object.get("weight_map").?.object;

    var up = cuda.exl3_upload.Uploader{ .d = gpu.d, .gpa = a, .max_buffers = 4 * E + 16 };
    var gt = try a.alloc(u64, E);
    var dt = try a.alloc(u64, E);
    const gu_suh = try a.alloc(u8, E * dims * 2);
    const gu_svh = try a.alloc(u8, E * 2 * width * 2);
    const d_suh = try a.alloc(u8, E * width * 2);
    const d_svh = try a.alloc(u8, E * dims * 2);
    var opened = std.StringHashMap(*sf.File).init(a);
    for (0..E) |e| {
        const names = [_][]const u8{
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.trellis", .{ layer, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.suh", .{ layer, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.svh", .{ layer, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.trellis", .{ layer, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.suh", .{ layer, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.svh", .{ layer, e }),
        };
        var tensors: [6]?sf.Tensor = undefined;
        for (names, 0..) |nm, ni| {
            const sh = wm.get(nm) orelse return error.MissingExl3Tensor;
            var file = opened.get(sh.string) orelse blk: {
                const path = try std.fs.path.join(a, &.{ dir, sh.string });
                const ptr = try a.create(sf.File);
                ptr.* = try sf.File.open(a, io, path);
                try opened.put(sh.string, ptr);
                break :blk ptr;
            };
            tensors[ni] = file.get(nm) orelse return error.MissingExl3Tensor;
        }
        gt[e] = try up.dev(tensors[0].?.bytes);
        dt[e] = try up.dev(tensors[3].?.bytes);
        @memcpy(gu_suh[e * dims * 2 ..][0 .. dims * 2], tensors[1].?.bytes);
        @memcpy(gu_svh[e * 2 * width * 2 ..][0 .. 2 * width * 2], tensors[2].?.bytes);
        @memcpy(d_suh[e * width * 2 ..][0 .. width * 2], tensors[4].?.bytes);
        @memcpy(d_svh[e * dims * 2 ..][0 .. dims * 2], tensors[5].?.bytes);
    }
    const k2 = try a.alloc(i32, E);
    @memset(k2, 6);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const L: cuda.exl3_experts.Layer = .{
        .gate = try up.dev(std.mem.sliceAsBytes(gt)),
        .up = try up.dev(std.mem.sliceAsBytes(gt)),
        .down = try up.dev(std.mem.sliceAsBytes(dt)),
        .gate_k2 = try up.dev(std.mem.sliceAsBytes(k2)),
        .up_k2 = try up.dev(std.mem.sliceAsBytes(k2)),
        .down_k2 = try up.dev(std.mem.sliceAsBytes(k2)),
        .suh_g = try up.dev(gu_suh),
        .suh_u = try up.dev(gu_suh),
        .svh_g = try up.dev(gu_svh),
        .svh_u = try up.dev(gu_svh),
        .suh_d = try up.dev(d_suh),
        .svh_d = try up.dev(d_svh),
        .experts = E,
        .dims = dims,
        .width = width,
        .codebook = .mul1,
        .uniform_k2 = 6,
        .gu_stride = width / 16,
    };
    const rows: usize = 1;
    const x = try up.dev(try a.alloc(u8, rows * dims * 2));
    const pick = try up.dev(try a.alloc(u8, rows * slots * 4));
    const wts = try up.dev(try a.alloc(u8, rows * slots * 2));
    const out = try up.dev(try a.alloc(u8, rows * dims * 4));
    const sc_bytes = try cuda.exl3_experts.scratchBytes(L, rows, slots);
    const sc = try cuda.exl3_experts.scratch(try up.dev(try a.alloc(u8, sc_bytes)), L, rows, slots);
    var ex = try cuda.exl3_experts.Experts.load(gpu.d);
    defer ex.unload();
    for (0..3) |_| try ex.run(s, L, x, dims, .f16, pick, wts, rows, out, sc);
    try s.synchronize();
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    try t0.record(s);
    for (0..50) |_| try ex.run(s, L, x, dims, .f16, pick, wts, rows, out, sc);
    try t1.record(s);
    try t1.synchronize();
    const ms = try cuda.Event.elapsedMs(t0, t1) / 50.0;
    std.debug.print("exl3experts L{d}: {d} real 3bpw experts, {d:.3} ms/token = {d:.1} tok/s\n", .{ layer, E, ms, 1000.0 / ms });
}


/// Standalone numeric check of the shared expert kernel: one expert, weight 1, vs a CPU dequant matvec.
pub fn expertCheck(gpu: Gpu, dir: []const u8, layer: usize, expert: usize, srcdir: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = gpu.io;
    const sf = @import("core").safetensors;
    const idx_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const idx_text = try std.Io.Dir.cwd().readFileAlloc(io, idx_path, a, .limited(64 << 20));
    var doc = try std.json.parseFromSlice(std.json.Value, a, idx_text, .{});
    const wm = doc.value.object.get("weight_map").?.object;

    const dims: usize = 1536;
    const width: usize = 1536;
    const slots: usize = 16;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();

    // load expert's gate_up + down tensors
    const gu_tr_name = try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.trellis", .{ layer, expert });
    const dn_tr_name = try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.trellis", .{ layer, expert });
    var files = std.StringHashMap(*sf.File).init(a);
    const tget = struct {
        fn get(al: std.mem.Allocator, io2: std.Io, fmap: *std.StringHashMap(*sf.File), wmap: std.json.ObjectMap, d: []const u8, nm: []const u8) !sf.Tensor {
            const sh = wmap.get(nm) orelse return error.MissingExl3Tensor;
            const file = fmap.get(sh.string) orelse blk: {
                const path = try std.fs.path.join(al, &.{ d, sh.string });
                const ptr = try al.create(sf.File);
                ptr.* = try sf.File.open(al, io2, path);
                try fmap.put(sh.string, ptr);
                break :blk ptr;
            };
            return file.get(nm) orelse error.MissingExl3Tensor;
        }
    }.get;
    const gu_tr = try tget(a, io, &files, wm, dir, gu_tr_name);
    const gu_su = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.suh", .{ layer, expert }));
    const gu_sv = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.svh", .{ layer, expert }));
    const dn_tr = try tget(a, io, &files, wm, dir, dn_tr_name);
    const dn_su = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.suh", .{ layer, expert }));
    const dn_sv = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.svh", .{ layer, expert }));
    std.debug.print("expertcheck: gu trellis {any} dn trellis {any}\n", .{ gu_tr.shape[0..3], dn_tr.shape[0..3] });

    // upload as a 1-expert layer
    var up = cuda.exl3_upload.Uploader{ .d = gpu.d, .gpa = a, .max_buffers = 2000 };
    defer up.deinit();
    const gu_suh = try a.alloc(u8, dims * 2);
    @memcpy(gu_suh, gu_su.bytes);
    const gu_svh = try a.alloc(u8, 2 * width * 2);
    @memcpy(gu_svh, gu_sv.bytes);
    const dn_suh = try a.alloc(u8, width * 2);
    @memcpy(dn_suh, dn_su.bytes);
    const dn_svh = try a.alloc(u8, dims * 2);
    @memcpy(dn_svh, dn_sv.bytes);
    const k2 = try a.alloc(i32, 1);
    k2[0] = @intCast(gu_tr.shape[2] / 8);
    const L: cuda.exl3_experts.Layer = .{
        .gate = try up.dev(std.mem.sliceAsBytes(&[_]u64{try up.dev(gu_tr.bytes)})),
        .up = try up.dev(std.mem.sliceAsBytes(&[_]u64{try up.dev(gu_tr.bytes)})),
        .down = try up.dev(std.mem.sliceAsBytes(&[_]u64{try up.dev(dn_tr.bytes)})),
        .gate_k2 = try up.dev(std.mem.sliceAsBytes(k2)),
        .up_k2 = try up.dev(std.mem.sliceAsBytes(k2)),
        .down_k2 = try up.dev(std.mem.sliceAsBytes(k2)),
        .suh_g = try up.dev(gu_suh),
        .suh_u = try up.dev(gu_suh),
        .svh_g = try up.dev(gu_svh),
        .svh_u = try up.dev(gu_svh),
        .suh_d = try up.dev(dn_suh),
        .svh_d = try up.dev(dn_svh),
        .experts = 1,
        .dims = dims,
        .width = width,
        .codebook = .mul1,
        .uniform_k2 = @intCast(gu_tr.shape[2] / 8),
        .gu_stride = width / 16,
        .limit = 10.0,
        .act_mode = 0,
    };
    const rows: usize = 1;
    const xh = try a.alloc(u16, dims);
    var prng = std.Random.DefaultPrng.init(9);
    for (xh) |*v| {
        const f: f32 = prng.random().float(f32) * 2.0 - 1.0;
        const bits: u32 = @bitCast(f);
        v.* = @truncate(bits >> 16);
    }
    const x = try up.dev(std.mem.sliceAsBytes(xh));
    const pick_host = try a.alloc(i32, slots);
    pick_host[0] = 0;
    for (pick_host[1..]) |*v| v.* = 0;
    const pick = try up.dev(std.mem.sliceAsBytes(pick_host));
    const wts_host = try a.alloc(f32, slots);
    wts_host[0] = 1.0;
    for (wts_host[1..]) |*v| v.* = 0.0;
    const wts = try up.dev(std.mem.sliceAsBytes(wts_host));
    const out = try up.dev(try a.alloc(u8, rows * dims * 4));
    const sc_bytes = try cuda.exl3_experts.scratchBytes(L, rows, slots);
    var scb = try cuda.memory.DeviceBuffer.alloc(gpu.d, sc_bytes);
    defer scb.free();
    try scb.fill8(0, s.handle);
    const sc = try cuda.exl3_experts.scratch(scb.ptr, L, rows, slots);
    var ex = try cuda.exl3_experts.Experts.load(gpu.d);
    defer ex.unload();
    try ex.run(s, L, x, dims, .bf16, pick, wts, rows, out, sc);
    try s.synchronize();
    const out_host = try a.alloc(f32, dims);
    var ob = cuda.memory.DeviceBuffer{ .d = gpu.d, .ptr = out, .len = dims * 4 };
    try ob.download(0, std.mem.sliceAsBytes(out_host));
    var mx: f32 = 0;
    for (out_host) |v| mx = @max(mx, @abs(v));
    std.debug.print("expertcheck: GPU out max|v| = {d:.5} first {d:.5} {d:.5}\n", .{ mx, out_host[0], out_host[1] });

    const fmt = @import("core").exl3_format;
    const gu_tr_host = try a.alloc(i16, gu_tr.shape[0] * gu_tr.shape[1] * gu_tr.shape[2]);
    @memcpy(std.mem.sliceAsBytes(gu_tr_host), gu_tr.bytes);
    const suh16 = try a.alloc(f16, dims);
    @memcpy(std.mem.sliceAsBytes(suh16), gu_su.bytes);
    const svh16 = try a.alloc(f16, 2 * width);
    @memcpy(std.mem.sliceAsBytes(svh16), gu_sv.bytes);
    const wq = try a.alloc(f64, 2 * width * dims);
    const scratch = try a.alloc(f64, 2 * 2 * width * dims);
    fmt.dequantize(.{ .halves = @intCast(gu_tr.shape[2] / 8) }, .mul1, gu_tr_host, gu_tr.shape[0], gu_tr.shape[1], suh16, svh16, wq, scratch);
    const xf = try a.alloc(f32, dims);
    for (xf, 0..) |*v, i| v.* = @as(f32, @bitCast(@as(u32, xh[i]) << 16));
    // Authoritative CPU reference: the ORIGINAL BF16 expert weights from the source pack.
    const sidx_path = try std.fs.path.join(a, &.{ srcdir, "model.safetensors.index.json" });
    const sidx_text = try std.Io.Dir.cwd().readFileAlloc(io, sidx_path, a, .limited(64 << 20));
    var sdoc = try std.json.parseFromSlice(std.json.Value, a, sidx_text, .{});
    const swm = sdoc.value.object.get("weight_map").?.object;
    var sfiles = std.StringHashMap(*sf.File).init(a);
    const gu_w = try tget(a, io, &sfiles, swm, srcdir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up_proj", .{layer}));
    const dn_w = try tget(a, io, &sfiles, swm, srcdir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down_proj", .{layer}));
    // gu_w: [E, 3072, 1536] bf16 (rank 3); expert slice at expert*3072*1536.
    std.debug.print("expertcheck src gu rank={d} shape={any} dn rank={d} shape={any}\n", .{ gu_w.rank, gu_w.shape[0..gu_w.rank], dn_w.rank, dn_w.shape[0..dn_w.rank] });
    const gu_all = std.mem.bytesAsSlice(u16, gu_w.bytes);
    const gu_e = gu_all[expert * 2 * width * dims ..][0 .. 2 * width * dims];
    const dn_all = std.mem.bytesAsSlice(u16, dn_w.bytes);
    const dn_e = dn_all[expert * dims * width ..][0 .. dims * width];
    const bf = struct {
        fn f(bits: u16) f32 {
            return @bitCast(@as(u32, bits) << 16);
        }
    }.f;
    const gu_out2 = try a.alloc(f32, 2 * width);
    for (0..2 * width) |n| {
        var acc: f64 = 0;
        for (0..dims) |c| acc += @as(f64, bf(gu_e[n * dims + c])) * xf[c];
        gu_out2[n] = @floatCast(acc);
    }
    const act2 = try a.alloc(f32, width);
    for (0..width) |i| {
        const g = @min(gu_out2[i], 10.0);
        const u = @max(-10.0, @min(gu_out2[width + i], 10.0));
        act2[i] = g / (1.0 + @exp(-g)) * u;
    }
    const ref2 = try a.alloc(f32, dims);
    for (0..dims) |n| {
        var acc: f64 = 0;
        for (0..width) |c| acc += @as(f64, bf(dn_e[n * width + c])) * act2[c];
        ref2[n] = @floatCast(acc);
    }
    var worst2: f32 = 0;
    var mxr: f32 = 0;
    for (0..dims) |i| {
        worst2 = @max(worst2, @abs(out_host[i] - ref2[i]) / @max(1e-4, @abs(ref2[i])));
        mxr = @max(mxr, @abs(ref2[i]));
    }
    std.debug.print("expertcheck BF16: ref max|v| = {d:.5} first {d:.5} {d:.5} worst_rel {d:.4}\n", .{ mxr, ref2[0], ref2[1], worst2 });

    // Multi-expert check: the runner's real picks/weights for L1, vs the weighted BF16 sum.
    const picks = [_]i32{ 223, 395, 382, 497 };
    const ws = [_]f32{ 0.069, 0.064, 0.064, 0.064 };
    const pick_m = try a.alloc(i32, slots);
    @memset(pick_m, 0);
    for (picks, 0..) |pv, i| pick_m[i] = pv;
    var pickm_buf = cuda.memory.DeviceBuffer{ .d = gpu.d, .ptr = pick, .len = slots * 4 };
    try pickm_buf.upload(0, std.mem.sliceAsBytes(pick_m));
    const wts_m = try a.alloc(f32, slots);
    @memset(wts_m, 0);
    for (ws, 0..) |wv, i| wts_m[i] = wv;
    var wtsm_buf = cuda.memory.DeviceBuffer{ .d = gpu.d, .ptr = wts, .len = slots * 4 };
    try wtsm_buf.upload(0, std.mem.sliceAsBytes(wts_m));
    try scb.fill8(0, s.handle);
    try ex.run(s, L, x, dims, .bf16, pick, wts, rows, out, sc);
    try s.synchronize();
    var obm = cuda.memory.DeviceBuffer{ .d = gpu.d, .ptr = out, .len = dims * 4 };
    const out_m = try a.alloc(f32, dims);
    try obm.download(0, std.mem.sliceAsBytes(out_m));
    var gmax: f32 = 0;
    for (out_m) |v| gmax = @max(gmax, @abs(v));
    // weighted BF16 reference over the 4 experts
    const refm = try a.alloc(f32, dims);
    @memset(refm, 0);
    for (picks, 0..) |pe, pi| {
        const ge = gu_all[@as(usize, @intCast(pe)) * 2 * width * dims ..][0 .. 2 * width * dims];
        const de = dn_all[@as(usize, @intCast(pe)) * dims * width ..][0 .. dims * width];
        const go = try a.alloc(f32, 2 * width);
        for (0..2 * width) |n| {
            var acc: f64 = 0;
            for (0..dims) |c| acc += @as(f64, bf(ge[n * dims + c])) * xf[c];
            go[n] = @floatCast(acc);
        }
        const ac = try a.alloc(f32, width);
        for (0..width) |i| {
            const g2 = @min(go[i], 10.0);
            const uu2 = @max(-10.0, @min(go[width + i], 10.0));
            ac[i] = g2 / (1.0 + @exp(-g2)) * uu2;
        }
        for (0..dims) |n| {
            var acc: f64 = 0;
            for (0..width) |c| acc += @as(f64, bf(de[n * width + c])) * ac[c];
            refm[n] += ws[pi] * @as(f32, @floatCast(acc));
        }
    }
    var rmax: f32 = 0;
    for (refm) |v| rmax = @max(rmax, @abs(v));
    std.debug.print("expertcheck MULTI: GPU max {d:.5} ref max {d:.5} first g {d:.5} r {d:.5}\n", .{ gmax, rmax, out_m[0], refm[0] });

    // 4-expert harness: reproduce the runner's multi-expert configuration.
    {
        const NE: usize = 512;
        var gt4 = try a.alloc(u64, NE);
        var dt4 = try a.alloc(u64, NE);
        const gs4 = try a.alloc(u8, NE * dims * 2);
        const gv4 = try a.alloc(u8, NE * 2 * width * 2);
        const ds4 = try a.alloc(u8, NE * width * 2);
        const dv4 = try a.alloc(u8, NE * dims * 2);
        const k24 = try a.alloc(i32, NE);
        for (0..NE) |e| {
            const gu4 = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.trellis", .{ layer, e }));
            const dn4 = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.trellis", .{ layer, e }));
            const su4 = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.suh", .{ layer, e }));
            const sv4 = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.svh", .{ layer, e }));
            const dsn4 = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.suh", .{ layer, e }));
            const dvn4 = try tget(a, io, &files, wm, dir, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.svh", .{ layer, e }));
            gt4[e] = try up.dev(gu4.bytes);
            dt4[e] = try up.dev(dn4.bytes);
            @memcpy(gs4[e * dims * 2 ..][0 .. dims * 2], su4.bytes);
            @memcpy(gv4[e * 2 * width * 2 ..][0 .. 2 * width * 2], sv4.bytes);
            @memcpy(ds4[e * width * 2 ..][0 .. width * 2], dsn4.bytes);
            @memcpy(dv4[e * dims * 2 ..][0 .. dims * 2], dvn4.bytes);
            k24[e] = @intCast(gu4.shape[2] / 8);
        }
        const L4: cuda.exl3_experts.Layer = .{
            .gate = try up.dev(std.mem.sliceAsBytes(gt4)),
            .up = try up.dev(std.mem.sliceAsBytes(gt4)),
            .down = try up.dev(std.mem.sliceAsBytes(dt4)),
            .gate_k2 = try up.dev(std.mem.sliceAsBytes(k24)),
            .up_k2 = try up.dev(std.mem.sliceAsBytes(k24)),
            .down_k2 = try up.dev(std.mem.sliceAsBytes(k24)),
            .suh_g = try up.dev(gs4),
            .suh_u = try up.dev(gs4),
            .svh_g = try up.dev(gv4),
            .svh_u = try up.dev(gv4),
            .suh_d = try up.dev(ds4),
            .svh_d = try up.dev(dv4),
            .experts = NE,
            .dims = dims,
            .width = width,
            .codebook = .mul1,
            .uniform_k2 = @intCast(gu_tr.shape[2] / 8),
            .gu_stride = width / 16,
            .limit = 10.0,
            .act_mode = 0,
        };
        const scb4_bytes = try cuda.exl3_experts.scratchBytes(L4, rows, slots);
        var scb4 = try cuda.memory.DeviceBuffer.alloc(gpu.d, scb4_bytes);
        defer scb4.free();
        try scb4.fill8(0, s.handle);
        const sc4 = try cuda.exl3_experts.scratch(scb4.ptr, L4, rows, slots);
        const p4 = try a.alloc(i32, slots);
        @memset(p4, 0);
        for (picks, 0..) |pv, i| p4[i] = pv;
        try pickm_buf.upload(0, std.mem.sliceAsBytes(p4));
        const w4 = try a.alloc(f32, slots);
        @memset(w4, 0);
        for (ws, 0..) |wv, i| w4[i] = wv;
        try wtsm_buf.upload(0, std.mem.sliceAsBytes(w4));
        try ex.run(s, L4, x, dims, .bf16, pick, wts, rows, out, sc4);
        try s.synchronize();
        try obm.download(0, std.mem.sliceAsBytes(out_m));
        var g4: f32 = 0;
        for (out_m) |v| g4 = @max(g4, @abs(v));
        const r4 = try a.alloc(f32, dims);
        @memset(r4, 0);
        for (0..4) |pe| {
            const ge = gu_all[pe * 2 * width * dims ..][0 .. 2 * width * dims];
            const de = dn_all[pe * dims * width ..][0 .. dims * width];
            const go = try a.alloc(f32, 2 * width);
            for (0..2 * width) |n| {
                var acc: f64 = 0;
                for (0..dims) |c| acc += @as(f64, bf(ge[n * dims + c])) * xf[c];
                go[n] = @floatCast(acc);
            }
            const ac = try a.alloc(f32, width);
            for (0..width) |i| {
                const g2 = @min(go[i], 10.0);
                const uu2 = @max(-10.0, @min(go[width + i], 10.0));
                ac[i] = g2 / (1.0 + @exp(-g2)) * uu2;
            }
            for (0..dims) |n| {
                var acc: f64 = 0;
                for (0..width) |c| acc += @as(f64, bf(de[n * width + c])) * ac[c];
                r4[n] += 0.25 * @as(f32, @floatCast(acc));
            }
        }
        var rr4: f32 = 0;
        for (r4) |v| rr4 = @max(rr4, @abs(v));
        std.debug.print("expertcheck E512: GPU max {d:.5} ref max {d:.5} first g {d:.5} r {d:.5}\n", .{ g4, rr4, out_m[0], r4[0] });
    }

    const gu_out = try a.alloc(f32, 2 * width);
    @memset(gu_out, 0);
    for (0..2 * width) |n| {
        var acc: f64 = 0;
        for (0..dims) |c| acc += wq[c * (2 * width) + n] * xf[c];
        gu_out[n] = @floatCast(acc);
    }
    const act = try a.alloc(f32, width);
    for (0..width) |i| {
        const g = @min(gu_out[i], 10.0);
        const u = @max(-10.0, @min(gu_out[width + i], 10.0));
        act[i] = g / (1.0 + @exp(-g)) * u;
    }
    // down: [dims, width]
    const dn_tr_host = try a.alloc(i16, dn_tr.shape[0] * dn_tr.shape[1] * dn_tr.shape[2]);
    @memcpy(std.mem.sliceAsBytes(dn_tr_host), dn_tr.bytes);
    const dsuh = try a.alloc(f16, width);
    @memcpy(std.mem.sliceAsBytes(dsuh), dn_su.bytes);
    const dsvh = try a.alloc(f16, dims);
    @memcpy(std.mem.sliceAsBytes(dsvh), dn_sv.bytes);
    const wq2 = try a.alloc(f64, dims * width);
    const scratch2 = try a.alloc(f64, 2 * dims * width);
    fmt.dequantize(.{ .halves = @intCast(dn_tr.shape[2] / 8) }, .mul1, dn_tr_host, dn_tr.shape[0], dn_tr.shape[1], dsuh, dsvh, wq2, scratch2);
    const ref = try a.alloc(f32, dims);
    for (0..dims) |n| {
        var acc: f64 = 0;
        for (0..width) |c| acc += wq2[c * dims + n] * act[c];
        ref[n] = @floatCast(acc);
    }
    var worst: f32 = 0;
    for (0..dims) |i| worst = @max(worst, @abs(out_host[i] - ref[i]) / @max(1e-4, @abs(ref[i])));
    std.debug.print("expertcheck: CPU ref max|v| = {d:.5} first {d:.5} {d:.5} worst_rel {d:.4}\n", .{ @abs(ref[0]), ref[0], ref[1], worst });
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
