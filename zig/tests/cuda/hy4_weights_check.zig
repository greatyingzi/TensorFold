//! The real Q4 pack through the GPU: one expert's affine-4 slice packed and read back
//! against the CPU dequantize-and-dot, so the serving path has a numeric witness.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");

const qlinear = cuda.qlinear;
const Gpu = check.Gpu;
const Rig = quant_ref.RigT;

fn bfOf(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// val = scale[g] * code + bias[g], code the (i mod 8) nibble of word i / 8, g = i / 64.
fn dequant(words: []const u32, scales: []const u16, biases: []const u16, out: []f32) void {
    for (out, 0..) |*o, i| {
        const q: u32 = (words[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
        o.* = bfOf(scales[i / 64]) * @as(f32, @floatFromInt(q)) + bfOf(biases[i / 64]);
    }
}

const Tensor = struct { bytes: []u8, dtype: []const u8, shape: []i64 };

/// One tensor's `expert`-th leading-axis slice out of a safetensors file.
fn readExpert(a: std.mem.Allocator, io: std.Io, path: []const u8, name: []const u8, expert: usize) !Tensor {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var prefix: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &prefix, 0) != prefix.len) return error.BadSafetensors;
    const hlen: u64 = std.mem.readInt(u64, &prefix, .little);
    const header = try a.alloc(u8, @intCast(hlen));
    if (try f.readPositionalAll(io, header, 8) != hlen) return error.BadSafetensors;
    var parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
    defer parsed.deinit();
    const t = parsed.value.object.get(name) orelse return error.NoSuchTensor;
    const dtype = try a.dupe(u8, t.object.get("dtype").?.string);
    const shape_v = t.object.get("shape").?.array.items;
    const shape = try a.alloc(i64, shape_v.len);
    for (shape_v, 0..) |v, i| shape[i] = v.integer;
    const offs = t.object.get("data_offsets").?.array.items;
    const start: u64 = @intCast(offs[0].integer);
    const per_elem: usize = if (std.mem.eql(u8, dtype, "U32")) 4 else 2;
    var per: usize = per_elem;
    for (shape[1..]) |d| per *= @intCast(d);
    const out = try a.alloc(u8, per);
    if (try f.readPositionalAll(io, out, 8 + hlen + start + expert * per) != per) return error.BadSafetensors;
    return .{ .bytes = out, .dtype = dtype, .shape = shape };
}
pub fn run(gpu: Gpu, dir: []const u8) !void {
    const io = gpu.io;
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(11);
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

    const n = 1536;
    const k = 1536;
    var path_buf: [512]u8 = undefined;
    const file = try std.fmt.bufPrint(&path_buf, "{s}/hy4-q4-00001-of-00082.safetensors", .{dir});
    const base = "model.layers.1.mlp.experts.down_proj";
    const codes = try readExpert(a, io, file, base ++ ".weight", 0);
    const scales = try readExpert(a, io, file, base ++ ".scales", 0);
    const biases = try readExpert(a, io, file, base ++ ".biases", 0);
    std.debug.print("down_proj expert 0: codes {d} B ({s}), scales {d} B, biases {d} B\n", .{ codes.bytes.len, codes.dtype, scales.bytes.len, biases.bytes.len });

    const x = try a.alloc(f32, k);
    for (x) |*v| v.* = bfOf(std.math.lossyCast(u16, bits: {
        const f: f32 = r.rng.float(f32) * 2.0 - 1.0;
        const u: u32 = @bitCast(f);
        break :bits @as(u16, @truncate((u +% 0x8000) >> 16));
    }));

    const w32 = try a.alloc(u32, n * k / 8);
    const sc = try a.alloc(u16, n * k / 64);
    const bi = try a.alloc(u16, n * k / 64);
    for (w32, 0..) |*v, i| v.* = std.mem.readInt(u32, codes.bytes[i * 4 ..][0..4], .little);
    for (sc, 0..) |*v, i| v.* = std.mem.readInt(u16, scales.bytes[i * 2 ..][0..2], .little);
    for (bi, 0..) |*v, i| v.* = std.mem.readInt(u16, biases.bytes[i * 2 ..][0..2], .little);
    const wf = try a.alloc(f32, n * k);
    dequant(w32, sc, bi, wf);
    const y_cpu = try a.alloc(f32, n);
    for (0..n) |row| {
        var sum: f64 = 0;
        for (0..k) |c| sum += @as(f64, wf[row * k + c]) * @as(f64, x[c]);
        y_cpu[row] = @floatCast(sum);
    }

    const lay = qlinear.Affine4.layout(n, k);
    const q: qlinear.Affine4 = .{
        .w = try r.zeros(lay.words),
        .s = try r.zeros(lay.scales),
        .b = try r.zeros(lay.scales),
        .n = n,
        .k = k,
        .npad = lay.npad,
    };
    try a4.pack(s, try r.dev(codes.bytes), try r.dev(scales.bytes), try r.dev(biases.bytes), q);
    const xb = try a.alloc(u16, k);
    for (x, 0..) |v, i| xb[i] = @truncate(@as(u32, @bitCast(v)) >> 16);
    const dx = try r.dev(std.mem.sliceAsBytes(xb));
    const sums_host = try a.alloc(f32, k / 64);
    for (0..k / 64) |g| {
        var t: f32 = 0;
        for (0..64) |j| t += x[g * 64 + j];
        sums_host[g] = t;
    }
    const dxs = try r.dev(std.mem.sliceAsBytes(sums_host));

    // decode() and cluster() write out (rows, n) bf16: two bytes an output, not four.
    const out_bytes = n * 2;
    const dy = try r.zeros(out_bytes);
    try a4.decode(s, dx, dxs, q, dy, 1);
    const raw = try r.back(dy, out_bytes);
    const y_gpu = try a.alloc(f32, n);
    for (0..n) |i| y_gpu[i] = bfOf(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));

    var worst: f32 = 0;
    var worst_row: usize = 0;
    var mag: f32 = 0;
    var nans: usize = 0;
    for (y_cpu, y_gpu, 0..) |c, g, i| {
        if (!std.math.isFinite(g) or g == 0) nans += 1;
        mag = @max(mag, @abs(c));
        const d = @abs(c - g) / @max(@abs(c), 1e-6);
        if (d > worst) {
            worst = d;
            worst_row = i;
        }
    }
    std.debug.print("expert down_proj: |y| max {d:.4}  worst rel {e:.3} at row {d}  zero or non-finite {d}\n", .{ mag, worst, worst_row, nans });
    std.debug.print("  cpu {d:.6}  gpu {d:.6}\n", .{ y_cpu[worst_row], y_gpu[worst_row] });
    std.debug.print("  cpu[0..4] {d:.5} {d:.5} {d:.5} {d:.5}\n", .{ y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3] });
    std.debug.print("  gpu[0..4] {d:.5} {d:.5} {d:.5} {d:.5}\n", .{ y_gpu[0], y_gpu[1], y_gpu[2], y_gpu[3] });
}
