//! EXL3 (exllamav3) expert tensor gate and scalar reader for Hy4 Flash.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const fmt = @import("../../core/exl3_format.zig");

pub const bits_min: f32 = 1;
pub const bits_max: f32 = 8;

/// Refuse a checkpoint whose declared EXL3 variant the scalar reader cannot decode.
pub fn admit(c: *const config.Config) !void {
    if (c.quant_method.len == 0) return error.NoQuantizationDeclared;
    if (!std.mem.eql(u8, c.quant_method, "exl3")) return error.UnsupportedQuantMethod;
    if (c.quant_bits < bits_min or c.quant_bits > bits_max) return error.UnsupportedExl3Bits;
    if (c.quant_codebook.len != 0 and
        !std.mem.eql(u8, c.quant_codebook, "inst3") and
        !std.mem.eql(u8, c.quant_codebook, "mcg") and
        !std.mem.eql(u8, c.quant_codebook, "mul1")) return error.UnsupportedExl3Codebook;
}

pub fn codebook(name: []const u8) !fmt.Codebook {
    if (std.mem.eql(u8, name, "inst3") or name.len == 0) return .inst3;
    if (std.mem.eql(u8, name, "mcg")) return .mcg;
    if (std.mem.eql(u8, name, "mul1")) return .mul1;
    return error.UnsupportedExl3Codebook;
}

fn vecI16(a: std.mem.Allocator, bytes: []const u8, count: usize) ![]i16 {
    if (bytes.len < count * 2) return error.BadExl3Tensor;
    const out = try a.alloc(i16, count);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
    return out;
}

fn vecF16(a: std.mem.Allocator, bytes: []const u8, count: usize) ![]f16 {
    if (bytes.len < count * 2) return error.BadExl3Tensor;
    const out = try a.alloc(f16, count);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
    return out;
}

fn get(bag: *const weights.Bag, name: []const u8) ![]const u8 {
    const t = bag.get(name) orelse return error.MissingExl3Tensor;
    return switch (t.value) {
        .f32s => |v| std.mem.sliceAsBytes(v),
        .i32s => |v| std.mem.sliceAsBytes(v),
    };
}

/// Decode directly from safetensors byte slices (used by the checkpoint loader).
pub fn decodeRaw(a: std.mem.Allocator, trellis_bytes: []const u8, su_bytes: []const u8, sv_bytes: []const u8, k: usize, n: usize, bits: f32, cb: fmt.Codebook) ![]f32 {
    if (k == 0 or n == 0 or k % 16 != 0 or n % 16 != 0) return error.BadExl3Tensor;
    const words_per_tile: usize = @intFromFloat(bits * 16);
    if (words_per_tile < 16 or words_per_tile % 8 != 0) return error.UnsupportedExl3Bits;
    const tiles_k = k / 16;
    const tiles_n = n / 16;
    const trellis = try vecI16(a, trellis_bytes, tiles_k * tiles_n * words_per_tile);
    defer a.free(trellis);
    const suh = try vecF16(a, su_bytes, k);
    defer a.free(suh);
    const svh = try vecF16(a, sv_bytes, n);
    defer a.free(svh);
    const out64 = try a.alloc(f64, k * n);
    defer a.free(out64);
    const scratch = try a.alloc(f64, 2 * k * n);
    defer a.free(scratch);
    fmt.dequantize(.{ .halves = @intFromFloat(bits * 2) }, cb, trellis, tiles_k, tiles_n, suh, svh, out64, scratch);
    const out = try a.alloc(f32, k * n);
    for (out, out64) |*v, x| v.* = @floatCast(x);
    return out;
}

/// Decode one expert linear W[K,N] from its trellis/suh/svh triple into f32.
pub fn decode(a: std.mem.Allocator, bag: *const weights.Bag, stem: []const u8, k: usize, n: usize, bits: f32, cb: fmt.Codebook) ![]f32 {
    if (k == 0 or n == 0 or k % 16 != 0 or n % 16 != 0) return error.BadExl3Tensor;
    const tn = try std.mem.concat(a, u8, &.{ stem, ".trellis" });
    defer a.free(tn);
    const sn = try std.mem.concat(a, u8, &.{ stem, ".suh" });
    defer a.free(sn);
    const vn = try std.mem.concat(a, u8, &.{ stem, ".svh" });
    defer a.free(vn);
    const words_per_tile: usize = @intFromFloat(bits * 16);
    if (words_per_tile < 16 or words_per_tile % 8 != 0) return error.UnsupportedExl3Bits;
    const tiles_k = k / 16;
    const tiles_n = n / 16;
    const trellis = try vecI16(a, try get(bag, tn), tiles_k * tiles_n * words_per_tile);
    defer a.free(trellis);
    const suh = try vecF16(a, try get(bag, sn), k);
    defer a.free(suh);
    const svh = try vecF16(a, try get(bag, vn), n);
    defer a.free(svh);
    const out64 = try a.alloc(f64, k * n);
    defer a.free(out64);
    const scratch = try a.alloc(f64, 2 * k * n);
    defer a.free(scratch);
    const width = fmt.Width{ .halves = @intFromFloat(bits * 2) };
    fmt.dequantize(width, cb, trellis, tiles_k, tiles_n, suh, svh, out64, scratch);
    const out = try a.alloc(f32, k * n);
    for (out, out64) |*v, x| v.* = @floatCast(x);
    return out;
}

test "hy4 host: EXL3 gate accepts declared variants and refuses others" {
    var c = config.Config{};
    c.quant_method = "exl3";
    c.quant_bits = 4;
    c.quant_codebook = "mcg";
    try admit(&c);
    c.quant_method = "mlx";
    try std.testing.expectError(error.UnsupportedQuantMethod, admit(&c));
    c.quant_method = "exl3";
    c.quant_bits = 12;
    try std.testing.expectError(error.UnsupportedExl3Bits, admit(&c));
    try std.testing.expectEqual(fmt.Codebook.mcg, try codebook("mcg"));
}


test "hy4 host: raw EXL3 decode reads trellis and scale triples" {
    const gpa = std.testing.allocator;
    const k: usize = 16;
    const n: usize = 16;
    const trellis = try gpa.alloc(u8, (k / 16) * (n / 16) * 64 * 2);
    defer gpa.free(trellis);
    @memset(trellis, 0);
    const scales = try gpa.alloc(u8, (k + n) * 2);
    defer gpa.free(scales);
    for (scales, 0..) |*v, i| v.* = @as(u8, if (i % 2 == 0) 0 else 0x3c);
    const out = try decodeRaw(gpa, trellis, scales[0 .. k * 2], scales[k * 2 ..], k, n, 4, .mcg);
    defer gpa.free(out);
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}
