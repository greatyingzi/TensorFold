//! Reader for Hy4 PTQ weights stored as MLX affine 4-bit, group 64.
const std = @import("std");
const core = @import("../../core/affine4_host.zig");
const weights = @import("weights.zig");

pub const group = 64;
pub const suffixes = struct {
    pub const codes = ".qcodes";
    pub const scales = ".qscales";
    pub const biases = ".qbiases";
};

fn u16s(bag: *const weights.Bag, name: []const u8, count: usize) ![]const u16 {
    const t = bag.get(name) orelse return error.MissingAffineTensor;
    const raw = switch (t.value) {
        .f32s => |v| std.mem.sliceAsBytes(v),
        .i32s => |v| std.mem.sliceAsBytes(v),
    };
    if (raw.len < count * 2) return error.BadAffineTensor;
    const out = @as([*]const u16, @ptrCast(raw.ptr))[0..count];
    return out;
}

fn u32s(bag: *const weights.Bag, name: []const u8, count: usize) ![]const u32 {
    const t = bag.get(name) orelse return error.MissingAffineTensor;
    const raw = switch (t.value) {
        .f32s => |v| std.mem.sliceAsBytes(v),
        .i32s => |v| std.mem.sliceAsBytes(v),
    };
    if (raw.len < count * 4) return error.BadAffineTensor;
    const out = @as([*]const u32, @ptrCast(raw.ptr))[0..count];
    return out;
}

/// Dequantize one affine4 tensor into f32; `values` is the flattened row-major element count.
pub fn materialize(gpa: std.mem.Allocator, bag: *const weights.Bag, name: []const u8, values: usize) ![]f32 {
    if (values == 0 or values % group != 0) return error.BadAffineTensor;
    const words = values / 8;
    const groups = values / group;
    const cname = try std.mem.concat(gpa, u8, &.{ name, suffixes.codes });
    defer gpa.free(cname);
    const sname = try std.mem.concat(gpa, u8, &.{ name, suffixes.scales });
    defer gpa.free(sname);
    const bname = try std.mem.concat(gpa, u8, &.{ name, suffixes.biases });
    defer gpa.free(bname);
    const w = try u32s(bag, cname, words);
    const s = try u16s(bag, sname, groups);
    const b = try u16s(bag, bname, groups);
    const out = try gpa.alloc(f32, values);
    errdefer gpa.free(out);
    core.dequantize(w, s, b, out);
    return out;
}

test "hy4 host: affine4 reader dequantizes a flat tensor using core layout" {
    const gpa = std.testing.allocator;
    const values: usize = 64;
    var words: [8]u32 = undefined;
    var scales: [1]u16 = undefined;
    var biases: [1]u16 = undefined;
    var raw: [64]f32 = undefined;
    for (&raw, 0..) |*v, i| v.* = @floatFromInt(i);
    core.quantize(&raw, &words, &scales, &biases);
    var bag = weights.Bag{ .arena = std.heap.ArenaAllocator.init(gpa), .map = std.StringHashMap(weights.Tensor).init(gpa) };
    defer bag.deinit();
    const a = bag.arena.allocator();
    const mk = struct {
        fn f(a_: std.mem.Allocator, b: *weights.Bag, k: []const u8, bytes: []const u8, dtype: u8) !void {
            const n = (bytes.len + 3) / 4;
            const buf = try a_.alloc(f32, n);
            @memcpy(std.mem.sliceAsBytes(buf)[0..bytes.len], bytes);
            const t = weights.Tensor{ .name = k, .rank = 1, .shape = .{ n, 0, 0 }, .value = .{ .f32s = buf } };
            try b.map.put(k, t);
            _ = dtype;
        }
    };
    try mk.f(a, &bag, "w.qcodes", std.mem.sliceAsBytes(&words), 4);
    try mk.f(a, &bag, "w.qscales", std.mem.sliceAsBytes(&scales), 2);
    try mk.f(a, &bag, "w.qbiases", std.mem.sliceAsBytes(&biases), 2);
    const back = try materialize(gpa, &bag, "w", values);
    defer gpa.free(back);
    const step: f32 = 63.0 / 15.0;
    for (back, raw) |v, want| try std.testing.expect(@abs(v - want) <= step * 0.5 + 1e-3);
}
