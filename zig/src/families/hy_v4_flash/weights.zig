//! Host-owned tensor bag: load a Hy4 checkpoint's BF16 and I32 tensors into flat f32/i32 buffers.
const std = @import("std");
const safetensors = @import("../../core/safetensors.zig");
const config = @import("config.zig");
const graph = @import("graph.zig");
const names = @import("tensor_names.zig");
const shapes = @import("shapes.zig");
const loader = @import("loader.zig");

pub const Value = union(enum) {
    f32s: []f32,
    i32s: []i32,
};

pub const Tensor = struct {
    name: []const u8,
    rank: u8,
    shape: [3]usize,
    value: Value,

    pub fn numel(t: *const Tensor) usize {
        var n: usize = 1;
        for (t.shape[0..t.rank]) |d| n *= d;
        return n;
    }
};

pub const Bag = struct {
    arena: std.heap.ArenaAllocator,
    map: std.StringHashMap(Tensor),

    pub fn deinit(self: *Bag) void {
        self.map.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn get(self: *const Bag, name: []const u8) ?*const Tensor {
        return self.map.getPtr(name);
    }

    pub fn count(self: *const Bag) usize {
        return self.map.count();
    }
};

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn join(a: std.mem.Allocator, dir: []const u8, file: []const u8) ![]u8 {
    return std.fs.path.join(a, &.{ dir, file });
}

/// Load every modeled tensor into host memory under its engine name.
pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, meta: *const loader.EngramMeta) !Bag {
    const cfg_path = try join(gpa, dir, "config.json");
    defer gpa.free(cfg_path);
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 22));
    defer gpa.free(cfg_text);
    const c = try config.parse(gpa, cfg_text);
    try graph.validate(&c);

    const idx_path = try join(gpa, dir, "model.safetensors.index.json");
    defer gpa.free(idx_path);
    const idx_text = try std.Io.Dir.cwd().readFileAlloc(io, idx_path, gpa, .limited(64 << 20));
    defer gpa.free(idx_text);
    var doc = try std.json.parseFromSlice(std.json.Value, gpa, idx_text, .{});
    defer doc.deinit();
    const wmv = doc.value.object.get("weight_map") orelse return error.BadHy4Index;
    if (wmv != .object) return error.BadHy4Index;
    const wm = wmv.object;

    var out = Bag{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .map = std.StringHashMap(Tensor).init(gpa),
    };
    errdefer out.deinit();
    const a = out.arena.allocator();
    const lookup_len: usize = if (meta.source_tokenizer_len > 0) meta.source_tokenizer_len else c.vocab;
    var shard_it = wm.iterator();
    while (shard_it.next()) |entry| {
        const path = try join(a, dir, entry.value_ptr.string);
        var sf = safetensors.File.open(a, io, path) catch |e| {
            if (e == error.FileNotFound) continue;
            return e;
        };
        defer sf.close(io);
        var it = wm.iterator();
        while (it.next()) |item| {
            if (!std.mem.eql(u8, item.value_ptr.string, entry.value_ptr.string)) continue;
            const ckpt = item.key_ptr.*;
            const t = sf.get(ckpt) orelse return error.IndexedTensorMissing;
            const want = shapes.expected(&c, ckpt, lookup_len) orelse continue;
            if (t.dtype != want.dtype or !want.is(t.rank, t.shape[0..t.rank])) return error.Hy4TensorShapeMismatch;
            const key = try names.rename(a, ckpt);
            const count = t.numel();
            var shape: [3]usize = .{ 0, 0, 0 };
            for (0..t.rank) |i| shape[i] = t.shape[i];
            switch (want.dtype) {
                .bf16 => {
                    const buf = try a.alloc(f32, count);
                    const bytes = std.mem.sliceAsBytes(t.bytes);
                    for (0..count) |i| {
                        const bits = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
                        buf[i] = bf16ToF32(bits);
                    }
                    try out.map.put(key, .{ .name = key, .rank = t.rank, .shape = shape, .value = .{ .f32s = buf } });
                },
                .i32 => {
                    const buf = try a.alloc(i32, count);
                    for (0..count) |i| buf[i] = @bitCast(std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little));
                    try out.map.put(key, .{ .name = key, .rank = t.rank, .shape = shape, .value = .{ .i32s = buf } });
                },
                .f32 => {
                    const buf = try a.alloc(f32, count);
                    for (0..count) |i| buf[i] = @bitCast(std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little));
                    try out.map.put(key, .{ .name = key, .rank = t.rank, .shape = shape, .value = .{ .f32s = buf } });
                },
                else => return error.UnsupportedHy4TensorDtype,
            }
        }
    }
    return out;
}

test "hy4 host: the synthetic checkpoint loads into a host tensor bag" {
    const gpa = std.testing.allocator;
    const meta = loader.EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    try std.testing.expect(bag.count() >= 204);
    const q = bag.get("layers.0.self_attn.q_a_proj.weight") orelse return error.MissingTensor;
    try std.testing.expectEqual(@as(usize, 2), q.rank);
    try std.testing.expectEqual(@as(usize, 16), q.shape[0]);
    try std.testing.expectEqual(@as(usize, 64), q.shape[1]);
    const norm = bag.get("layers.0.input_layernorm.weight") orelse return error.MissingTensor;
    try std.testing.expectEqual(@as(f32, 1), norm.value.f32s[0]);
    const map = bag.get("layers.1.engram_layer.lookup_table") orelse return error.MissingTensor;
    try std.testing.expectEqual(@as(usize, 128), map.numel());
    try std.testing.expectEqual(@as(i32, 7), map.value.i32s[7]);
}
