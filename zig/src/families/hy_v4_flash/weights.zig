//! Host-owned tensor bag: load a Hy4 checkpoint's BF16 and I32 tensors into flat f32/i32 buffers.
const std = @import("std");
const safetensors = @import("../../core/safetensors.zig");
const config = @import("config.zig");
const graph = @import("graph.zig");
const names = @import("tensor_names.zig");
const shapes = @import("shapes.zig");
const loader = @import("loader.zig");
const affine4 = @import("affine4.zig");
const exl3 = @import("exl3.zig");

pub const Affine4 = struct {
    codes: []const u8,
    scales: []const u8,
    biases: []const u8,
    values: usize,
    experts: usize,
    rows: usize,
};

pub const Value = union(enum) {
    f32s: []f32,
    i32s: []i32,
    affine4: Affine4,
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
    io: std.Io,
    map: std.StringHashMap(Tensor),
    files: std.ArrayList(safetensors.File),

    pub fn deinit(self: *Bag) void {
        // maps left open until process exit
        self.files.deinit(self.arena.allocator());
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
    return loadInner(gpa, io, dir, meta, 0);
}

/// Load only the first `limit` backbone layers plus global tensors, for a bounded real-checkpoint smoke.
pub fn loadPrefix(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, meta: *const loader.EngramMeta, limit: usize) !Bag {
    return loadInner(gpa, io, dir, meta, limit);
}

fn loadInner(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, meta: *const loader.EngramMeta, prefix_limit: usize) !Bag {
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
        .io = io,
        .map = std.StringHashMap(Tensor).init(gpa),
        .files = .empty,
    };
    errdefer out.deinit();
    const a = out.arena.allocator();
    const lookup_len: usize = if (meta.source_tokenizer_len > 0) meta.source_tokenizer_len else c.vocab;
    var opened = std.StringHashMap(usize).init(a);
    var shard_it = wm.iterator();
    while (shard_it.next()) |entry| {
        var sf: *safetensors.File = undefined;
        if (opened.get(entry.value_ptr.string)) |idx| {
            sf = &out.files.items[idx];
        } else {
            const path = try join(a, dir, entry.value_ptr.string);
            const file = safetensors.File.open(a, io, path) catch |e| {
                if (e == error.FileNotFound) continue;
                return e;
            };
            const idx = out.files.items.len;
            try out.files.append(a, file);
            try opened.put(entry.value_ptr.string, idx);
            sf = &out.files.items[idx];
        }
        var it = wm.iterator();
        while (it.next()) |item| {
            if (!std.mem.eql(u8, item.value_ptr.string, entry.value_ptr.string)) continue;
            const ckpt = item.key_ptr.*;
            if (prefix_limit > 0) {
                const lm = "model.layers.";
                if (std.mem.startsWith(u8, ckpt, lm)) {
                    const rest = ckpt[lm.len..];
                    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse 0;
                    const li = std.fmt.parseInt(usize, rest[0..dot], 10) catch 0;
                    if (li >= prefix_limit) continue;
                }
                if (std.mem.startsWith(u8, ckpt, "model.dfly_layers.")) continue;
                if (std.mem.indexOf(u8, ckpt, "engram_layer.") != null) continue;
                if (std.mem.startsWith(u8, ckpt, "vit.")) continue;
                if (std.mem.eql(u8, ckpt, "lm_head.weight") or std.mem.eql(u8, ckpt, "model.embed_tokens.weight")) continue;
            }
            const t = sf.get(ckpt) orelse return error.IndexedTensorMissing;
            var base = ckpt;
            if (std.mem.endsWith(u8, ckpt, ".weight") and
                std.mem.indexOf(u8, ckpt, "mlp.experts.") != null) {
                base = ckpt[0 .. ckpt.len - ".weight".len];
            }
            const want = shapes.expected(&c, base, lookup_len) orelse continue;
            if (c.quant_method.len > 0 and std.mem.eql(u8, c.quant_method, "exl3")) {
                try exl3.admit(&c);
                const tr = try std.mem.concat(a, u8, &.{ base, ".trellis" });
                if (sf.get(tr)) |tt| {
                    const sun = try std.mem.concat(a, u8, &.{ base, ".suh" });
                    const svn = try std.mem.concat(a, u8, &.{ base, ".svh" });
                    const su = sf.get(sun) orelse return error.MissingExl3Tensor;
                    const sv = sf.get(svn) orelse return error.MissingExl3Tensor;
                    if (want.rank != 2) return error.Exl3PackedRankUnsupported;
                    const buf = try exl3.decodeRaw(a, tt.bytes, su.bytes, sv.bytes, want.a, want.b, c.quant_bits, try exl3.codebook(c.quant_codebook));
                    const ekey = try names.rename(a, base);
                    try out.map.put(ekey, .{ .name = ekey, .rank = want.rank, .shape = .{ want.a, want.b, 0 }, .value = .{ .f32s = buf } });
                    continue;
                }
            }
            const qn = try std.mem.concat(a, u8, &.{ base, ".weight" });
            if (sf.get(qn)) |qt| {
                if (qt.dtype == .u32) {
                    const sn = try std.mem.concat(a, u8, &.{ base, ".scales" });
                    const bn = try std.mem.concat(a, u8, &.{ base, ".biases" });
                    const st = sf.get(sn) orelse return error.MissingAffineTensor;
                    const bt = sf.get(bn) orelse return error.MissingAffineTensor;
                    if (st.dtype != .bf16 or bt.dtype != .bf16) return error.UnsupportedHy4TensorDtype;
                    var vals: usize = want.a;
                    if (want.rank > 1) vals *= want.b;
                    if (want.rank > 2) vals *= want.c;
                    const experts: usize = if (want.rank == 3) want.a else 1;
                    const rows: usize = if (want.rank == 3) want.b else 1;
                    const qkey = try names.rename(a, base);
                    try out.map.put(qkey, .{ .name = qkey, .rank = want.rank, .shape = .{ want.a, want.b, want.c }, .value = .{ .affine4 = .{ .codes = qt.bytes, .scales = st.bytes, .biases = bt.bytes, .values = vals, .experts = experts, .rows = rows } } });
                    continue;
                }
            }
            if (t.dtype != want.dtype or !want.is(t.rank, t.shape[0..t.rank])) {
                std.debug.print("SHAPE_MISMATCH {s} dtype={s}/{s} rank={d}/{d} got={any} want={any}\n", .{ ckpt, @tagName(t.dtype), @tagName(want.dtype), t.rank, want.rank, t.shape[0..t.rank], want });
                return error.Hy4TensorShapeMismatch;
            }
            const key = try names.rename(a, base);
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
