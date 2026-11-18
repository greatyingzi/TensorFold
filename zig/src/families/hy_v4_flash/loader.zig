//! Open a Hy4 Flash checkpoint's index and shards before any weight is consumed by a forward.
const std = @import("std");
const safetensors = @import("../../core/safetensors.zig");
const config = @import("config.zig");
const graph = @import("graph.zig");
const names = @import("tensor_names.zig");
const shapes = @import("shapes.zig");

pub const EngramMeta = struct {
    source_tokenizer_len: u32 = 0,
    compressed_vocab_size: u32 = 0,
    engram_seed: u64 = 0,
    lookup_sha256: []const u8 = "",
    normalizer_signature: []const u8 = "",
};

pub const Admission = struct {
    tensors: usize = 0,
    shards: usize = 0,
    total_bytes: u64 = 0,
    meta: EngramMeta = .{},
    shaped: usize = 0,
};

fn join(gpa: std.mem.Allocator, dir: []const u8, file: []const u8) ![]u8 {
    return std.fs.path.join(gpa, &.{ dir, file });
}

fn metadata(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !EngramMeta {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var prefix: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &prefix, 0) != prefix.len) return error.BadSafetensors;
    const len = try safetensors.headerSize(std.mem.readInt(u64, &prefix, .little), std.math.cast(usize, try file.length(io)) orelse return error.BadSafetensors);
    const text = try gpa.alloc(u8, len);
    defer gpa.free(text);
    if (try file.readPositionalAll(io, text, 8) != len) return error.BadSafetensors;
    var doc = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return error.NoEngramMetadata;
    defer doc.deinit();
    const root = doc.value.object.get("__metadata__") orelse return error.NoEngramMetadata;
    if (root != .object) return error.NoEngramMetadata;
    const raw = root.object.get("engram_vocab_map_metadata") orelse return error.NoEngramMetadata;
    if (raw != .string) return error.NoEngramMetadata;
    var inner = try std.json.parseFromSlice(std.json.Value, gpa, raw.string, .{});
    defer inner.deinit();
    if (inner.value != .object) return error.NoEngramMetadata;
    const o = inner.value.object;
    var out = EngramMeta{};
    if (o.get("source_tokenizer_len")) |v| {
        if (v == .integer) out.source_tokenizer_len = @intCast(v.integer);
    }
    if (o.get("compressed_vocab_size")) |v| {
        if (v == .integer) out.compressed_vocab_size = @intCast(v.integer);
    }
    if (o.get("engram_seed")) |v| {
        if (v == .integer) out.engram_seed = @intCast(v.integer);
    }
    if (o.get("lookup_sha256")) |v| {
        if (v == .string) out.lookup_sha256 = try gpa.dupe(u8, v.string);
    }
    if (o.get("normalizer_signature")) |v| {
        if (v == .string) out.normalizer_signature = try gpa.dupe(u8, v.string);
    }
    return out;
}

/// Verify every indexed tensor exists in its shard with the declared dtype and byte range.
pub fn inspect(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Admission {
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
    var idx = try std.json.parseFromSlice(std.json.Value, gpa, idx_text, .{});
    defer idx.deinit();
    if (idx.value != .object) return error.BadHy4Index;
    const wmv = idx.value.object.get("weight_map") orelse return error.BadHy4Index;
    if (wmv != .object) return error.BadHy4Index;
    const wm = wmv.object;

    var shards = std.StringHashMap(void).init(gpa);
    defer shards.deinit();
    var mi = wm.iterator();
    while (mi.next()) |entry| {
        if (entry.value_ptr.* != .string) return error.BadHy4Index;
        const file = entry.value_ptr.string;
        if (std.mem.indexOf(u8, file, "/") != null or std.mem.indexOf(u8, file, "..") != null) return error.BadHy4Index;
        const path = try join(gpa, dir, file);
        defer gpa.free(path);
        std.Io.Dir.cwd().access(io, path, .{}) catch return error.MissingHy4Shard;
        try shards.put(file, {});
    }

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var seen = std.StringHashMap(void).init(gpa);
    defer seen.deinit();
    var out = Admission{};
    var shard_it = shards.iterator();
    while (shard_it.next()) |entry| {
        const path = try join(gpa, dir, entry.key_ptr.*);
        defer gpa.free(path);
        var sf = try safetensors.File.open(gpa, io, path);
        defer sf.close(io);
        out.total_bytes += sf.map.memory.len;
        var tit = sf.names.iterator();
        while (tit.next()) |_| out.tensors += 1;
        if (out.meta.lookup_sha256.len == 0) {
            out.meta = metadata(a, io, path) catch EngramMeta{};
        }
        var it = wm.iterator();
        while (it.next()) |item| {
            if (!std.mem.eql(u8, item.value_ptr.string, entry.key_ptr.*)) continue;
            const tensor = sf.get(item.key_ptr.*) orelse return error.IndexedTensorMissing;
            if (tensor.numel() * @as(usize, tensor.dtype.size()) != tensor.bytes.len) return error.BadSafetensors;
            const lookup_len: usize = if (out.meta.source_tokenizer_len > 0) out.meta.source_tokenizer_len else c.vocab;
            if (shapes.expected(&c, item.key_ptr.*, lookup_len)) |want_shape| {
                if (tensor.dtype != want_shape.dtype) return error.UnsupportedHy4TensorDtype;
                if (!want_shape.is(tensor.rank, tensor.shape[0..tensor.rank])) return error.Hy4TensorShapeMismatch;
                out.shaped += 1;
            } else if (tensor.dtype != .bf16) return error.UnsupportedHy4TensorDtype;
            const engine = try names.rename(a, item.key_ptr.*);
            const got = try seen.getOrPut(engine);
            if (got.found_existing) return error.DuplicateHy4Tensor;
        }
    }
    out.shards = shards.count();
    if (out.tensors != seen.count()) return error.UnaccountedHy4Tensor;
    return out;
}

test "hy4 host: the synthetic checkpoint is admitted with all tensors and metadata" {
    const a = try inspect(std.testing.allocator, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic");
    try std.testing.expectEqual(@as(usize, 245), a.tensors);
    try std.testing.expectEqual(@as(usize, 4), a.shards);
    try std.testing.expect(a.total_bytes > 1_000_000);
    try std.testing.expect(a.meta.source_tokenizer_len > 0);
    try std.testing.expect(a.meta.compressed_vocab_size > 0);
    try std.testing.expect(a.shaped >= 169);
}

test "hy4 host: a missing checkpoint directory is refused" {
    try std.testing.expectError(error.FileNotFound, inspect(std.testing.allocator, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/no_such_dir"));
}
