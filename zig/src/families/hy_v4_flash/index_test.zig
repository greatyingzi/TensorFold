//! The released checkpoint's whole tensor index must map one-to-one without the weights themselves.
const std = @import("std");
const names = @import("tensor_names.zig");

const fixture = @embedFile("index_fixture.json");

fn countBytes(c: *std.json.Value, prefix: []const u8, needle: []const u8) usize {
    const map = c.object.get("weight_map") orelse return 0;
    if (map != .object) return 0;
    var n: usize = 0;
    var it = map.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, key, prefix)) continue;
        if (std.mem.indexOf(u8, key, needle) != null) n += 1;
    }
    return n;
}

test "hy4 host: all released tensors map one-to-one without weights" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var doc = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer doc.deinit();
    const map = doc.value.object.get("weight_map") orelse return error.MissingHy4Index;
    if (map != .object) return error.MissingHy4Index;
    var seen = std.StringHashMap(void).init(a);
    var total: usize = 0;
    var it = map.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const engine = try names.rename(a, key);
        try std.testing.expect(engine.len > 0);
        const got = try seen.getOrPut(engine);
        try std.testing.expect(!got.found_existing);
        total += 1;
    }
    try std.testing.expectEqual(@as(usize, 1964), total);
    try std.testing.expectEqual(total, seen.count());
}

test "hy4 host: the index groups survive with their expected counts" {
    var doc = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture, .{});
    defer doc.deinit();
    try std.testing.expectEqual(@as(usize, 87), countBytes(&doc.value, "model.dfly_layers.", ""));
    try std.testing.expectEqual(@as(usize, 164), countBytes(&doc.value, "", "engram_layer."));
    try std.testing.expectEqual(@as(usize, 96), countBytes(&doc.value, "", "compressed_indexer."));
    try std.testing.expectEqual(@as(usize, 16), countBytes(&doc.value, "", "unified_per_hc_scales"));
    try std.testing.expectEqual(@as(usize, 441), countBytes(&doc.value, "vit.", ""));
    try std.testing.expectEqual(@as(usize, 42), countBytes(&doc.value, "model.layers.", "input_layernorm.weight"));
    try std.testing.expectEqual(@as(usize, 5), countBytes(&doc.value, "model.dfly_layers.", "input_layernorm.weight"));
    try std.testing.expectEqual(@as(usize, 27), countBytes(&doc.value, "vit.", "input_layernorm.weight"));
}

test "hy4 host: no renamed tensor keeps a checkpoint-only spelling" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var doc = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer doc.deinit();
    const map = doc.value.object.get("weight_map").?.object;
    const forbidden = [_][]const u8{ "hc_pre", "learnable_sink_param", "linear_gate", "hc_head_fn", "hc_head_base", "hc_head_scale" };
    var it = map.iterator();
    while (it.next()) |entry| {
        const engine = try names.rename(a, entry.key_ptr.*);
        for (forbidden) |bad| try std.testing.expect(std.mem.indexOf(u8, engine, bad) == null);
        if (std.mem.startsWith(u8, engine, "vision_tower."))
            try std.testing.expect(std.mem.indexOf(u8, engine, "input_layernorm") == null);
    }
}
