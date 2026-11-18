//! Engram n-gram memory injection into the four hyper-connection streams.
const std = @import("std");
const config = @import("config.zig");
const weights = @import("weights.zig");
const ops = @import("ops.zig");
const rng = @import("engram_rng.zig");

fn f32s(bag: *const weights.Bag, name: []const u8) ![]const f32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .f32s => |v| v,
        .i32s, .affine4 => error.ExpectedF32Tensor,
        .bf16raw => return error.ExpectedF32Tensor,
    };
}

fn matvecW(y: []f32, bag: *const weights.Bag, name: []const u8, x: []const f32, rows: usize, cols: usize) !void {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    switch (t.value) {
        .f32s => |v| try ops.matvec(y, v, x, rows, cols),
        .bf16raw => |v| try ops.matvecBf16(y, v, x, rows, cols),
        .i32s, .affine4 => return error.ExpectedF32Tensor,
    }
}

fn i32s(bag: *const weights.Bag, name: []const u8) ![]const i32 {
    const t = bag.get(name) orelse return error.MissingHy4Tensor;
    return switch (t.value) {
        .i32s => |v| v,
        .f32s, .bf16raw, .affine4 => error.ExpectedI32Tensor,
    };
}

fn concat(a: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    return std.mem.concat(a, u8, parts);
}

/// Hash one position's 2- and 3-gram into the 16 head bucket ids, following the official mixed-XOR mod prime scheme.
pub fn hash(ngram: [3]u32, table: []const i32, multipliers: [3]u64, primes: [16]u32, pad: u32, dst: *[16]u64) !void {
    if (table.len == 0 or pad >= table.len) return error.BadEngramTable;
    var mixed: [3]i64 = undefined;
    for (ngram, 0..) |tok, i| {
        const id: usize = if (tok < table.len) tok else pad;
        mixed[i] = @as(i64, table[id]) * @as(i64, @intCast(multipliers[i] & 0x7fffffffffffffff));
    }
    var m2 = mixed[0] ^ mixed[1];
    var m3 = m2 ^ mixed[2];
    if (m2 < 0) m2 += @as(i64, std.math.maxInt(i64));
    if (m3 < 0) m3 += @as(i64, std.math.maxInt(i64));
    for (0..8) |h| {
        dst[h] = @intCast(@as(u64, @intCast(m2)) % primes[h]);
        dst[8 + h] = @intCast(@as(u64, @intCast(m3)) % primes[8 + h]);
    }
}

/// Inject one token's Engram residual into every stream; `streams` are [hc_mult, hidden].
pub fn inject(gpa: std.mem.Allocator, c: *const config.Config, bag: *const weights.Bag, layer: usize, ngram: [3]u32, streams: []const []const f32, out: [][]f32) !void {
    const hc = c.hc_mult;
    const total = (c.engram.ngram - 1) * c.engram.embed;
    const head_dim = total / (2 * c.engram.heads);
    if (streams.len != hc or out.len != hc or head_dim == 0) return error.BadEngramShape;
    for (streams, out) |s, o| if (s.len != c.hidden or o.len != c.hidden) return error.BadEngramShape;

    const prefix = try std.fmt.allocPrint(gpa, "layers.{d}.engram_layer.", .{layer});
    defer gpa.free(prefix);
    const table_name = try concat(gpa, &.{ prefix, "lookup_table" });
    defer gpa.free(table_name);
    const table = try i32s(bag, table_name);
    const embed_name = try concat(gpa, &.{ prefix, "multi_head_embedding.embedding.weight" });
    defer gpa.free(embed_name);
    const embed = try f32s(bag, embed_name);
    var primes: [16]u32 = undefined;
    for (c.engram.layers[0..c.engram.count]) |e| {
        if (e.id - 1 == layer) primes = e.primes;
    }
    const muls = try rng.multipliers(gpa, c.engram.seed, @intCast(layer + 1), c.engram.vocab[0]);
    var ids: [16]u64 = undefined;
    try hash(ngram, table, muls, primes, c.engram.pad, &ids);

    const flat = try gpa.alloc(f32, total);
    defer gpa.free(flat);
    var offset: usize = 0;
    for (ids, 0..) |id, h| {
        const norm_name = try std.fmt.allocPrint(gpa, "{s}lookup_per_head_norms.{d}.weight", .{ prefix, h });
        defer gpa.free(norm_name);
        const vec = flat[h * head_dim ..][0..head_dim];
        const row = embed[(offset + @as(usize, id)) * head_dim ..][0..head_dim];
        try ops.rmsNorm(vec, row, try f32s(bag, norm_name), c.eps);
        offset += primes[h];
    }

    const vp_name = try concat(gpa, &.{ prefix, "value_proj.weight" });
    defer gpa.free(vp_name);
    const gamma_name = try concat(gpa, &.{ prefix, "unified_pre_gamma" });
    defer gpa.free(gamma_name);
    const value = try gpa.alloc(f32, c.hidden);
    defer gpa.free(value);
    try matvecW(value, bag, vp_name, flat, c.hidden, total);
    const gamma = try f32s(bag, gamma_name);
    var sum: f64 = 0;
    for (value) |v| sum += @as(f64, v) * @as(f64, v);
    const inv = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(c.hidden)) + @as(f64, c.eps));
    for (value, gamma) |*v, g| v.* = v.* * @as(f32, @floatCast(inv)) * g;

    for (0..hc) |s| {
        const kn = try std.fmt.allocPrint(gpa, "{s}key_projs.{d}.weight", .{ prefix, s });
        defer gpa.free(kn);
        const kd = try std.fmt.allocPrint(gpa, "{s}norm_key.{d}.weight", .{ prefix, s });
        defer gpa.free(kd);
        const qd = try std.fmt.allocPrint(gpa, "{s}norm_query.{d}.weight", .{ prefix, s });
        defer gpa.free(qd);
        const sn = try std.fmt.allocPrint(gpa, "{s}unified_per_hc_scales.{d}", .{ prefix, s });
        defer gpa.free(sn);
        const key = try gpa.alloc(f32, c.hidden);
        defer gpa.free(key);
        const query = try gpa.alloc(f32, c.hidden);
        defer gpa.free(query);
        try matvecW(key, bag, kn, flat, c.hidden, total);
        try ops.rmsNorm(key, key, try f32s(bag, kd), c.eps);
        try ops.rmsNorm(query, streams[s], try f32s(bag, qd), c.eps);
        var dot: f64 = 0;
        for (key, query) |a, b| dot += @as(f64, a) * @as(f64, b);
        dot *= 1.0 / @sqrt(@as(f64, @floatFromInt(c.hidden)));
        const mag: f64 = if (dot < 0) -@sqrt(@max(1e-6, -dot)) else @sqrt(@max(1e-6, dot));
        const gate = ops.sigmoid(@floatCast(mag));
        const scale = try f32s(bag, sn);
        if (scale.len != c.hidden) return error.BadEngramShape;
        for (out[s], value, scale) |*o, v, sc| o.* = (gate + sc) * v;
    }
}

test "hy4 host: Engram injection stays finite on the synthetic layer" {
    const gpa = std.testing.allocator;
    const meta = @import("loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const c = try @import("config.zig").parse(gpa, @embedFile("config_synthetic_fixture.json"));
    const row = try gpa.alloc(f32, c.hidden);
    defer gpa.free(row);
    @memset(row, 1);
    var streams: [4][]const f32 = undefined;
    var outs: [4][]f32 = undefined;
    for (0..4) |i| {
        streams[i] = row;
        outs[i] = try gpa.alloc(f32, c.hidden);
    }
    defer for (outs) |o| gpa.free(o);
    try inject(gpa, &c, &bag, 1, .{ 1, 2, 3 }, &streams, &outs);
    for (outs) |o| for (o) |v| try std.testing.expect(std.math.isFinite(v));
}
