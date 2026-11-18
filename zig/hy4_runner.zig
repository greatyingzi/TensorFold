//! Smoke runner: load a bounded slice of an Hy4 checkpoint and print the next greedy token.
const std = @import("std");
const _sys = @import("sys");
const _checkpoint_config = @import("checkpoint_config");
const _save_guard = @import("save_guard");
const _bf16_round = @import("bf16_round");
const _qwen35_layers = @import("qwen35_layers");
const _qwen35_rules = @import("qwen35_rules");
const _rope_config = @import("rope_config");
const _profile = @import("profile");

const config = @import("src/families/hy_v4_flash/config.zig");
const weights = @import("src/families/hy_v4_flash/weights.zig");
const session = @import("src/families/hy_v4_flash/session.zig");
const loader = @import("src/families/hy_v4_flash/loader.zig");
const spec = @import("src/families/hy_v4_flash/spec.zig");

fn nowNs() i64 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, ts.sec) * std.time.ns_per_s + ts.nsec;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.next() orelse return error.BadArgs;
    _ = _sys.windows;
    _ = &.{ _checkpoint_config, _save_guard, _bf16_round, _qwen35_layers, _qwen35_rules, _rope_config, _profile };
    const dir = it.next() orelse return error.BadArgs;
    const token_s = it.next() orelse return error.BadArgs;
    const layers_s = it.next() orelse return error.BadArgs;
    const token = try std.fmt.parseInt(usize, token_s, 10);
    const layers = try std.fmt.parseInt(usize, layers_s, 10);
    const io = init.io;
    std.debug.print("PHASE inspect\n", .{});
    const adm = try loader.inspect(a, io, dir);
    const meta = adm.meta;
    var c = try config.read(a, io, dir);
    if (layers > 0) c.layers = @intCast(@min(layers, c.layers));
    c.engram.count = 0;
    c.allow_no_engram = true;
    c.dfly.layers = 1;
    c.dfly.targets[0] = c.layers;
    c.dfly.target_count = 1;
    std.debug.print("PHASE load\n", .{});
    var bag = try weights.loadPrefix(a, io, dir, &meta, c.layers);
    std.debug.print("PHASE load_done\n", .{});
    const small_vocab: usize = @min(c.vocab, 256);
    c.vocab = @intCast(small_vocab);
    const emb = try a.alloc(f32, small_vocab * c.hidden);
    for (emb, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) - 8.0;
    try bag.map.put("embed_tokens.weight", .{ .name = "embed_tokens.weight", .rank = 2, .shape = .{ small_vocab, c.hidden, 0 }, .value = .{ .f32s = emb } });
    const head = try a.alloc(f32, small_vocab * c.hidden);
    for (head, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) - 6.0;
    try bag.map.put("lm_head.weight", .{ .name = "lm_head.weight", .rank = 2, .shape = .{ small_vocab, c.hidden, 0 }, .value = .{ .f32s = head } });
    defer bag.deinit();
    var count: usize = 1;
    if (it.next()) |extra| count = try std.fmt.parseInt(usize, extra, 10);
    std.debug.print("PHASE session\n", .{});
    var s = try session.Session.init(a, &c, &bag, count + 2);
    defer s.deinit();
    std.debug.print("PHASE step\n", .{});
    const t0 = nowNs();
    var last: u32 = 0;
    for (0..count) |i| last = try spec.pick(try s.step(token + i % c.vocab));
    const ns: i64 = nowNs() - t0;
    std.debug.print("LOADED {d} tensors, LAYERS {d}, TOKENS {d}, FORWARD_NS {d}, TOK_PER_S {d}, NEXT {d}\n", .{ bag.count(), c.layers, count, ns, if (ns > 0) @divTrunc(@as(i64, @intCast(count)) * @as(i64, std.time.ns_per_s), ns) else 0, last });
}
