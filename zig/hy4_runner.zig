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
const ops = @import("src/families/hy_v4_flash/ops.zig");
const dfly_mod = @import("src/families/hy_v4_flash/dfly.zig");
const rotary = @import("src/families/hy_v4_flash/rotary.zig");

fn nowNs() i64 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, ts.sec) * std.time.ns_per_s + ts.nsec;
}

fn embedRow(a: std.mem.Allocator, bag: *const @import("src/families/hy_v4_flash/weights.zig").Bag, token: usize, h: usize) ![]const f32 {
    const t = bag.get("embed_tokens.weight") orelse return error.MissingHy4Tensor;
    const out = try a.alloc(f32, h);
    switch (t.value) {
        .f32s => |v| @memcpy(out, v[token * h ..][0..h]),
        .bf16raw => |v| {
            const base = token * h * 2;
            for (0..h) |i| out[i] = @bitCast(@as(u32, std.mem.readInt(u16, v[base + i * 2 ..][0..2], .little)) << 16);
        },
        else => return error.ExpectedF32Tensor,
    }
    return out;
}

fn probe(a: std.mem.Allocator, io: std.Io, dir: []const u8, seed: usize, steps: usize) !void {
    const adm = try loader.inspect(a, io, dir);
    const meta = adm.meta;
    var c = try config.read(a, io, dir);
    c.engram.count = 0;
    c.allow_no_engram = true;
    std.debug.print("PROBE load layers {d} dfly {d}\n", .{ c.layers, c.dfly.layers });
    var bag = try weights.loadStreaming(a, io, dir, &meta, 0);
    defer bag.deinit();
    const h: usize = c.hidden;
    const ntarget = @min(c.dfly.target_count, 8);
    var s = try session.Session.init(a, &c, &bag, steps + 2);
    defer s.deinit();
    for (0..ntarget) |i| {
        s.dump_buf[i] = try a.alloc(f32, h);
        s.dump_layers[i] = @intCast(c.dfly.targets[i]);
    }
    s.dump_n = ntarget;
    var dr_sum = try dfly_mod.Drafter.init(a, &c, &bag, steps + 2);
    defer dr_sum.deinit();
    var dr_tgt = try dfly_mod.Drafter.init(a, &c, &bag, steps + 2);
    defer dr_tgt.deinit();
    const logits_buf = try a.alloc(f32, c.vocab * 3);
    var agree_sum: usize = 0;
    var agree_tgt: usize = 0;
    var tok: usize = seed % c.vocab;
    const t0 = nowNs();
    for (0..steps) |step_i| {
        const row = embedRow(a, &bag, tok, h) catch break;
        const logits = try s.step(tok);
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            if (v > logits[best]) best = i;
        }
        const dh = try a.alloc(f32, h);
        try dr_sum.step(s.dump_buf[0..ntarget], row, step_i, .sum_chain, dh);
        const lsum: []f32 = logits_buf[0..c.vocab];
        try headVec(lsum, &bag, dh, c.vocab, h);
        var asum: usize = 0;
        for (lsum, 0..) |v, i| {
            if (v > lsum[asum]) asum = i;
        }
        try dr_tgt.step(s.dump_buf[0..ntarget], row, step_i, .target_only, dh);
        const ltgt: []f32 = logits_buf[c.vocab ..][0..c.vocab];
        try headVec(ltgt, &bag, dh, c.vocab, h);
        var atgt: usize = 0;
        for (ltgt, 0..) |v, i| {
            if (v > ltgt[atgt]) atgt = i;
        }
        if (asum == best) agree_sum += 1;
        if (atgt == best) agree_tgt += 1;
        if (step_i % 5 == 0 or step_i == steps - 1) std.debug.print("PROBE step {d}: main {d} sum {d} tgt {d}\n", .{ step_i, best, asum, atgt });
        tok = best;
    }
    const ns = nowNs() - t0;
    std.debug.print("PROBE RESULT steps {d} agree_sum {d} ({d}%) agree_target {d} ({d}%) ns {d}\n", .{ steps, agree_sum, agree_sum * 100 / steps, agree_tgt, agree_tgt * 100 / steps, ns });
}

fn headVec(y: []f32, bag: *const @import("src/families/hy_v4_flash/weights.zig").Bag, x: []const f32, rows: usize, cols: usize) !void {
    const t = bag.get("lm_head.weight") orelse return error.MissingHy4Tensor;
    switch (t.value) {
        .f32s => |v| try ops.matvec(y, v, x, rows, cols),
        .bf16raw => |v| try ops.matvecBf16(y, v, x, rows, cols),
        else => return error.ExpectedF32Tensor,
    }
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
    const io = init.io;
    if (std.mem.eql(u8, token_s, "probe")) {
        const seed = try std.fmt.parseInt(usize, layers_s, 10);
        const steps = try std.fmt.parseInt(usize, it.next() orelse return error.BadArgs, 10);
        probe(a, io, dir, seed, steps) catch |e| {
            std.debug.print("PROBE_ERR {t}\n", .{e});
            if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
            return e;
        };
        return;
    }
    const token = try std.fmt.parseInt(usize, token_s, 10);
    const layers = try std.fmt.parseInt(usize, layers_s, 10);
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
    var bag = try weights.loadStreaming(a, io, dir, &meta, c.layers);
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
