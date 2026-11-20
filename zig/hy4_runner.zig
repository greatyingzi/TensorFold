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
const full_forward = @import("src/families/hy_v4_flash/full_forward.zig");

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


fn topIndices(a: std.mem.Allocator, v: []const f32, k: usize) ![]usize {
    const idx = try a.alloc(usize, v.len);
    for (idx, 0..) |*p, i| p.* = i;
    std.mem.sort(usize, idx, v, struct {
        fn lessThan(xs: []const f32, p: usize, q: usize) bool {
            if (xs[p] != xs[q]) return xs[p] > xs[q];
            return p < q;
        }
    }.lessThan);
    return idx[0..k];
}

fn topOverlap(xs: []const usize, ys: []const usize) usize {
    var n: usize = 0;
    for (xs) |x| {
        for (ys) |y| {
            if (x == y) n += 1;
        }
    }
    return n;
}

fn cosine(x: []const f32, y: []const f32) f32 {
    var dot: f64 = 0;
    var nx: f64 = 0;
    var ny: f64 = 0;
    for (x, y) |a, b| {
        dot += @as(f64, a) * @as(f64, b);
        nx += @as(f64, a) * @as(f64, a);
        ny += @as(f64, b) * @as(f64, b);
    }
    if (nx == 0 or ny == 0) return 0;
    return @floatCast(dot / @sqrt(nx * ny));
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
        const hcos = cosine(dh, s.last_hidden);
        const eco = cosine(row, s.last_hidden);
        const t41 = if (ntarget > 0) cosine(s.dump_buf[ntarget - 1], s.last_hidden) else 0;
        const lsum: []f32 = logits_buf[0..c.vocab];
        try headVec(lsum, &bag, dh, c.vocab, h, c.eps);
        var asum: usize = 0;
        for (lsum, 0..) |v, i| {
            if (v > lsum[asum]) asum = i;
        }
        try dr_tgt.step(s.dump_buf[0..ntarget], row, step_i, .target_only, dh);
        const ltgt: []f32 = logits_buf[c.vocab ..][0..c.vocab];
        try headVec(ltgt, &bag, dh, c.vocab, h, c.eps);
        var atgt: usize = 0;
        for (ltgt, 0..) |v, i| {
            if (v > ltgt[atgt]) atgt = i;
        }
        if (asum == best) agree_sum += 1;
        if (atgt == best) agree_tgt += 1;
        const top_main = try topIndices(a, logits, 10);
        const top_sum = try topIndices(a, lsum, 10);
        const top_tgt = try topIndices(a, ltgt, 10);
        if (step_i % 5 == 0 or step_i == steps - 1) std.debug.print("PROBE step {d}: main {d} sum {d} tgt {d} top10 sum {d}/10 tgt {d}/10 hcos {d:.4} emb {d:.4} t41 {d:.4}\n", .{ step_i, best, asum, atgt, topOverlap(top_main, top_sum), topOverlap(top_main, top_tgt), hcos, eco, t41 });
        tok = best;
    }
    const ns = nowNs() - t0;
    std.debug.print("PROBE RESULT steps {d} agree_sum {d} ({d}%) agree_target {d} ({d}%) ns {d}\n", .{ steps, agree_sum, agree_sum * 100 / steps, agree_tgt, agree_tgt * 100 / steps, ns });
}


fn dumpRef(a: std.mem.Allocator, io: std.Io, dir: []const u8, token: usize) !void {
    const adm = try loader.inspect(a, io, dir);
    const meta = adm.meta;
    var c = try config.read(a, io, dir);
    c.engram.count = 0;
    c.allow_no_engram = true;
    var bag = try weights.loadStreaming(a, io, dir, &meta, 0);
    defer bag.deinit();
    const h = c.hidden;
    var s = try session.Session.init(a, &c, &bag, 4);
    defer s.deinit();
    const ndump = @min(c.layers, 42);
    for (0..ndump) |i| {
        s.dump_buf[i] = try a.alloc(f32, h);
        s.dump_layers[i] = @intCast(i);
    }
    s.dump_n = ndump;
    const logits = try s.step(token);
    var best: usize = 0;
    for (logits, 0..) |v, i| {
        if (v > logits[best]) best = i;
    }
    for (0..ndump) |i| {
        std.debug.print("CPUDUMP L{d} mixed", .{i});
        for (s.dump_buf[i][0..8]) |v| std.debug.print(" {d}", .{v});
        std.debug.print("\n", .{});
    }
    std.debug.print("CPUDUMP NEXT {d}\n", .{best});
}


fn exportLogits(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const meta = @import("src/families/hy_v4_flash/loader.zig").EngramMeta{ .source_tokenizer_len = 128 };
    var bag = try weights.load(a, io, dir, &meta);
    defer bag.deinit();
    var c = try config.parse(a, @embedFile("src/families/hy_v4_flash/config_synthetic_fixture.json"));
    c.engram.count = 0;
    c.allow_no_engram = true;
    var s = try session.Session.init(a, &c, &bag, 8);
    defer s.deinit();
    const seq = [_]usize{ 1, 5, 9, 13, 20 };
    for (seq, 0..) |tok, step| {
        const logits = try s.step(tok);
        std.debug.print("ZIG_SEQ {d} tok={d} logits8", .{ step, tok });
        for (logits[0..8]) |v| std.debug.print(" {d:.4}", .{v});
        std.debug.print("\n", .{});
    }
}

fn genRef(a: std.mem.Allocator, io: std.Io, dir: []const u8, prompt_csv: []const u8, steps: usize) !void {
    const adm = try loader.inspect(a, io, dir);
    var c = try config.read(a, io, dir);
    // Engram enabled: lookup_table is stored in the checkpoint weights; loader.meta has a c.vocab fallback.
    std.debug.print("GENREF engram.count={d} allow_no={}\n", .{ c.engram.count, c.allow_no_engram });
    var bag = try weights.loadStreaming(a, io, dir, &adm.meta, 0);
    defer bag.deinit();
    var s = try session.Session.init(a, &c, &bag, 8 + steps);
    defer s.deinit();
    var it = std.mem.splitScalar(u8, prompt_csv, ',');
    var toks = std.ArrayList(usize).empty;
    while (it.next()) |t| {
        const v = std.fmt.parseInt(usize, std.mem.trim(u8, t, " "), 10) catch continue;
        try toks.append(a, v);
    }
    if (toks.items.len == 0) try toks.append(a, 0);
    var logits: []const f32 = &.{};
    for (toks.items) |t| logits = try s.step(t);
    std.debug.print("GENREF prompt done ({d} tokens)\n", .{toks.items.len});
    for (0..steps) |i| {
        var best: usize = 0;
        for (logits, 0..) |v, j| {
            if (v > logits[best]) best = j;
        }
        std.debug.print("GENREF step {d}: -> {d}\n", .{ i, best });
        logits = try s.step(best);
    }
}

fn headVec(y: []f32, bag: *const @import("src/families/hy_v4_flash/weights.zig").Bag, x: []f32, rows: usize, cols: usize, eps: f32) !void {
    _ = eps;
    const t = bag.get("lm_head.weight") orelse return error.MissingHy4Tensor;
    switch (t.value) {
        .f32s => |v| try ops.matvec(y, v, x, rows, cols),
        .bf16raw => |v| try ops.matvecBf16(y, v, x, rows, cols),
        else => return error.ExpectedF32Tensor,
    }
}


fn specDecode(a: std.mem.Allocator, io: std.Io, dir: []const u8, token: usize, count: usize) !void {
    const adm = try loader.inspect(a, io, dir);
    const meta = adm.meta;
    var c = try config.read(a, io, dir);
    c.engram.count = 0;
    c.allow_no_engram = true;
    var bag = try weights.loadStreaming(a, io, dir, &meta, 0);
    defer bag.deinit();
    const ntarget = @min(c.dfly.target_count, 8);
    var s = try session.Session.init(a, &c, &bag, count + 2);
    defer s.deinit();
    for (0..ntarget) |i| {
        s.dump_buf[i] = try a.alloc(f32, c.hidden);
        s.dump_layers[i] = @intCast(c.dfly.targets[i]);
    }
    s.dump_n = ntarget;
    var dr = try dfly_mod.Drafter.init(a, &c, &bag, count + 2);
    defer dr.deinit();
    const drafts = try a.alloc(u32, 1);
    var tok: usize = token % c.vocab;
    var logits: []const f32 = try s.step(tok);
    var accepted: usize = 0;
    const t0 = nowNs();
    for (0..count) |step_i| {
        const next = try spec.pick(logits);
        try @import("src/families/hy_v4_flash/draft.zig").proposeDfly(a, &c, &bag, &s, &dr, tok, drafts);
        const hit: usize = if (drafts[0] == next) 1 else 0;
        accepted += hit;
        tok = next;
        logits = try s.step(tok);
        if (step_i % 5 == 0 or step_i == count - 1) std.debug.print("SPEC step {d}: draft {d} target {d} accepted {d}\n", .{ step_i, drafts[0], next, hit });
    }
    const ns = nowNs() - t0;
    std.debug.print("SPEC RESULT tokens {d} accepted {d} ({d}%) ns {d} tok_per_s {d}\n", .{ count, accepted, if (count > 0) accepted * 100 / count else 0, ns, if (ns > 0) @divTrunc(@as(i64, @intCast(count)) * @as(i64, std.time.ns_per_s), ns) else 0 });
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
    if (std.mem.eql(u8, token_s, "spec")) {
        const seed = try std.fmt.parseInt(usize, layers_s, 10);
        const steps = try std.fmt.parseInt(usize, it.next() orelse return error.BadArgs, 10);
        try specDecode(a, io, dir, seed, steps);
        return;
    }
    if (std.mem.eql(u8, token_s, "exportlogits")) {
        try exportLogits(a, io, dir);
        return;
    }
    if (std.mem.eql(u8, token_s, "genref")) {
        const steps_s = it.next() orelse return error.BadArgs;
        try genRef(a, io, dir, layers_s, try std.fmt.parseInt(usize, steps_s, 10));
        return;
    }
    if (std.mem.eql(u8, token_s, "dumpref")) {
        const tok = try std.fmt.parseInt(usize, layers_s, 10);
        try dumpRef(a, io, dir, tok);
        return;
    }
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
    {
        const nd = @min(c.layers, s.dump_n);
        for (0..nd) |i| {
            std.debug.print("CPUDUMP L{d} mixed", .{i});
            for (s.dump_buf[i][0..8]) |v| std.debug.print(" {d}", .{v});
            std.debug.print("\n", .{});
        }
    }
    std.debug.print("LOADED {d} tensors, LAYERS {d}, TOKENS {d}, FORWARD_NS {d}, TOK_PER_S {d}, NEXT {d}\n", .{ bag.count(), c.layers, count, ns, if (ns > 0) @divTrunc(@as(i64, @intCast(count)) * @as(i64, std.time.ns_per_s), ns) else 0, last });
}
