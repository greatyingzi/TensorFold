//! Draft proposal and verification over the target session.

// The DFly2 weight path is the next seam.
const std = @import("std");
const spec = @import("spec.zig");
const session_mod = @import("session.zig");
const dfly = @import("dfly.zig");

/// Propose `count` tokens from the target logits, best first.
pub fn propose(gpa: std.mem.Allocator, logits: []const f32, count: usize) ![]u32 {
    if (count == 0 or count > logits.len) return error.BadDraftCount;
    const idx = try gpa.alloc(usize, logits.len);
    defer gpa.free(idx);
    for (idx, 0..) |*v, i| v.* = i;
    std.mem.sort(usize, idx, logits, struct {
        fn lessThan(values: []const f32, a: usize, b: usize) bool {
            if (values[a] != values[b]) return values[a] > values[b];
            return a < b;
        }
    }.lessThan);
    const out = try gpa.alloc(u32, count);
    for (0..count) |i| out[i] = @intCast(idx[i]);
    return out;
}

/// Propose via the DFly2 drafter: one step over the session dump, lm_head picks the best.
pub fn proposeDfly(gpa: std.mem.Allocator, c: *const @import("config.zig").Config, bag: *const @import("weights.zig").Bag, s: *session_mod.Session, dr: *dfly.Drafter, token: usize, out: []u32) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const h = c.hidden;
    var targets: []const []const f32 = &.{};
    if (s.dump_n > 0) targets = s.dump_buf[0..s.dump_n];
    const embed = try embedRow(a, bag, token, h);
    const hidden = try a.alloc(f32, h);
    try dr.step(targets, embed, s.pos, .sum_chain, hidden);
    const logits = try a.alloc(f32, c.vocab);
    try headMatvec(logits, bag, hidden, c.vocab, h);
    const best = try propose(gpa, logits, out.len);
    defer gpa.free(best);
    @memcpy(out, best);
}

fn embedRow(a: std.mem.Allocator, bag: *const @import("weights.zig").Bag, token: usize, h: usize) ![]const f32 {
    const t = bag.get("embed_tokens.weight") orelse return error.MissingHy4Tensor;
    const out = try a.alloc(f32, h);
    switch (t.value) {
        .f32s => |v| @memcpy(out, v[token * h ..][0..h]),
        .bf16raw => |v| {
            const base = token * h * 2;
            for (0..h) |i| {
                const bits = std.mem.readInt(u16, v[base + i * 2 ..][0..2], .little);
                out[i] = @bitCast(@as(u32, bits) << 16);
            }
        },
        .i32s, .affine4 => return error.ExpectedF32Tensor,
    }
    return out;
}

fn headMatvec(y: []f32, bag: *const @import("weights.zig").Bag, x: []const f32, rows: usize, cols: usize) !void {
    const ops = @import("ops.zig");
    const t = bag.get("lm_head.weight") orelse return error.MissingHy4Tensor;
    switch (t.value) {
        .f32s => |v| try ops.matvec(y, v, x, rows, cols),
        .bf16raw => |v| try ops.matvecBf16(y, v, x, rows, cols),
        .i32s, .affine4 => return error.ExpectedF32Tensor,
    }
}

/// Verify drafts against the target: accept while each draft equals the target's greedy pick.

// `logits` is the target output at the current position; the session advances by accepted tokens.
pub fn verify(session: *session_mod.Session, logits: []const f32, drafts: []const u32) !usize {
    if (drafts.len == 0) return 0;
    var accepted: usize = 0;
    var current = logits;
    while (accepted < drafts.len) {
        const greedy = try spec.pick(current);
        if (greedy != drafts[accepted]) break;
        const next = try session.step(@intCast(greedy));
        accepted += 1;
        current = next;
    }
    return accepted;
}

test "hy4 host: drafted continuation equals the plain greedy continuation" {
    const gpa = std.testing.allocator;
    var bag = try weights_load(gpa);
    defer bag.deinit();
    const c = try config_parse(gpa);

    var plain = try session_mod.Session.init(gpa, &c, &bag, 4);
    defer plain.deinit();
    _ = try plain.step(1);
    var draft_session = try session_mod.Session.init(gpa, &c, &bag, 4);
    defer draft_session.deinit();
    const logits = try draft_session.step(1);
    const drafts = try propose(gpa, logits, 4);
    defer gpa.free(drafts);
    const n = try verify(&draft_session, logits, drafts);
    try std.testing.expect(n >= 1);
    for (drafts[0..n]) |tok| _ = try plain.step(@intCast(tok));
    try std.testing.expectEqual(try spec.pick(plain.logits), try spec.pick(draft_session.logits));
}

fn weights_load(gpa: std.mem.Allocator) !@import("weights.zig").Bag {
    const w = @import("weights.zig");
    return w.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &.{});
}

fn config_parse(gpa: std.mem.Allocator) !@import("config.zig").Config {
    return @import("config.zig").parse(gpa, @embedFile("config_synthetic_fixture.json"));
}


test "hy4 host: DFly2 drafts verify as the plain continuation" {
    const gpa = std.testing.allocator;
    var bag = try weights_load(gpa);
    defer bag.deinit();
    const c = try config_parse(gpa);
    var s = try session_mod.Session.init(gpa, &c, &bag, 4);
    defer s.deinit();
    var dr = try dfly.Drafter.init(gpa, &c, &bag, 4);
    defer dr.deinit();
    const logits = try s.step(1);
    const drafts = try gpa.alloc(u32, c.dfly.drafts);
    defer gpa.free(drafts);
    try proposeDfly(gpa, &c, &bag, &s, &dr, 1, drafts);
    const n = try verify(&s, logits, drafts);
    try std.testing.expect(n <= drafts.len);
    var plain = try session_mod.Session.init(gpa, &c, &bag, 4);
    defer plain.deinit();
    _ = try plain.step(1);
    for (drafts[0..n]) |tok| _ = try plain.step(@intCast(tok));
    try std.testing.expectEqual(try spec.pick(plain.logits), try spec.pick(s.logits));
}
