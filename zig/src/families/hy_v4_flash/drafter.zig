//! DFly2 draft geometry from config: target layers, depth and reused KV/index sources.
const std = @import("std");
const cfg = @import("config.zig");

pub const Plan = struct {
    layers: u32,
    drafts: u32,
    index_topk: u32,
    targets: [cfg.max_draft_targets]u32,
    count: usize,

    pub fn isTarget(p: *const Plan, layer: usize) bool {
        for (p.targets[0..p.count]) |t| if (t - 1 == layer) return true;
        return false;
    }
};

pub fn plan(c: *const cfg.Config) Plan {
    return .{
        .layers = c.dfly.layers,
        .drafts = c.dfly.drafts,
        .index_topk = c.dfly.index_topk,
        .targets = c.dfly.targets,
        .count = c.dfly.target_count,
    };
}

/// DFly2 reads the target layers' KV and sparse index instead of keeping a separate full KV pool.
pub fn validate(c: *const cfg.Config) error{UnsupportedHy4Drafter}!void {
    const p = plan(c);
    if (p.layers == 0 or p.drafts == 0 or p.count == 0 or p.count > p.layers) return error.UnsupportedHy4Drafter;
    if (p.drafts > 15 or p.index_topk == 0) return error.UnsupportedHy4Drafter;
    for (p.targets[0..p.count]) |t| {
        if (t == 0 or t > c.layers) return error.UnsupportedHy4Drafter;
    }
}

/// Draft rows reuse the target layer's KV and sparse index, so only target layers may be tapped.
pub fn tapLayer(c: *const cfg.Config, draft: usize) !usize {
    const p = plan(c);
    if (draft >= p.count) return error.NoSuchDraftTap;
    return @as(usize, p.targets[draft]) - 1;
}

test "hy4 host: the released DFly2 plan taps five backbone layers" {
    const c = try @import("config.zig").parse(std.testing.allocator, @embedFile("config_fixture.json"));
    try validate(&c);
    const p = plan(&c);
    try std.testing.expectEqual(@as(u32, 5), p.layers);
    try std.testing.expectEqual(@as(u32, 5), p.drafts);
    try std.testing.expectEqual(@as(usize, 5), p.count);
    try std.testing.expectEqual(@as(u32, 2048), p.index_topk);
    try std.testing.expect(p.isTarget(24) and p.isTarget(40));
    try std.testing.expect(!p.isTarget(23));
    try std.testing.expectEqual(@as(usize, 24), try tapLayer(&c, 0));
    try std.testing.expectEqual(@as(usize, 40), try tapLayer(&c, 4));
    try std.testing.expectError(error.NoSuchDraftTap, tapLayer(&c, 5));
}

test "hy4 host: the synthetic drafter plan validates too" {
    const c = try @import("config.zig").parse(std.testing.allocator, @embedFile("config_synthetic_fixture.json"));
    try validate(&c);
    const p = plan(&c);
    try std.testing.expectEqual(@as(u32, 1), p.layers);
    try std.testing.expectEqual(@as(u32, 2), p.drafts);
    try std.testing.expect(p.isTarget(3));
}
