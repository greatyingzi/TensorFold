//! Hy4 Flash admission: what the engine can serve today, and how it refuses everything else.
const std = @import("std");
const cfg = @import("config.zig");
const graph = @import("graph.zig");

pub const model_type = "hy_v4_flash";
/// Only original-precision weights are admitted until the 4-bit reader lands.
pub const formats = [_][]const u8{"bf16"};

pub const Admission = enum {
    admitted,
    vision_not_ready,
    format_refused,
    config_refused,
};

pub fn problem(a: Admission) []const u8 {
    return switch (a) {
        .admitted => "",
        .vision_not_ready => "--vision: the Hy4 Flash image tower is not read yet; serve text only",
        .format_refused => "hy_v4_flash: unsupported checkpoint format (bf16 only until the 4-bit reader lands); " ++
            "W8A8 INT8 activations are not admitted",
        .config_refused => "hy_v4_flash: config.json is malformed or names an unsupported geometry",
    };
}

fn isFormat(want: []const u8) bool {
    for (formats) |f| if (std.mem.eql(u8, f, want)) return true;
    return false;
}

/// Decide before opening any checkpoint tensor. No model is read here.
pub fn admit(gpa: std.mem.Allocator, bytes: []const u8, format: []const u8, vision: bool) !Admission {
    if (!isFormat(format)) return .format_refused;
    const c = cfg.parse(gpa, bytes) catch return .config_refused;
    graph.validate(&c) catch return .config_refused;
    if (vision and c.vision.present) return .vision_not_ready;
    return .admitted;
}

test "hy4 host: the released text config is admitted" {
    const a = try admit(std.testing.allocator, @embedFile("config_fixture.json"), "bf16", false);
    try std.testing.expectEqual(Admission.admitted, a);
    try std.testing.expectEqualStrings("", problem(a));
}

test "hy4 host: vision and W8A8 are refused with actionable messages" {
    const bytes = @embedFile("config_fixture.json");
    const vision = try admit(std.testing.allocator, bytes, "bf16", true);
    try std.testing.expectEqual(Admission.vision_not_ready, vision);
    try std.testing.expect(std.mem.indexOf(u8, problem(vision), "--vision") != null);
    const w8a8 = try admit(std.testing.allocator, bytes, "w8a8-int8", false);
    try std.testing.expectEqual(Admission.format_refused, w8a8);
    try std.testing.expect(std.mem.indexOf(u8, problem(w8a8), "W8A8") != null);
    const fp8 = try admit(std.testing.allocator, bytes, "fp8", false);
    try std.testing.expectEqual(Admission.format_refused, fp8);
}

test "hy4 host: malformed geometry is refused before any weight is read" {
    const bytes = @embedFile("config_fixture.json");
    const bad = try std.mem.replaceOwned(u8, std.testing.allocator, bytes, "\"hc_mult\": 4", "\"hc_mult\": 8");
    defer std.testing.allocator.free(bad);
    try std.testing.expectEqual(Admission.config_refused, try admit(std.testing.allocator, bad, "bf16", false));
    try std.testing.expectEqual(Admission.config_refused, try admit(std.testing.allocator, "{}", "bf16", false));
}
