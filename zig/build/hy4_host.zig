//! Hy4 Flash host-only contracts: config shape and tensor names, no device work.
const std = @import("std");

pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const tests = b.addTest(.{
        .filters = &.{"hy4 host:"},
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/hy4_family_tests.zig"),
            .target = b.graph.host,
            .optimize = .debug,
            .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } },
        }),
    });
    const run = b.addRunArtifact(tests);
    const named = b.step("test-hy4-host", "Hy4 Flash config and tensor naming contracts on the host");
    named.dependOn(&run.step);
    test_step.dependOn(&run.step);
}


pub fn runner(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const sys = b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys.zig"), .target = target, .optimize = .debug });
    const cc = b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_config.zig"), .target = target, .optimize = .debug });
    const sg = b.createModule(.{ .root_source_file = b.path("zig/src/core/save_guard.zig"), .target = target, .optimize = .debug });
    const br = b.createModule(.{ .root_source_file = b.path("zig/src/core/bf16_round.zig"), .target = target, .optimize = .debug });
    const q3l = b.createModule(.{ .root_source_file = b.path("zig/src/core/qwen35_layers.zig"), .target = target, .optimize = .debug });
    const q3r = b.createModule(.{ .root_source_file = b.path("zig/src/core/qwen35_rules.zig"), .target = target, .optimize = .debug });
    const rc = b.createModule(.{ .root_source_file = b.path("zig/src/core/rope_config.zig"), .target = target, .optimize = .debug });
    const pf = b.createModule(.{ .root_source_file = b.path("zig/src/core/profile.zig"), .target = target, .optimize = .debug });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "profile", .module = pf }, .{ .name = "sys", .module = sys }, .{ .name = "checkpoint_config", .module = cc }, .{ .name = "save_guard", .module = sg }, .{ .name = "bf16_round", .module = br }, .{ .name = "qwen35_layers", .module = q3l }, .{ .name = "qwen35_rules", .module = q3r }, .{ .name = "rope_config", .module = rc } } });
    const root = b.createModule(.{
        .root_source_file = b.path("zig/hy4_runner.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .imports = &.{
            .{ .name = "sys", .module = sys },
            .{ .name = "checkpoint_config", .module = cc },
            .{ .name = "save_guard", .module = sg },
            .{ .name = "bf16_round", .module = br },
            .{ .name = "qwen35_layers", .module = q3l },
            .{ .name = "qwen35_rules", .module = q3r },
            .{ .name = "rope_config", .module = rc },
            .{ .name = "profile", .module = pf },
            .{ .name = "core", .module = core },
        },
    });
    const exe = b.addExecutable(.{ .name = "hy4-runner", .root_module = root });
    const step = b.step("hy4-runner", "Build the bounded Hy4 checkpoint smoke runner");
    step.dependOn(&b.addInstallArtifact(exe, .{}).step);
}
