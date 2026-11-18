//! Hy4 Flash host-only contracts: config shape and tensor names, no device work.
const std = @import("std");

pub fn steps(b: *std.Build, test_step: *std.Build.Step) void {
    const tests = b.addTest(.{
        .filters = &.{"hy4 host:"},
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/hy4_family_tests.zig"),
            .target = b.graph.host,
            .optimize = .debug,
        }),
    });
    const run = b.addRunArtifact(tests);
    const named = b.step("test-hy4-host", "Hy4 Flash config and tensor naming contracts on the host");
    named.dependOn(&run.step);
    test_step.dependOn(&run.step);
}
