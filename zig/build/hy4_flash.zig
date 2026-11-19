//! The Hy4 Flash native CUDA family module: registration over the 3bpw GPU decode step.
const std = @import("std");

/// The device-resident 3bpw pack, shared by the runner and the family's open.
pub fn packModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/hy_v4_flash/gpu_pack.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// The reusable decode step, over the same pack module the family opens.
pub fn runnerModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, pack: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/hy_v4_flash/gpu_runner.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "hy4pack", .module = pack } } });
}

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy_v4_flash/native.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("engine_api", api);
    const pack = packModule(b, target, optimize, cuda, core);
    m.addImport("hy4pack", pack);
    m.addImport("hy4runner", runnerModule(b, target, optimize, cuda, pack));
    return m;
}
