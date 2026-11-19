//! `hy4-gpu-bench`: the Hy4 Flash timings, and with a pack path the real-weight check.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const hy4_bench = @import("hy4_bench.zig");
const hy4_weights_check = @import("hy4_weights_check.zig");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    if (args.len > 1) {
        if (std.mem.eql(u8, args[1], "forward")) {
            try hy4_bench.forwardBench(gpu);
            return 0;
        }
        try hy4_weights_check.run(gpu, args[1]);
        return 0;
    }
    try hy4_bench.run(gpu);
    return 0;
}
