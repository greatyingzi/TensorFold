//! One real-weight GPU decode step via the serve's Runner: the fast A/B testbed.
const std = @import("std");
const cuda = @import("cuda");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.BadArgs;
    const dir = args[1];
    const token: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 120000;
    const steps: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 1;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const a = init.arena.allocator();
    const io = init.io;
    const packmod = @import("hy4pack");
    const runner_mod = @import("hy4runner");
    var p = try packmod.load(a, io, &driver, dir);
    defer p.deinit();
    std.debug.print("pack loaded\n", .{});
    var r = try runner_mod.Runner.init(a, &driver, &ctx, &p);
    defer r.deinit();
    std.debug.print("runner ready\n", .{});
    var tok = token;
    for (0..steps) |i| {
        const logits = try r.step(tok);
        var best: usize = 0;
        for (logits, 0..) |v, j| {
            if (v > logits[best]) best = j;
        }
        // top-5 diagnostic
        const tmp = try a.alloc(f32, logits.len);
        defer a.free(tmp);
        @memcpy(tmp, logits);
        std.debug.print("STEP {d}: token {d} -> {d} logit {d:.4} top5:", .{ i, tok, best, logits[best] });
        for (0..5) |_| {
            var bi: usize = 0;
            for (tmp, 0..) |v, j| {
                if (v > tmp[bi]) bi = j;
            }
            std.debug.print(" {d}:{d:.2}", .{ bi, tmp[bi] });
            tmp[bi] = -1e30;
        }
        std.debug.print("\n", .{});
        tok = best;
    }
}
