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

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.next() orelse return error.BadArgs;
    _ = _sys.windows;
    _ = &.{ _checkpoint_config, _save_guard, _bf16_round, _qwen35_layers, _qwen35_rules, _rope_config, _profile };
    const dir = it.next() orelse return error.BadArgs;
    const token_s = it.next() orelse return error.BadArgs;
    const layers_s = it.next() orelse return error.BadArgs;
    const token = try std.fmt.parseInt(usize, token_s, 10);
    const layers = try std.fmt.parseInt(usize, layers_s, 10);
    const io = init.io;
    const adm = try loader.inspect(a, io, dir);
    const meta = adm.meta;
    var c = try config.read(a, io, dir);
    if (layers > 0 and layers < c.layers) {
        c.layers = @intCast(layers);
        c.engram.count = 0;
        c.allow_no_engram = true;
        c.dfly.layers = 1;
        c.dfly.targets[0] = c.layers;
        c.dfly.target_count = 1;
    }
    var bag = try weights.loadPrefix(a, io, dir, &meta, layers);
    defer bag.deinit();
    var s = try session.Session.init(a, &c, &bag, 2);
    defer s.deinit();
    const logits = try s.step(token);
    std.debug.print("LOADED {d} tensors, LAYERS {d}, NEXT {d}\n", .{ bag.count(), c.layers, try spec.pick(logits) });
}
