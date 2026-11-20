//! Hy4 Flash in the native CUDA registry: 3bpw device decode with a host fallback.
//!
//! CUDA builds with embedded kernels load the EXL3 pack and run the GPU step;
//! builds without kernels keep the original host Session registration.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const capability = @import("capability.zig");
const config = @import("config.zig");
const loader = @import("loader.zig");
const weights = @import("weights.zig");
const backend = @import("backend.zig");
const gpu_pack = @import("hy4pack");
const gpu_runner = @import("hy4runner");

/// The registry entry (native/cuda_families.zig) names this module as native.
pub const native = @This();

pub const model_type = capability.model_type;
pub const formats: []const []const u8 = &capability.formats;
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 256; // startup text only: prefill_step is not offered
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: hy_v4_flash has no target-side draft head on CUDA; serve without --drafter";
pub const explains_open = true;
/// No prompt snapshots until the GPU backend owns the KV caches.
pub const keeps_no_prompts = true;

pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    lanes: u32 = 8,
    lanes_fixed: bool = false,
};

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
};

const Owned = struct {
    gpa: std.mem.Allocator,
    cfg: config.Config,
    bag: ?weights.Bag = null,
    pack: ?gpu_pack.Pack = null,
    runner: ?gpu_runner.Runner = null,
    impl: backend.Backend,
};

/// Rows a lane may consume: large enough for a smoke reply, bounded so --parallel
/// does not multiply a full model's host KV. The GPU step has no cache yet, so this
/// is only a row counter there.
fn cacheCap(context: usize) usize {
    return @min(context, 64);
}

/// The compressed EXL3 device budget: about 63 GiB of uploaded 3bpw weights plus
/// headroom for uploader staging and scratch, not the pack's file bytes (which
/// include unquantized shards the runner never loads).
pub fn textWeightBytes(_: std.mem.Allocator, _: std.Io, _: []const u8) !?u64 {
    return @as(u64, 66) * (1 << 30); // actual GPU-resident pack ~65.3 GiB
}

fn gpuStep(ptr: *anyopaque, token: usize) anyerror![]const f32 {
    const r: *gpu_runner.Runner = @ptrCast(@alignCast(ptr));
    return r.step(token);
}

fn readConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !config.Config {
    const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(cfg_path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(16 << 20));
    defer gpa.free(bytes);
    return config.parse(gpa, bytes);
}

fn loaded(own: *Owned, stream_bytes: usize) Loaded {
    const slots: usize = @max(1, own.impl.slots);
    return .{
        .backend = own.impl.backend(),
        .facts = own.impl.facts(),
        .rows = prompt_rows,
        .stream_bytes = stream_bytes,
        .free_streams = @intCast(slots),
        .ctx = own,
        .deinit = release,
    };
}

/// The real path: one device pack, one Runner, one opaque forward hook.
fn openGpu(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .cfg = try readConfig(gpa, io, dir), .impl = undefined };
    own.pack = try gpu_pack.load(gpa, io, ctx.d, dir);
    errdefer own.pack.?.deinit();
    own.runner = try gpu_runner.Runner.init(gpa, ctx.d, ctx, &own.pack.?);
    errdefer own.runner.?.deinit();
    const slots: usize = @max(1, o.lanes);
    own.impl = try backend.Backend.init(gpa, .{
        .config = &own.cfg,
        .cap = cacheCap(o.context),
        .slots = slots,
        .gpu = .{ .ptr = &own.runner.?, .step = gpuStep },
    });
    errdefer own.impl.deinit();
    std.log.info("hy_v4_flash: 3bpw GPU decode active, {d} bytes of scratch and {d} stream{s}", .{ own.runner.?.deviceBytes(), slots, if (slots == 1) "" else "s" });
    return loaded(own, own.runner.?.deviceBytes());
}

/// Builds without embedded kernels keep the original host-session registration.
fn openHost(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .cfg = try readConfig(gpa, io, dir), .impl = undefined };
    const meta = (try loader.inspect(gpa, io, dir)).meta;
    own.bag = try weights.load(gpa, io, dir, &meta);
    errdefer own.bag.?.deinit();
    const slots: usize = @max(1, o.lanes);
    own.impl = try backend.Backend.init(gpa, .{ .config = &own.cfg, .bag = &own.bag.?, .cap = cacheCap(o.context), .slots = slots });
    errdefer own.impl.deinit();
    std.log.info("hy_v4_flash: no embedded kernels, forward falling back to the host session ({d} KV rows, {d} stream{s})", .{ cacheCap(o.context), slots, if (slots == 1) "" else "s" });
    return loaded(own, 0);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (!cuda.kernels.available) return openHost(gpa, io, dir, o);
    if (kernels != null) return error.CapturedKernels;
    return openGpu(gpa, io, ctx, dir, o);
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.impl.deinit();
    if (own.runner) |*r| r.deinit();
    if (own.pack) |*pk| pk.deinit();
    if (own.bag) |*b| b.deinit();
    own.* = undefined;
    own.gpa.destroy(own);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.CapturedKernels => "hy_v4_flash runs its own fused GPU kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.UnsupportedHy4Model => "hy_v4_flash: this checkpoint's geometry is not the released text model",
        error.AttentionCacheFull => "hy_v4_flash: the lane's row budget is full; shorten the prompt or reply",
        error.EmptyPrompt => "hy_v4_flash: the lane needs at least one prompt token",
        else => null,
    };
}

test "hy4 native: the CUDA declarations name the admitted checkpoint" {
    try std.testing.expectEqualStrings("hy_v4_flash", model_type);
    try std.testing.expectEqual(@as(usize, 1), formats.len);
    try std.testing.expectEqualStrings("bf16", formats[0]);
    try std.testing.expectEqual(@as(usize, 64), cacheCap(32768));
    try std.testing.expect(@as(?api.Family, null) == null);
    _ = cuda;
    _ = lanes;
}

test "hy4 native: admission counts the compressed device budget, not the pack's files" {
    const bytes = (try textWeightBytes(std.testing.allocator, std.testing.io, "unused")).?;
    try std.testing.expect(bytes >= 60 * (1 << 30));
    try std.testing.expect(bytes <= 101 * (1 << 30));
}
