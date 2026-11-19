//! Hy4 Flash in the native CUDA registry: minimal wiring over the host session.
//!
//! This is a registration skeleton, not a GPU engine: the checkpoint is read on
//! the host and every forward goes through families/hy_v4_flash/session.zig.
//! TODO(GPU decode): open the CUDA pack and swap backend.advance for its step.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const capability = @import("capability.zig");
const config = @import("config.zig");
const loader = @import("loader.zig");
const weights = @import("weights.zig");
const backend = @import("backend.zig");

/// The registry entry (native/cuda_families.zig) names this module as `native`.
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
    bag: weights.Bag,
    impl: backend.Backend,
};

/// Rows a temporary host session keeps: large enough for a smoke reply, small
/// enough that --parallel does not multiply a full model's KV on the host.
fn cacheCap(context: usize) usize {
    return @min(context, 64);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = ctx; // the fallback reads weights on the host; no device work yet
    if (kernels != null) return error.CapturedKernels;
    const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(cfg_path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(16 << 20));
    defer gpa.free(bytes);
    const cfg = try config.parse(gpa, bytes);

    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .cfg = cfg, .bag = undefined, .impl = undefined };
    const meta = (try loader.inspect(gpa, io, dir)).meta;
    own.bag = try weights.load(gpa, io, dir, &meta);
    errdefer own.bag.deinit();
    const slots: usize = @max(1, o.lanes);
    own.impl = try backend.Backend.init(gpa, .{ .config = &own.cfg, .bag = &own.bag, .cap = cacheCap(o.context), .slots = slots });
    errdefer own.impl.deinit();

    std.log.info("hy_v4_flash: CUDA registration active, forward falling back to the host session ({d} KV rows, {d} stream{s})", .{ cacheCap(o.context), slots, if (slots == 1) "" else "s" });
    return .{
        .backend = own.impl.backend(),
        .facts = own.impl.facts(),
        .rows = prompt_rows,
        .stream_bytes = 0,
        .free_streams = @intCast(slots),
        .ctx = own,
        .deinit = release,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.impl.deinit();
    own.bag.deinit();
    own.* = undefined;
    own.gpa.destroy(own);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.CapturedKernels => "hy_v4_flash runs its own host session: unset TENSORFOLD_CUDA_KERNELS",
        error.UnsupportedHy4Model => "hy_v4_flash: this checkpoint's geometry is not the released text model",
        error.AttentionCacheFull => "hy_v4_flash: the temporary host session's KV rows are full; shorten the prompt or reply",
        error.EmptyPrompt => "hy_v4_flash: the temporary host session needs at least one prompt token",
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
