//! Hy4 Flash lane backend: greedy lanes over a host Session or a GPU decode step.
//!
//! The GPU path receives an opaque Runner hook from native.zig, so this module stays
//! device-agnostic and keeps compiling in host-only test builds. The lane callbacks
//! around the forward seam (handles, windows, keep, release) are shared.

const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const ops = @import("ops.zig");
const config = @import("config.zig");
const weights = @import("weights.zig");
const Session = @import("session.zig").Session;

/// Handles name a token in a small ring, as in the Dory CUDA backend.
const ring = 1024;

/// The device forward seam, handed in by native.zig to avoid importing CUDA here.
pub const Gpu = struct {
    ptr: *anyopaque,
    step: *const fn (*anyopaque, usize) anyerror![]const f32,
};

pub const Options = struct {
    config: *const config.Config,
    /// The host bag is not loaded on the GPU path.
    bag: ?*const weights.Bag = null,
    /// Rows a lane may consume; the host session's KV is bounded to a few dozen.
    cap: usize,
    slots: usize = 8,
    /// When set, tokens go through the 3bpw GPU step instead of Session.step.
    gpu: ?Gpu = null,
};

const Lane = struct {
    session: ?Session = null, // only the host fallback owns one
    len: u64 = 0, // rows consumed: prompt length + committed replies
    first: u32 = 0, // greedy pick off the prompt's last row, drawn at prefill
};

pub const Backend = struct {
    gpa: std.mem.Allocator,
    c: *const config.Config,
    bag: ?*const weights.Bag,
    gpu: ?Gpu,
    cap: usize,
    slots: usize,
    lanes: std.AutoHashMapUnmanaged(*const Stream, *Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, o: Options) !Backend {
        if (o.cap == 0 or o.slots == 0) return error.BadBackendOptions;
        if (o.gpu == null and o.bag == null) return error.BadBackendOptions;
        return .{ .gpa = gpa, .c = o.config, .bag = o.bag, .gpu = o.gpu, .cap = o.cap, .slots = o.slots };
    }

    pub fn deinit(x: *Backend) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |slot| {
            if (slot.*.session) |*s| s.deinit();
            x.gpa.destroy(slot.*);
        }
        x.lanes.deinit(x.gpa);
        x.* = undefined;
    }

    pub fn backend(x: *Backend) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    /// One row a step, no draft head and no shared windows: the host session is single-stream state.
    pub fn facts(_: *const Backend) lanes.Model {
        return .{ .exact_width = 1, .hidden_rows = false, .gpu_tokens = false, .mtp = false, .speculate = false, .copy_drafts = false, .lone_copies = false, .plain_streams = 1, .drafts = 1 };
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn get(x: *Backend, s: *const Stream) !*Lane {
        return x.lanes.get(s) orelse error.UnknownStream;
    }

    fn remember(x: *Backend, token: u32) u64 {
        const h = x.next;
        x.next += 1;
        x.drawn[@intCast(h % ring)] = token;
        return h;
    }

    fn value(x: *Backend, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(@ptrCast(x), h),
        };
    }

    /// THE forward seam: the only place a token enters the model.
    fn advance(x: *Backend, l: *Lane, token: u32) !u32 {
        if (token >= x.c.vocab) return error.BadForwardShape;
        if (l.len >= x.cap) return error.AttentionCacheFull;
        const logits = if (x.gpu) |g| try g.step(g.ptr, token) else try l.session.?.step(token);
        l.len += 1;
        return @intCast(try ops.argmax(logits));
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        if (x.lanes.contains(s)) return error.StreamLive;
        if (x.lanes.count() >= x.slots) return error.NoFreeSlot;
        const ids = s.prompt();
        if (ids.len == 0) return error.EmptyPrompt;
        if (ids.len > x.cap) return error.AttentionCacheFull;
        const l = try x.gpa.create(Lane);
        errdefer x.gpa.destroy(l);
        l.* = .{};
        if (x.gpu) |g| {
            // One synchronous Runner serves every lane; no per-lane KV state exists yet.
            var last: []const f32 = &.{};
            for (ids) |tok| {
                if (tok >= x.c.vocab) return error.BadForwardShape;
                last = try g.step(g.ptr, tok);
            }
            l.first = @intCast(try ops.argmax(last));
        } else {
            l.session = try Session.init(x.gpa, x.c, x.bag.?, x.cap);
            errdefer l.session.?.deinit();
            for (ids) |tok| {
                if (tok >= x.c.vocab) return error.BadForwardShape;
                _ = try l.session.?.step(tok);
            }
            l.first = @intCast(try ops.argmax(l.session.?.logits));
        }
        l.len = ids.len;
        x.lanes.put(x.gpa, s, l) catch |e| {
            if (l.session) |*ss| ss.deinit();
            return e;
        };
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = try get(x, s);
        if (position != l.len) return error.PositionMismatch;
        return x.remember(l.first);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = try get(x, s);
        const token = try x.value(feed);
        if (position != l.len + 1) return error.PositionMismatch;
        return x.remember(try advance(x, l, token));
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        return x.drawn[@intCast(h % ring)];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        var rowbuf: [64]u32 = undefined;
        for (windows, out) |w, o| {
            if (w.held != 0) return error.NoHeldDrafts;
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            if (w.rows() > rowbuf.len) return error.WindowTooWide;
            const l = try get(x, w.stream);
            rowbuf[0] = w.pending;
            for (w.tokens, 1..) |tok, i| rowbuf[i] = tok;
            for (0..w.rows()) |r| {
                if (w.positions[r] != l.len + 1) return error.PositionMismatch;
                const picked = try advance(x, l, rowbuf[r]);
                o.sampled[r] = picked;
                if (r + 1 < w.rows()) o.drafts[r] = rowbuf[r + 1];
            }
        }
    }

    fn keep(_: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        for (windows, paths) |w, path| {
            if (path.len == 0 or path.len > 1) return error.ChainsOnly;
            _ = w;
        }
        // The host Session cannot rewind its KV rows. With no draft head every
        // window is one row and keep only confirms it; a rejected row cannot be
        // rolled back until the GPU decode step owns the cache.
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.lanes.fetchRemove(s)) |entry| {
            const l = entry.value;
            if (l.session) |*ss| ss.deinit();
            x.gpa.destroy(l);
        }
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release };
};

test "hy4 host: the lane backend draws greedy tokens from the synthetic session" {
    const gpa = std.testing.allocator;
    const meta: @import("loader.zig").EngramMeta = .{ .source_tokenizer_len = 128 };
    var bag = try weights.load(gpa, std.testing.io, "zig/src/families/hy_v4_flash/fixtures/synthetic", &meta);
    defer bag.deinit();
    const cfg = try config.parse(gpa, @embedFile("config_synthetic_fixture.json"));
    var impl = try Backend.init(gpa, .{ .config = &cfg, .bag = &bag, .cap = 4, .slots = 1 });
    defer impl.deinit();
    const prompt = [_]u32{ 1, 2, 3 };
    var s = try Stream.init(gpa, .{ .id = "t", .prompt = &prompt, .max_new = 2 });
    defer s.deinit(gpa);
    const b = impl.backend();
    try be.Backend.prefill(b, &s);
    const h = try be.Backend.first(b, &s, prompt.len);
    const t = try be.Backend.read(b, h);
    try std.testing.expect(t < cfg.vocab);
}
