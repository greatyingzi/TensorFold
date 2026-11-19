//! Hy4 Flash four-stream IHC glue kernels (hy4_ihc.cu).
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = @import("stream.zig").Stream;
const Args = launch_.Args;

pub const Ihc = struct {
    mod: Module,
    collapse_fn: Function,
    update_fn: Function,
    head_fn: Function,

    pub fn load(d: *const Driver) !Ihc {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.hy4_ihc);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .collapse_fn = try mod.function("tf_hy4_ihc_collapse"),
            .update_fn = try mod.function("tf_hy4_stream_update"),
            .head_fn = try mod.function("tf_hy4_hc_head"),
        };
    }

    pub fn unload(self: *Ihc) void {
        self.mod.unload();
    }

    /// streams [rows][HC][H] bf16 collapsed into out [rows][H] bf16.
    pub fn collapse(self: *const Ihc, s: Stream, streams: u64, fn_w: u64, base: u64, scale: u64, out: u64, rows: usize, H: usize, HC: usize, eps: f32, magnitude: f32, post: bool) !void {
        if (rows == 0 or H == 0 or HC == 0) return;
        const int = struct {
            fn f(v: usize) c_int {
                return @intCast(v);
            }
        }.f;
        var a: Args = .{};
        a.add(streams);
        a.add(fn_w);
        a.add(base);
        a.add(scale);
        a.add(out);
        a.add(int(rows));
        a.add(int(H));
        a.add(int(HC));
        a.add(eps);
        a.add(magnitude);
        a.add(@as(c_int, if (post) 1 else 0));
        try launch_.launch(self.collapse_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// streams[r][c][h] = post[c] * branch[r][h] + streams[r][c][h], in place.
    pub fn update(self: *const Ihc, s: Stream, streams: u64, branch: u64, post: u64, rows: usize, H: usize, HC: usize) !void {
        if (rows == 0 or H == 0 or HC == 0) return;
        const int = struct {
            fn f(v: usize) c_int {
                return @intCast(v);
            }
        }.f;
        var a: Args = .{};
        a.add(streams);
        a.add(branch);
        a.add(post);
        a.add(int(rows));
        a.add(int(H));
        a.add(int(HC));
        try launch_.launch(self.update_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(HC) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Final stream collapse [rows][HC][H] -> out [rows][H] bf16.
    pub fn head(self: *const Ihc, s: Stream, streams: u64, fn_w: u64, base: u64, scale: u64, out: u64, rows: usize, H: usize, HC: usize, eps: f32) !void {
        if (rows == 0 or H == 0 or HC == 0) return;
        const int = struct {
            fn f(v: usize) c_int {
                return @intCast(v);
            }
        }.f;
        var a: Args = .{};
        a.add(streams);
        a.add(fn_w);
        a.add(base);
        a.add(scale);
        a.add(out);
        a.add(int(rows));
        a.add(int(H));
        a.add(int(HC));
        a.add(eps);
        try launch_.launch(self.head_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }
};
