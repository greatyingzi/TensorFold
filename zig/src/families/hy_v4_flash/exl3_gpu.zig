//! Hy4 Flash EXL3 pack layout on the GPU: trellis/suh/svh over the shared kernels.
const std = @import("std");

/// One quantized projection's raw EXL3 tensors, as safetensors bytes.
pub const Raw = struct {
    trellis: []const u8 = &.{}, // i16 [k/16, n/16, 64]
    suh: []const u8 = &.{}, // f16 [k]
    svh: []const u8 = &.{}, // f16 [n]
};

/// Hy4's quantized projections of one decoder layer (MoE layers; layer 0 is dense MLP).
pub const Layer = struct {
    q_a: Raw = .{},
    q_b: Raw = .{},
    kv_a: Raw = .{},
    kv_b: Raw = .{},
    o: Raw = .{},
    shared_up: Raw = .{},
    shared_gate: Raw = .{},
    shared_down: Raw = .{},
    routed_down: Raw = .{},
    routed_up: Raw = .{},
    experts_gate_up: []Raw = &.{},
    experts_down: []Raw = &.{},
};

/// The checkpoint suffix of every quantized projection, with its [n, k].
pub const Projection = struct { name: []const u8, n: usize, k: usize };

pub const attention = [_]Projection{
    .{ .name = "self_attn.q_a_proj", .n = 1024, .k = 3072 },
    .{ .name = "self_attn.q_b_proj", .n = 8192, .k = 1024 },
    .{ .name = "self_attn.kv_a_proj_with_mqa", .n = 576, .k = 3072 },
    .{ .name = "self_attn.kv_b_proj", .n = 14336, .k = 512 },
    .{ .name = "self_attn.o_proj", .n = 3072, .k = 8192 },
};

pub const moe = [_]Projection{
    .{ .name = "mlp.shared_experts.up_proj", .n = 1536, .k = 3072 },
    .{ .name = "mlp.shared_experts.gate_proj", .n = 1536, .k = 3072 },
    .{ .name = "mlp.shared_experts.down_proj", .n = 3072, .k = 1536 },
    .{ .name = "mlp.routed_expert_down_proj", .n = 1536, .k = 3072 },
    .{ .name = "mlp.routed_expert_up_proj", .n = 3072, .k = 1536 },
};

/// The latent experts: gate_up [2*inter, latent] and down [latent, inter], 512 of each.
pub const expert_gate_up = Projection{ .name = "mlp.experts.gate_up", .n = 3072, .k = 1536 };
pub const expert_down = Projection{ .name = "mlp.experts.down", .n = 1536, .k = 1536 };

/// Every EXL3 tensor name of layer `li` (experts excluded) with its [n, k].
pub fn projectionNames(a: std.mem.Allocator, li: usize, out: *std.ArrayList([]const u8)) !void {
    for (attention) |p| try out.append(a, try std.fmt.allocPrint(a, "model.layers.{d}.{s}", .{ li, p.name }));
    if (li >= 1) {
        for (moe) |p| try out.append(a, try std.fmt.allocPrint(a, "model.layers.{d}.{s}", .{ li, p.name }));
    }
}

/// The expected trellis element count of an EXL3 projection: (k/16) * (n/16) * 64 i16.
pub fn trellisElems(p: Projection) usize {
    return (p.k / 16) * (p.n / 16) * 64;
}
