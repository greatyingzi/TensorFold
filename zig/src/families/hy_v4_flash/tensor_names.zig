//! Checkpoint tensor names to engine names for Hy4 Flash, spelled out to avoid short-rule collisions.
const std = @import("std");

const Rule = struct { from: []const u8, to: []const u8 };

/// Text renames from the official conversion mapping. Long HC paths go before any short rule.
const text_rules = [_]Rule{
    .{ .from = ".self_attn.learnable_sink_param", .to = ".self_attn.sinks" },
    .{ .from = ".self_attn.linear_gate", .to = ".self_attn.gate_proj" },
    .{ .from = ".hc_attn_layer.hc_pre.hc_fn", .to = ".attn_hc.fn" },
    .{ .from = ".hc_attn_layer.hc_pre.hc_base", .to = ".attn_hc.base" },
    .{ .from = ".hc_attn_layer.hc_pre.hc_scale", .to = ".attn_hc.scale" },
    .{ .from = ".hc_mlp_layer.hc_pre.hc_fn", .to = ".ffn_hc.fn" },
    .{ .from = ".hc_mlp_layer.hc_pre.hc_base", .to = ".ffn_hc.base" },
    .{ .from = ".hc_mlp_layer.hc_pre.hc_scale", .to = ".ffn_hc.scale" },
    .{ .from = "hc_head.hc_head_fn", .to = "hc_head.hc_fn" },
    .{ .from = "hc_head.hc_head_base", .to = "hc_head.hc_base" },
    .{ .from = "hc_head.hc_head_scale", .to = "hc_head.hc_scale" },
};

/// The tower is Megatron-named; these rules are scoped to `vision_tower.` and never touch text layers.
const vision_rules = [_]Rule{
    .{ .from = "mlp.dense_h_to_4h", .to = "mlp.fc1" },
    .{ .from = "mlp.dense_4h_to_h", .to = "mlp.fc2" },
    .{ .from = "input_layernorm", .to = "layer_norm1" },
    .{ .from = "post_attention_layernorm", .to = "layer_norm2" },
};

fn stripPrefix(name: []const u8) []const u8 {
    if (std.mem.startsWith(u8, name, "model.")) return name["model.".len..];
    return name;
}

fn apply(gpa: std.mem.Allocator, start: []u8, rules: []const Rule) ![]u8 {
    var current = start;
    for (rules) |rule| {
        if (std.mem.indexOf(u8, current, rule.from) == null) continue;
        const next = std.mem.replaceOwned(u8, gpa, current, rule.from, rule.to) catch |e| {
            gpa.free(current);
            return e;
        };
        gpa.free(current);
        current = next;
    }
    return current;
}

/// Returns an allocator-owned engine name. Drafter tensors keep `dfly_layers` for the drafter loader.
pub fn rename(gpa: std.mem.Allocator, checkpoint_name: []const u8) ![]u8 {
    var current = try gpa.dupe(u8, stripPrefix(checkpoint_name));
    if (std.mem.startsWith(u8, current, "vit.")) {
        const next = std.mem.replaceOwned(u8, gpa, current, "vit.", "vision_tower.") catch |e| {
            gpa.free(current);
            return e;
        };
        gpa.free(current);
        current = next;
    }
    const scoped = if (std.mem.startsWith(u8, current, "vision_tower.")) vision_rules[0..] else text_rules[0..];
    current = try apply(gpa, current, scoped);
    return current;
}

fn check(gpa: std.mem.Allocator, from: []const u8, want: []const u8) !void {
    const got = try rename(gpa, from);
    defer gpa.free(got);
    try std.testing.expectEqualStrings(want, got);
}

test "hy4 host: text renames match the official conversion mapping" {
    const gpa = std.testing.allocator;
    try check(gpa, "model.layers.0.self_attn.learnable_sink_param", "layers.0.self_attn.sinks");
    try check(gpa, "model.layers.0.self_attn.linear_gate.weight", "layers.0.self_attn.gate_proj.weight");
    try check(gpa, "model.layers.0.hc_attn_layer.hc_pre.hc_fn", "layers.0.attn_hc.fn");
    try check(gpa, "model.layers.0.hc_attn_layer.hc_pre.hc_base", "layers.0.attn_hc.base");
    try check(gpa, "model.layers.0.hc_attn_layer.hc_pre.hc_scale", "layers.0.attn_hc.scale");
    try check(gpa, "model.layers.0.hc_mlp_layer.hc_pre.hc_fn", "layers.0.ffn_hc.fn");
    try check(gpa, "model.layers.0.hc_mlp_layer.hc_pre.hc_base", "layers.0.ffn_hc.base");
    try check(gpa, "model.layers.0.hc_mlp_layer.hc_pre.hc_scale", "layers.0.ffn_hc.scale");
    try check(gpa, "model.hc_head.hc_head_fn", "hc_head.hc_fn");
    try check(gpa, "model.hc_head.hc_head_base", "hc_head.hc_base");
    try check(gpa, "model.hc_head.hc_head_scale", "hc_head.hc_scale");
}

test "hy4 host: the HC guard and released drafter/expert names survive unchanged" {
    const gpa = std.testing.allocator;
    try check(gpa, "model.layers.0.unified_per_hc_scales", "layers.0.unified_per_hc_scales");
    try check(gpa, "model.layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.gate_up_proj");
    try check(gpa, "model.layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.down_proj");
    try check(gpa, "model.dfly_layers.0.self_attn.q_a_proj.weight", "dfly_layers.0.self_attn.q_a_proj.weight");
    try check(gpa, "model.layers.0.input_layernorm.weight", "layers.0.input_layernorm.weight");
}

test "hy4 host: vision tensors use the tower namespace and Megatron MLP names" {
    const gpa = std.testing.allocator;
    try check(gpa, "vit.layers.0.mlp.dense_h_to_4h.weight", "vision_tower.layers.0.mlp.fc1.weight");
    try check(gpa, "vit.layers.0.input_layernorm.weight", "vision_tower.layers.0.layer_norm1.weight");
    try check(gpa, "vit.layers.0.post_attention_layernorm.weight", "vision_tower.layers.0.layer_norm2.weight");
}
