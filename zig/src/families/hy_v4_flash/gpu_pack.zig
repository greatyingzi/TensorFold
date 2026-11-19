//! Load a complete Hy4 Flash EXL3 pack into device weights for decode.
const std = @import("std");
const cuda = @import("cuda");
const sf = @import("core").safetensors;

pub const Config = struct {
    hidden: usize = 3072,
    layers: usize = 42,
    hc: usize = 4,
    heads: usize = 32,
    kv_lora: usize = 512,
    nope: usize = 192,
    rope: usize = 64,
    vdim: usize = 256,
    eps: f32 = 1e-6,
    hc_eps: f32 = 1e-6,
    hc_magnitude: f32 = 2.0,
    vocab: usize = 120000,
};

fn join(a: std.mem.Allocator, dir: []const u8, f: []const u8) ![]const u8 {
    return std.fs.path.join(a, &.{ dir, f });
}

/// One dense EXL3 projection on the device.
pub const Linear = cuda.exl3.Weight;

pub const LayerWeights = struct {
    q_a: Linear = undefined,
    q_b: Linear = undefined,
    kv_a: Linear = undefined,
    kv_b: Linear = undefined,
    o: Linear = undefined,
    up: Linear = undefined,
    gate: Linear = undefined,
    down: Linear = undefined,
    experts: ?cuda.exl3_experts.Layer = null,
};

pub const Pack = struct {
    a: std.mem.Allocator,
    up: cuda.exl3_upload.Uploader,
    cfg: Config,
    w: [42]LayerWeights = @splat(.{}),
    embed: u64 = 0,
    lm_head: u64 = 0,
    final_norm: u64 = 0,
    // HC projections: 2*hc rows of hc*hidden each, per layer; and head.
    attn_fn: [42]u64 = @splat(0),
    attn_base: [42]u64 = @splat(0),
    attn_scale: [42]u64 = @splat(0),
    ffn_fn: [42]u64 = @splat(0),
    ffn_base: [42]u64 = @splat(0),
    ffn_scale: [42]u64 = @splat(0),
    head_fn: u64 = 0,
    head_base: u64 = 0,
    head_scale: u64 = 0,
    input_norm: [42]u64 = @splat(0),
    post_attn_norm: [42]u64 = @splat(0),
    q_a_norm: [42]u64 = @splat(0),
    kv_a_norm: [42]u64 = @splat(0),
    post_proj_norm: [42]u64 = @splat(0),
    sinks: [42]u64 = @splat(0),
    gate_proj: [42]u64 = @splat(0),

    pub fn deinit(self: *Pack) void {
        self.up.deinit();
    }
};

fn getTensor(a: std.mem.Allocator, io: std.Io, files: *std.StringHashMap(*sf.File), wm: std.json.ObjectMap, dir: []const u8, name: []const u8) !sf.Tensor {
    const shard = wm.get(name) orelse {
        std.debug.print("PACK_MISSING {s}\n", .{name});
        return error.MissingPackTensor;
    };
    const file = files.get(shard.string) orelse blk: {
        const path = try join(a, dir, shard.string);
        const p = try a.create(sf.File);
        p.* = try sf.File.open(a, io, path);
        try files.put(shard.string, p);
        break :blk p;
    };
    return file.get(name) orelse error.MissingPackTensor;
}

fn dense(a: std.mem.Allocator, io: std.Io, p: *Pack, files: *std.StringHashMap(*sf.File), wm: std.json.ObjectMap, dir: []const u8, name: []const u8, k: usize, n: usize) !Linear {
    const t = try getTensor(a, io, files, wm, dir, try std.fmt.allocPrint(a, "{s}.trellis", .{name}));
    const su = try getTensor(a, io, files, wm, dir, try std.fmt.allocPrint(a, "{s}.suh", .{name}));
    const sv = try getTensor(a, io, files, wm, dir, try std.fmt.allocPrint(a, "{s}.svh", .{name}));
    const cb: cuda.exl3.Codebook = if (wm.contains(try std.fmt.allocPrint(a, "{s}.mcg", .{name}))) .mcg else .mul1;
    const halves: u32 = @intCast(t.shape[2] / 8);
    const st = cuda.exl3.strides(k, n, halves, false);
    return .{
        .trellis = try p.up.dev(t.bytes),
        .suh = try p.up.dev(su.bytes),
        .svh = try p.up.dev(sv.bytes),
        .k = @intCast(k), .n = @intCast(n), .halves = halves, .codebook = cb,
        .stride_k = st.k, .stride_nb = st.nb,
    };
}

fn scalar(a: std.mem.Allocator, io: std.Io, p: *Pack, files: *std.StringHashMap(*sf.File), wm: std.json.ObjectMap, dir: []const u8, name: []const u8) !u64 {
    const t = try getTensor(a, io, files, wm, dir, name);
    return p.up.dev(t.bytes);
}

pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Pack {
    const idx_path = try join(a, dir, "model.safetensors.index.json");
    const idx_text = try std.Io.Dir.cwd().readFileAlloc(io, idx_path, a, .limited(64 << 20));
    var doc = try std.json.parseFromSlice(std.json.Value, a, idx_text, .{});
    const wm = doc.value.object.get("weight_map").?.object;
    var files = std.StringHashMap(*sf.File).init(a);
    var p: Pack = .{ .a = a, .up = .{ .d = d, .gpa = a, .max_buffers = 20000 }, .cfg = .{} };
    errdefer p.deinit();
    p.embed = try scalar(a, io, &p, &files, wm, dir, "model.embed_tokens.weight");
    p.lm_head = try scalar(a, io, &p, &files, wm, dir, "lm_head.weight");
    p.final_norm = try scalar(a, io, &p, &files, wm, dir, "model.norm.weight");
    p.head_fn = try scalar(a, io, &p, &files, wm, dir, "model.hc_head.hc_head_fn");
    p.head_base = try scalar(a, io, &p, &files, wm, dir, "model.hc_head.hc_head_base");
    p.head_scale = try scalar(a, io, &p, &files, wm, dir, "model.hc_head.hc_head_scale");
    for (0..p.cfg.layers) |li| {
        const pre = try std.fmt.allocPrint(a, "model.layers.{d}.", .{li});
        const L = &p.w[li];
        L.q_a = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.q_a_proj", .{pre}), 3072, 1024);
        L.q_b = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.q_b_proj", .{pre}), 1024, 8192);
        L.kv_a = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.kv_a_proj_with_mqa", .{pre}), 3072, 576);
        L.kv_b = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.kv_b_proj", .{pre}), 512, 14336);
        L.o = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.o_proj", .{pre}), 8192, 3072);
        if (li > 0) {
            L.up = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.shared_experts.up_proj", .{pre}), 3072, 1536);
            L.gate = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.shared_experts.gate_proj", .{pre}), 3072, 1536);
            L.down = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.shared_experts.down_proj", .{pre}), 1536, 3072);
        }
        p.attn_fn[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_attn_layer.hc_pre.hc_fn", .{pre}));
        p.attn_base[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_attn_layer.hc_pre.hc_base", .{pre}));
        p.attn_scale[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_attn_layer.hc_pre.hc_scale", .{pre}));
        p.ffn_fn[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_mlp_layer.hc_pre.hc_fn", .{pre}));
        p.ffn_base[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_mlp_layer.hc_pre.hc_base", .{pre}));
        p.ffn_scale[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_mlp_layer.hc_pre.hc_scale", .{pre}));
        p.input_norm[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}input_layernorm.weight", .{pre}));
        p.post_attn_norm[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}post_attention_layernorm.weight", .{pre}));
        p.q_a_norm[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.q_a_layernorm.weight", .{pre}));
        p.kv_a_norm[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.kv_a_layernorm.weight", .{pre}));
        p.post_proj_norm[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.post_proj_norm.weight", .{pre}));
        p.sinks[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.learnable_sink_param", .{pre}));
        p.gate_proj[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.linear_gate.weight", .{pre}));
    }
    return p;
}
