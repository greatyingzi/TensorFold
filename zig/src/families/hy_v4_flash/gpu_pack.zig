//! Load a complete Hy4 Flash EXL3 pack into device weights for decode.
const std = @import("std");
const cuda = @import("cuda");
const sf = @import("core").safetensors;

pub const Config = struct {
    hidden: usize = 3072,
    layers: usize = 42,
    hc: usize = 4,
    heads: usize = 32,
    max_heads: usize = 64, // sliding layers' 64; attention buffers take the max
    routed_scale: f32 = 1.0,
    topk: usize = 16,
    swiglu_limit: f32 = 10.0,
    kv_lora: usize = 512,
    q_lora: usize = 1024,
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
    heads: usize = 32, // this layer's attention head count (64 on sliding layers)
    kv_a_w: u64 = 0, // BF16 original for the n=576 projection (EXL3 needs n % 128 == 0)
    kv_b_w: u64 = 0, // BF16 original kv_b weight, used to rebuild absorb/expand matrices
    kv_b: Linear = undefined,
    o: Linear = undefined,
    up: Linear = undefined,
    gate: Linear = undefined,
    attn_gate_w: u64 = 0, // self_attn.linear_gate.weight (BF16, not EXL3)
    routed_down: Linear = undefined, // mlp.routed_expert_down_proj (EXL3)
    routed_up: Linear = undefined, // mlp.routed_expert_up_proj (EXL3)
    router_w: u64 = 0, // mlp.gate.weight BF16 [512, 3072]
    router_bias: u64 = 0, // mlp.gate.e_score_correction_bias F32 [512]
    combine_norm: u64 = 0, // mlp.combine_output_norm.weight, fp32 on device
    dense_gate: u64 = 0, // layer 0's dense MLP (BF16)
    dense_up: u64 = 0,
    dense_down: u64 = 0,
    down: Linear = undefined,
    experts: ?cuda.exl3_experts.Layer = null,
};

pub const Pack = struct {
    a: std.mem.Allocator,
    up: cuda.exl3_upload.Uploader,
    cfg: Config,
    w: [42]LayerWeights = @splat(.{}),
    embed: u64 = 0,
    embed_host: []const u8 = &.{},
    lm_head: u64 = 0,
    lm_head_w: u64 = 0, // BF16 original; vocab 120000 is not 128-aligned
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


/// One layer's 512 fused gate/up + down experts, uploaded as pointer tables and stacked scales.
fn loadExperts(a: std.mem.Allocator, io: std.Io, p: *Pack, files: *std.StringHashMap(*sf.File), wm: std.json.ObjectMap, dir: []const u8, li: usize) !cuda.exl3_experts.Layer {
    const E: usize = 512;
    const dims: usize = 1536;
    const width: usize = 1536;
    var gt = try a.alloc(u64, E);
    var dt = try a.alloc(u64, E);
    const gu_suh = try a.alloc(u8, E * dims * 2);
    const gu_svh = try a.alloc(u8, E * 2 * width * 2);
    const d_suh = try a.alloc(u8, E * width * 2);
    const d_svh = try a.alloc(u8, E * dims * 2);
    for (0..E) |e| {
        const names = [_][]const u8{
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.trellis", .{ li, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.suh", .{ li, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.gate_up.{d}.svh", .{ li, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.trellis", .{ li, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.suh", .{ li, e }),
            try std.fmt.allocPrint(a, "model.layers.{d}.mlp.experts.down.{d}.svh", .{ li, e }),
        };
        var t: [6]sf.Tensor = undefined;
        for (names, 0..) |nm, ni| t[ni] = try getTensor(a, io, files, wm, dir, nm);
        gt[e] = try p.up.dev(t[0].bytes);
        dt[e] = try p.up.dev(t[3].bytes);
        @memcpy(gu_suh[e * dims * 2 ..][0 .. dims * 2], t[1].bytes);
        @memcpy(gu_svh[e * 2 * width * 2 ..][0 .. 2 * width * 2], t[2].bytes);
        @memcpy(d_suh[e * width * 2 ..][0 .. width * 2], t[4].bytes);
        @memcpy(d_svh[e * dims * 2 ..][0 .. dims * 2], t[5].bytes);
    }
    const k2 = try a.alloc(i32, E);
    @memset(k2, 6);
    return .{
        .gate = try p.up.dev(std.mem.sliceAsBytes(gt)),
        .up = try p.up.dev(std.mem.sliceAsBytes(gt)),
        .down = try p.up.dev(std.mem.sliceAsBytes(dt)),
        .gate_k2 = try p.up.dev(std.mem.sliceAsBytes(k2)),
        .up_k2 = try p.up.dev(std.mem.sliceAsBytes(k2)),
        .down_k2 = try p.up.dev(std.mem.sliceAsBytes(k2)),
        .suh_g = try p.up.dev(gu_suh),
        .suh_u = try p.up.dev(gu_suh),
        .svh_g = try p.up.dev(gu_svh),
        .svh_u = try p.up.dev(gu_svh),
        .suh_d = try p.up.dev(d_suh),
        .svh_d = try p.up.dev(d_svh),
        .experts = E,
        .dims = dims,
        .width = width,
        .codebook = .mul1,
        .uniform_k2 = 6,
        .gu_stride = width / 16,
        .limit = p.cfg.swiglu_limit,
        .act_mode = 0,
    };
}

fn dense(a: std.mem.Allocator, io: std.Io, p: *Pack, files: *std.StringHashMap(*sf.File), wm: std.json.ObjectMap, dir: []const u8, name: []const u8) !Linear {
    const t = try getTensor(a, io, files, wm, dir, try std.fmt.allocPrint(a, "{s}.trellis", .{name}));
    const su = try getTensor(a, io, files, wm, dir, try std.fmt.allocPrint(a, "{s}.suh", .{name}));
    const sv = try getTensor(a, io, files, wm, dir, try std.fmt.allocPrint(a, "{s}.svh", .{name}));
    const cb: cuda.exl3.Codebook = if (wm.contains(try std.fmt.allocPrint(a, "{s}.mcg", .{name}))) .mcg else .mul1;
    const halves: u32 = @intCast(t.shape[2] / 8);
    // The trellis carries the true geometry: [k/16, n/16, words]. Sliding-window
    // layers have 64 heads, DSA layers 32, so n differs per layer.
    const n: usize = t.shape[1] * 16;
    const k: usize = su.numel();
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


/// Upload a BF16 tensor expanded to fp32 (the shared norms kernels take fp32 weights).
fn scalarF32(a: std.mem.Allocator, io: std.Io, p: *Pack, files: *std.StringHashMap(*sf.File), wm: std.json.ObjectMap, dir: []const u8, name: []const u8) !u64 {
    const t = try getTensor(a, io, files, wm, dir, name);
    const bf = t.bytes.len / 2;
    const out = try a.alloc(f32, bf);
    for (out, 0..) |*v, i| {
        const bits = std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little);
        v.* = @bitCast(@as(u32, bits) << 16);
    }
    return p.up.dev(std.mem.sliceAsBytes(out));
}

pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Pack {
    const idx_path = try join(a, dir, "model.safetensors.index.json");
    const idx_text = try std.Io.Dir.cwd().readFileAlloc(io, idx_path, a, .limited(64 << 20));
    var doc = try std.json.parseFromSlice(std.json.Value, a, idx_text, .{});
    const wm = doc.value.object.get("weight_map").?.object;
    var files = std.StringHashMap(*sf.File).init(a);
    var cfg: Config = .{};
    // Read routing scale from the pack's text_config (the converter preserves it).
    const cfg_path = try join(a, dir, "config.json");
    if (std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(16 << 20))) |ct| {
        var cdoc = std.json.parseFromSlice(std.json.Value, a, ct, .{}) catch null;
        if (cdoc) |*cd| {
            if (cd.value.object.get("text_config")) |tc| {
                const getf = struct {
                    fn get(o: std.json.ObjectMap, key: []const u8, cur: f32) f32 {
                        if (o.get(key)) |v| return switch (v) {
                            .float => |fv| @floatCast(fv),
                            .integer => |iv| @floatFromInt(iv),
                            else => cur,
                        };
                        return cur;
                    }
                }.get;
                const getn = struct {
                    fn get(o: std.json.ObjectMap, key: []const u8, cur: usize) usize {
                        if (o.get(key)) |v| return switch (v) {
                            .integer => |iv| @intCast(iv),
                            .float => |fv| @intFromFloat(fv),
                            else => cur,
                        };
                        return cur;
                    }
                }.get;
                cfg.routed_scale = getf(tc.object, "routed_scaling_factor", cfg.routed_scale);
                cfg.eps = getf(tc.object, "rms_norm_eps", cfg.eps);
                cfg.hc_eps = getf(tc.object, "hc_eps", cfg.hc_eps);
                cfg.hc_magnitude = getf(tc.object, "hc_magnitude", cfg.hc_magnitude);
                cfg.swiglu_limit = getf(tc.object, "swiglu_limit", cfg.swiglu_limit);
                cfg.topk = getn(tc.object, "num_experts_per_tok", cfg.topk);
                cfg.hidden = getn(tc.object, "hidden_size", cfg.hidden);
                cfg.nope = getn(tc.object, "qk_nope_head_dim", cfg.nope);
                cfg.rope = getn(tc.object, "qk_rope_head_dim", cfg.rope);
                cfg.q_lora = getn(tc.object, "q_lora_rank", cfg.q_lora);
                cfg.kv_lora = getn(tc.object, "kv_lora_rank", cfg.kv_lora);
                cfg.vdim = getn(tc.object, "v_head_dim", cfg.vdim);
            }
        }
    } else |_| {}
    var p: Pack = .{ .a = a, .up = .{ .d = d, .gpa = a, .max_buffers = 200000 }, .cfg = cfg };
    errdefer p.deinit();
    p.embed_host = (try getTensor(a, io, &files, wm, dir, "model.embed_tokens.weight")).bytes;
    p.embed = try p.up.dev(p.embed_host);
    p.lm_head = try scalar(a, io, &p, &files, wm, dir, "lm_head.weight");
    // The converter stores lm_head as fp16; dense_rows wants bf16. Convert once on load.
    const lmt = try getTensor(a, io, &files, wm, dir, "lm_head.weight");
    const lm16 = std.mem.bytesAsSlice(f16, lmt.bytes);
    const lmbf = try a.alloc(u16, lm16.len);
    for (lm16, 0..) |v, i| {
        const f: f32 = @floatCast(v);
        const bits: u32 = @bitCast(f);
        lmbf[i] = @truncate(bits >> 16);
    }
    p.lm_head_w = try p.up.dev(std.mem.sliceAsBytes(lmbf));
    p.final_norm = try scalarF32(a, io, &p, &files, wm, dir, "model.norm.weight");
    p.head_fn = try scalar(a, io, &p, &files, wm, dir, "model.hc_head.hc_head_fn");
    p.head_base = try scalar(a, io, &p, &files, wm, dir, "model.hc_head.hc_head_base");
    p.head_scale = try scalar(a, io, &p, &files, wm, dir, "model.hc_head.hc_head_scale");
    for (0..p.cfg.layers) |li| {
        const pre = try std.fmt.allocPrint(a, "model.layers.{d}.", .{li});
        const L = &p.w[li];
        L.q_a = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.q_a_proj", .{pre}));
        L.q_b = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.q_b_proj", .{pre}));
        L.heads = L.q_b.n / @as(u32, @intCast(p.cfg.nope + p.cfg.rope));
        L.heads = L.q_b.n / @as(u32, @intCast(p.cfg.nope + p.cfg.rope));
        L.kv_a_w = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.kv_a_proj_with_mqa.weight", .{pre}));
        L.kv_b_w = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.kv_b_proj.weight", .{pre}));
        L.attn_gate_w = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.linear_gate.weight", .{pre}));
        L.kv_b = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.kv_b_proj", .{pre}));
        L.o = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.o_proj", .{pre}));
        if (li > 0) {
            L.routed_down = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.routed_expert_down_proj", .{pre}));
            L.routed_up = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.routed_expert_up_proj", .{pre}));
            L.router_w = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.gate.weight", .{pre}));
            L.router_bias = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.gate.e_score_correction_bias", .{pre}));
            L.combine_norm = try scalarF32(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.combine_output_norm.weight", .{pre}));
            L.experts = try loadExperts(a, io, &p, &files, wm, dir, li);
            L.up = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.shared_experts.up_proj", .{pre}));
            L.gate = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.shared_experts.gate_proj", .{pre}));
            L.down = try dense(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.shared_experts.down_proj", .{pre}));
        } else {
            L.dense_gate = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.gate_proj.weight", .{pre}));
            L.dense_up = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.up_proj.weight", .{pre}));
            L.dense_down = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}mlp.down_proj.weight", .{pre}));
        }
        p.attn_fn[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_attn_layer.hc_pre.hc_fn", .{pre}));
        p.attn_base[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_attn_layer.hc_pre.hc_base", .{pre}));
        p.attn_scale[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_attn_layer.hc_pre.hc_scale", .{pre}));
        p.ffn_fn[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_mlp_layer.hc_pre.hc_fn", .{pre}));
        p.ffn_base[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_mlp_layer.hc_pre.hc_base", .{pre}));
        p.ffn_scale[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}hc_mlp_layer.hc_pre.hc_scale", .{pre}));
        p.input_norm[li] = try scalarF32(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}input_layernorm.weight", .{pre}));
        p.post_attn_norm[li] = try scalarF32(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}post_attention_layernorm.weight", .{pre}));
        p.q_a_norm[li] = try scalarF32(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.q_a_layernorm.weight", .{pre}));
        p.kv_a_norm[li] = try scalarF32(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.kv_a_layernorm.weight", .{pre}));
        p.post_proj_norm[li] = try scalarF32(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.post_proj_norm.weight", .{pre}));
        p.sinks[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.learnable_sink_param", .{pre}));
        p.gate_proj[li] = try scalar(a, io, &p, &files, wm, dir, try std.fmt.allocPrint(a, "{s}self_attn.linear_gate.weight", .{pre}));
    }
    return p;
}
