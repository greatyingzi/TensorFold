//! Hy4 Flash text shape from config.json: hybrid attention, Engram, LatentMoE and DFly2.
const std = @import("std");

pub const max_layers = 64;
pub const max_engram_layers = 8;
pub const max_draft_layers = 8;
pub const max_draft_targets = 8;

pub const Kind = enum { sparse, sliding };

pub const Layer = struct {
    kind: Kind = .sparse,
    theta: f32 = 0,
    heads: u32 = 0,
    moe: bool = false,
};

pub const EngramLayer = struct {
    id: u32 = 0,
    primes: [16]u32 = @splat(0),
};

pub const Engram = struct {
    layers: [max_engram_layers]EngramLayer = undefined,
    count: usize = 0,
    ngram: u32 = 0,
    heads: u32 = 0,
    embed: u32 = 0,
    pad: u32 = 0,
    seed: u64 = 0,
    vocab: [2]u32 = .{ 0, 0 },
    fp8: bool = false,
};

pub const Drafter = struct {
    layers: u32 = 0,
    drafts: u32 = 0,
    index_topk: u32 = 0,
    targets: [max_draft_targets]u32 = @splat(0),
    target_count: usize = 0,
};

pub const Vision = struct {
    present: bool = false,
    layers: u32 = 0,
    hidden: u32 = 0,
    patch: u32 = 0,
    merge: u32 = 0,
    out: u32 = 0,
};

pub const Config = struct {
    hidden: u32 = 0,
    vocab: u32 = 0,
    layers: u32 = 0,
    eps: f32 = 0,
    heads: u32 = 0,
    swa_heads: u32 = 0,
    kv_heads: u32 = 0,
    qk_nope: u32 = 0,
    qk_rope: u32 = 0,
    v_dim: u32 = 0,
    swa_qk_nope: u32 = 0,
    swa_qk_rope: u32 = 0,
    swa_v_dim: u32 = 0,
    q_lora: u32 = 0,
    kv_lora: u32 = 0,
    intermediate: u32 = 0,
    moe_intermediate: u32 = 0,
    latent: u32 = 0,
    experts: u32 = 0,
    shared: u32 = 0,
    topk: u32 = 0,
    routed_scale: f32 = 0,
    n_group: u32 = 0,
    topk_group: u32 = 0,
    norm_topk_prob: bool = false,
    swiglu_limit: f32 = 0,
    mrope_section: [3]u32 = @splat(0),
    quant_method: []const u8 = "",
    quant_bits: f32 = 0,
    quant_codebook: []const u8 = "",
    quant_scope: []const u8 = "",
    dense_layers: u32 = 0,
    context: u32 = 0,
    window: u32 = 0,
    index_heads: u32 = 0,
    index_dim: u32 = 0,
    index_ratio: u32 = 0,
    index_group_topk: u32 = 0,
    index_candidate_topk: u32 = 0,
    hc_mult: u32 = 0,
    hc_magnitude: f32 = 0,
    hc_eps: f32 = 0,
    ihc: bool = false,
    sandwich: bool = false,
    gated_mla: bool = false,
    learnable_sink: bool = false,
    engram: Engram = .{},
    dfly: Drafter = .{},
    vision: Vision = .{},
    layer: [max_layers]Layer = undefined,

    pub fn ropeTheta(c: *const Config, kind: Kind) f32 {
        for (c.layer[0..c.layers]) |l| if (l.kind == kind) return l.theta;
        return 0;
    }

    pub fn kvWidth(c: *const Config, kind: Kind) u32 {
        const heads: u32 = if (kind == .sliding) c.swa_heads else c.heads;
        const rope: u32 = if (kind == .sliding) c.swa_qk_rope else c.qk_rope;
        return heads * rope;
    }
};

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadHy4Config;
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadHy4Config,
    };
}

fn int(o: std.json.ObjectMap, key: []const u8) !u32 {
    const v = try field(o, key);
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.BadHy4Config;
    return @intCast(v.integer);
}

fn flag(o: std.json.ObjectMap, key: []const u8) !bool {
    const v = try field(o, key);
    if (v != .bool) return error.BadHy4Config;
    return v.bool;
}

fn list(o: std.json.ObjectMap, key: []const u8, n: usize) ![]const std.json.Value {
    const v = try field(o, key);
    if (v != .array or v.array.items.len < n) return error.BadHy4Config;
    return v.array.items;
}

fn kindOf(name: []const u8) !Kind {
    if (std.mem.eql(u8, name, "deepseek_sparse_attention") or std.mem.eql(u8, name, "indexed_attention")) return .sparse;
    if (std.mem.eql(u8, name, "sliding_attention")) return .sliding;
    return error.UnsupportedHy4Layer;
}

fn ropeTheta(ropes: std.json.ObjectMap, kind: Kind) !f32 {
    const name: []const u8 = if (kind == .sliding) "sliding_attention" else "deepseek_sparse_attention";
    const v = ropes.get(name) orelse return error.MissingHy4Rope;
    if (v != .object) return error.BadHy4Config;
    return @floatCast(try number(try field(v.object, "rope_theta")));
}

fn parseEngram(c: *Config, t: std.json.ObjectMap) !void {
    var e = &c.engram;
    e.ngram = try int(t, "engram_max_ngram_size");
    e.heads = try int(t, "engram_n_head_per_ngram");
    e.embed = try int(t, "engram_n_embed_per_ngram");
    e.pad = try int(t, "engram_pad_id");
    e.fp8 = try flag(t, "engram_fp8_embedding");
    const seed = try field(t, "engram_seed");
    if (seed != .integer or seed.integer < 0) return error.BadHy4Config;
    e.seed = @intCast(seed.integer);
    const vocab = try field(t, "engram_vocab_size");
    if (vocab != .array or vocab.array.items.len != 2) return error.BadHy4Config;
    for (&e.vocab, vocab.array.items) |*dst, v| {
        if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.BadHy4Config;
        dst.* = @intCast(v.integer);
    }
    if (e.ngram != 3 or e.heads != 8 or e.embed != e.heads * 96) return error.UnsupportedHy4Engram;

    const ids = try field(t, "engram_layer_ids");
    if (ids != .array or ids.array.items.len == 0 or ids.array.items.len > max_engram_layers) return error.UnsupportedHy4Engram;
    const sizes = try field(t, "engram_layer_head_vocab_sizes");
    if (sizes != .object) return error.BadHy4Config;
    for (ids.array.items, 0..) |iv, i| {
        if (iv != .integer or iv.integer <= 0 or iv.integer > c.layers) return error.UnsupportedHy4Engram;
        var buf: [24]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "{d}", .{iv.integer});
        const pv = sizes.object.get(key) orelse return error.MissingHy4EngramPrimes;
        if (pv != .array or pv.array.items.len != e.ngram - 1) return error.UnsupportedHy4Engram;
        var prime: usize = 0;
        for (pv.array.items) |group| {
            if (group != .array or group.array.items.len != e.heads) return error.UnsupportedHy4Engram;
            for (group.array.items) |sv| {
                if (sv != .integer or sv.integer <= 0 or sv.integer > std.math.maxInt(u32)) return error.BadHy4Config;
                if (prime >= e.layers[i].primes.len) return error.UnsupportedHy4Engram;
                e.layers[i].primes[prime] = @intCast(sv.integer);
                prime += 1;
            }
        }
        e.layers[i].id = @intCast(iv.integer);
        e.count += 1;
    }
}

fn parseDrafter(c: *Config, top: std.json.ObjectMap) !void {
    const spec = top.get("speculative_config") orelse return;
    if (spec != .object) return error.BadHy4Config;
    var d = &c.dfly;
    d.layers = try int(top, "dfly_num_hidden_layers");
    d.drafts = try int(spec.object, "num_speculative_tokens");
    d.index_topk = try int(top, "dfly_target_index_topk");
    if (d.layers > max_draft_layers) return error.UnsupportedHy4Drafter;
    const targets = try field(top, "dfly_target_index_layer");
    if (targets != .array or targets.array.items.len > max_draft_targets) return error.UnsupportedHy4Drafter;
    for (targets.array.items, 0..) |v, i| {
        if (v != .integer or v.integer <= 0 or v.integer > c.layers) return error.UnsupportedHy4Drafter;
        d.targets[i] = @intCast(v.integer);
        d.target_count += 1;
    }
}

fn parseVision(c: *Config, top: std.json.ObjectMap) !void {
    const v = top.get("vision_config") orelse return;
    if (v != .object) return error.BadHy4Config;
    const o = v.object;
    c.vision.present = true;
    c.vision.layers = try int(o, "num_hidden_layers");
    c.vision.hidden = try int(o, "hidden_size");
    c.vision.patch = try int(o, "patch_size");
    c.vision.merge = try int(o, "spatial_merge_size");
    c.vision.out = try int(o, "out_hidden_size");
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadHy4Config;
    const top = doc.value.object;
    const mt = try field(top, "model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, "hy_v4_flash")) return error.UnsupportedHy4Model;
    const tv = try field(top, "text_config");
    if (tv != .object) return error.BadHy4Config;
    const t = tv.object;

    var c = Config{};
    c.hidden = try int(t, "hidden_size");
    c.layers = try int(t, "num_hidden_layers");
    c.vocab = try int(t, "vocab_size");
    c.eps = @floatCast(try number(try field(t, "rms_norm_eps")));
    c.heads = try int(t, "num_attention_heads");
    c.swa_heads = try int(t, "swa_num_attention_heads");
    c.kv_heads = try int(t, "num_key_value_heads");
    c.qk_nope = try int(t, "qk_nope_head_dim");
    c.qk_rope = try int(t, "qk_rope_head_dim");
    c.v_dim = try int(t, "v_head_dim");
    c.swa_qk_nope = try int(t, "swa_qk_nope_head_dim");
    c.swa_qk_rope = try int(t, "swa_qk_rope_head_dim");
    c.swa_v_dim = try int(t, "swa_v_head_dim");
    c.q_lora = try int(t, "q_lora_rank");
    c.kv_lora = try int(t, "kv_lora_rank");
    c.intermediate = try int(t, "intermediate_size");
    c.moe_intermediate = try int(t, "moe_intermediate_size");
    c.latent = try int(t, "routed_expert_hidden_size");
    c.experts = try int(t, "n_routed_experts");
    c.shared = try int(t, "n_shared_experts");
    c.topk = try int(t, "num_experts_per_tok");
    c.routed_scale = @floatCast(try number(try field(t, "routed_scaling_factor")));
    c.n_group = try int(t, "n_group");
    c.topk_group = try int(t, "topk_group");
    c.norm_topk_prob = try flag(t, "norm_topk_prob");
    c.swiglu_limit = @floatCast(try number(try field(t, "swiglu_limit")));
    c.dense_layers = try int(t, "first_k_dense_replace");
    c.context = try int(t, "max_position_embeddings");
    c.index_heads = try int(t, "compressed_indexer_num_attention_heads");
    c.index_dim = try int(t, "compressed_indexer_head_dim");
    c.index_ratio = try int(t, "compressed_indexer_compress_ratio");
    c.index_group_topk = try int(t, "compressed_indexer_group_topk");
    c.index_candidate_topk = try int(t, "compressed_indexer_candidate_topk");
    c.hc_mult = try int(t, "hc_mult");
    c.hc_magnitude = @floatCast(try number(try field(t, "hc_magnitude")));
    c.hc_eps = @floatCast(try number(try field(t, "hc_eps")));
    c.ihc = try flag(t, "enable_ihc");
    c.sandwich = try flag(t, "sandwich_norm");
    c.gated_mla = try flag(t, "gated_mla");
    c.learnable_sink = try flag(t, "learnable_sink");
    c.window = try int(t, "sliding_window");

    if (c.layers > max_layers or c.experts == 0 or c.experts > 512 or c.topk == 0 or c.topk > c.experts or
        c.latent == 0 or c.latent > c.moe_intermediate or c.hc_mult != 4 or !c.ihc) return error.UnsupportedHy4Model;

    const ropes = try field(t, "rope_parameters");
    if (ropes != .object) return error.BadHy4Config;
    const sparse_theta = try ropeTheta(ropes.object, .sparse);
    const rp = ropes.object.get("sliding_attention") orelse return error.MissingHy4Rope;
    if (rp != .object) return error.BadHy4Config;
    const sec = try field(rp.object, "mrope_section");
    if (sec != .array or sec.array.items.len != 3) return error.BadHy4Config;
    for (&c.mrope_section, sec.array.items) |*dst, v| {
        if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.BadHy4Config;
        dst.* = @intCast(v.integer);
    }
    const sliding_theta = try ropeTheta(ropes.object, .sliding);
    const kinds = try list(t, "layer_types", c.layers);
    const mlps = try list(t, "mlp_layer_types", c.layers);
    for (0..c.layers) |i| {
        if (kinds[i] != .string or mlps[i] != .string) return error.BadHy4Config;
        const kind = try kindOf(kinds[i].string);
        if (!std.mem.eql(u8, mlps[i].string, "dense") and !std.mem.eql(u8, mlps[i].string, "sparse")) return error.BadHy4Config;
        c.layer[i] = .{
            .kind = kind,
            .theta = if (kind == .sliding) sliding_theta else sparse_theta,
            .heads = if (kind == .sliding) c.swa_heads else c.heads,
            .moe = std.mem.eql(u8, mlps[i].string, "sparse"),
        };
    }
    if (top.get("quantization_config")) |qv| {
        if (qv == .object) {
            const q = qv.object;
            if (q.get("quant_method")) |v| {
                if (v == .string) c.quant_method = v.string;
            }
            if (q.get("bits")) |v| {
                if (v == .integer) c.quant_bits = @floatFromInt(v.integer) else if (v == .float) c.quant_bits = @floatCast(v.float);
            }
            if (q.get("codebook")) |v| {
                if (v == .string) c.quant_codebook = v.string;
            }
            if (q.get("scope")) |v| {
                if (v == .string) c.quant_scope = v.string;
            }
        }
    }
    try parseEngram(&c, t);
    try parseDrafter(&c, top);
    try parseVision(&c, top);
    return c;
}

pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    return parse(gpa, bytes);
}

test "hy4 host: the released config parses into the hybrid stack" {
    const c = try parse(std.testing.allocator, @embedFile("config_fixture.json"));
    try std.testing.expectEqual(@as(u32, 42), c.layers);
    try std.testing.expectEqual(@as(u32, 3072), c.hidden);
    try std.testing.expectEqual(@as(u32, 120832), c.vocab);
    try std.testing.expectEqual(@as(u32, 32), c.heads);
    try std.testing.expectEqual(@as(u32, 64), c.swa_heads);
    try std.testing.expectEqual(@as(u32, 512), c.experts);
    try std.testing.expectEqual(@as(u32, 16), c.topk);
    try std.testing.expectEqual(@as(u32, 1536), c.latent);
    try std.testing.expectEqual(@as(u32, 4), c.hc_mult);
    try std.testing.expectEqual(@as(u32, 512), c.window);
    try std.testing.expectEqual(@as(u32, 4), c.index_ratio);
    try std.testing.expectEqual(@as(u32, 512), c.index_group_topk);
    var sparse: usize = 0;
    var sliding: usize = 0;
    for (c.layer[0..c.layers]) |l| switch (l.kind) {
        .sparse => {
            sparse += 1;
            try std.testing.expectEqual(@as(f32, 1e7), l.theta);
            try std.testing.expectEqual(@as(u32, 32), l.heads);
        },
        .sliding => {
            sliding += 1;
            try std.testing.expectEqual(@as(f32, 1e4), l.theta);
            try std.testing.expectEqual(@as(u32, 64), l.heads);
        },
    };
    try std.testing.expectEqual(@as(usize, 12), sparse);
    try std.testing.expectEqual(@as(usize, 30), sliding);
    try std.testing.expectEqual(@as(usize, 4), c.engram.count);
    try std.testing.expectEqual(@as(u32, 2), c.engram.layers[0].id);
    try std.testing.expectEqual(@as(u32, 10312007), c.engram.layers[0].primes[0]);
    try std.testing.expectEqual(@as(u32, 5), c.dfly.layers);
    try std.testing.expectEqual(@as(u32, 25), c.dfly.targets[0]);
    try std.testing.expect(c.vision.present and c.vision.layers == 27);
}

test "hy4 host: a missing rotary entry is refused, not defaulted" {
    const fixture = @embedFile("config_fixture.json");
    const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, "deepseek_sparse_attention\": {", "wrong_name\": {");
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(error.MissingHy4Rope, parse(std.testing.allocator, bytes));
}

test "hy4 host: an unknown layer kind is refused" {
    const fixture = @embedFile("config_fixture.json");
    const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, "\"sliding_attention\",\n      \"sliding_attention\",\n      \"sliding_attention\",\n      \"deepseek_sparse_attention\"", "\"sliding_attention\",\n      \"sliding_attention\",\n      \"sliding_attention\",\n      \"other_attention\"");
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(error.UnsupportedHy4Layer, parse(std.testing.allocator, bytes));
}

test "hy4 host: a truncated Engram prime table is refused" {
    const fixture = @embedFile("config_fixture.json");
    const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, "10312007,", "0,");
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(error.BadHy4Config, parse(std.testing.allocator, bytes));
}


test "hy4 host: the scaled synthetic config parses with the same schema" {
    const c = try parse(std.testing.allocator, @embedFile("config_synthetic_fixture.json"));
    try std.testing.expectEqual(@as(u32, 4), c.layers);
    try std.testing.expectEqual(@as(u32, 64), c.hidden);
    try std.testing.expectEqual(@as(u32, 128), c.vocab);
    try std.testing.expectEqual(@as(u32, 4), c.experts);
    try std.testing.expectEqual(@as(u32, 2), c.topk);
    try std.testing.expectEqual(@as(usize, 1), c.engram.count);
    try std.testing.expectEqual(@as(u32, 2), c.engram.layers[0].id);
    try std.testing.expectEqual(@as(u32, 1), c.dfly.layers);
    try std.testing.expectEqual(@as(u32, 2), c.dfly.drafts);
}
