#!/usr/bin/env python3
"""Config, names, shapes and safetensors writer for the tiny Hy4-Flash fixture."""
from __future__ import annotations

import json
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]          # tools/zig/hy4_synthetic.py -> repo root
FAM = REPO / "zig" / "src" / "families" / "hy_v4_flash"
CONFIG_FIXTURE = FAM / "config_fixture.json"
INDEX_FIXTURE = FAM / "index_fixture.json"

BF16_ZERO = 0x0000


def _load_json(path, *, detect_dups=False):
    """Load JSON; optionally fail on duplicate keys (used by --verify)."""
    hooks = {}
    if detect_dups:
        def hook(pairs):
            seen = {}
            for k, v in pairs:
                if k in seen:
                    raise ValueError("duplicate JSON key: %r" % k)
                seen[k] = v
            return seen
        hooks["object_pairs_hook"] = hook
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh, **hooks)


def bf16_bits(x):
    """Round a python float to the nearest BF16 bit pattern (as uint16)."""
    b = int(np.float32(x).view(np.uint32))
    return (b + 0x7FFF + ((b >> 16) & 1)) >> 16


def zeros(shape):
    n = 1
    for dd in shape:
        n *= int(dd)
    return np.full(n, BF16_ZERO, dtype=np.uint16)


def const(shape, value):
    n = 1
    for dd in shape:
        n *= int(dd)
    return np.full(n, bf16_bits(value), dtype=np.uint16)


def varied(shape):
    n = 1
    for dd in shape:
        n *= int(dd)
    vals = ((np.arange(n, dtype=np.float32) % 13) - np.float32(6)) * np.float32(0.02)
    return (vals.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


# ---------------------------------------------------------------------------
DEFAULTS = dict(
    layers=4, hidden=64, heads=8, swa_heads=8, kv_heads=8,
    qk_nope=16, qk_rope=8, v_dim=16, q_lora=16, kv_lora=16,
    intermediate=128, moe_intermediate=32, latent=32,
    experts=4, topk=2, shared=1, vocab=128,
    engram_layers=(2,),                 # 1-based, as in the official config
    sliding_window=32, index_ratio=2, index_heads=2, index_dim=16,
    index_group_topk=4, index_candidate_topk=8,
    dfly_layers=1, spec_tokens=2, vision_layers=2, vision_hidden=64,
    shards=4, seed=0,
)

# small distinct primes for the engram bucket table (16 = 2 orders x 8 heads)
_SMALL_PRIMES = [101, 103, 107, 109, 113, 127, 131, 137,
                 139, 149, 151, 157, 163, 167, 173, 179]


def build_config(a):
    """Clone the official config and shrink dimensions in place."""
    cfg = _load_json(CONFIG_FIXTURE)
    t = cfg["text_config"]
    v = cfg["vision_config"]

    # --- text geometry -------------------------------------------------------
    t["hidden_size"] = a.hidden
    t["num_hidden_layers"] = a.layers
    t["vocab_size"] = a.vocab
    t["num_attention_heads"] = a.heads
    t["swa_num_attention_heads"] = a.swa_heads
    t["num_key_value_heads"] = a.kv_heads
    t["qk_nope_head_dim"] = a.qk_nope
    t["qk_rope_head_dim"] = a.qk_rope
    t["v_head_dim"] = a.v_dim
    t["swa_qk_nope_head_dim"] = a.qk_nope
    t["swa_qk_rope_head_dim"] = a.qk_rope
    t["swa_v_head_dim"] = a.v_dim
    t["head_dim"] = a.qk_nope + a.qk_rope
    t["q_lora_rank"] = a.q_lora
    t["kv_lora_rank"] = a.kv_lora
    t["intermediate_size"] = a.intermediate
    t["moe_intermediate_size"] = a.moe_intermediate
    t["routed_expert_hidden_size"] = a.latent
    t["n_routed_experts"] = a.experts
    t["n_shared_experts"] = a.shared
    t["num_experts_per_tok"] = a.topk
    t["first_k_dense_replace"] = 1
    t["max_position_embeddings"] = 1024

    # --- hybrid attention layout: sparse, sparse(+engram), swa, sparse -------
    n = a.layers
    kinds = ["deepseek_sparse_attention"] * n
    if n >= 3:
        kinds[2] = "sliding_attention"
    mlps = ["dense"] + ["sparse"] * (n - 1)
    t["layer_types"] = kinds
    t["mlp_layer_types"] = mlps
    t["swa_pattern"] = [0 if k == "deepseek_sparse_attention" else 1 for k in kinds]
    t["indexer_types"] = ["full"] * n
    t["sliding_window"] = a.sliding_window
    t["window_size"] = [a.sliding_window - 1, 0]
    for rope in t["rope_parameters"].values():
        rope["mrope_section"] = [max(1, a.qk_rope // 2), max(1, a.qk_rope // 4),
                                 max(1, a.qk_rope - a.qk_rope // 2 - a.qk_rope // 4)]

    # --- compressed indexer --------------------------------------------------
    t["compressed_indexer_num_attention_heads"] = a.index_heads
    t["compressed_indexer_head_dim"] = a.index_dim
    t["compressed_indexer_compress_ratio"] = a.index_ratio
    t["compressed_indexer_group_topk"] = a.index_group_topk
    t["compressed_indexer_candidate_topk"] = a.index_candidate_topk
    t["compressed_indexer_stage2_query_block_size"] = 128
    t["index_n_heads"] = a.index_heads
    t["index_head_dim"] = a.index_dim
    t["index_topk"] = a.index_candidate_topk

    # --- engram: ngram/heads/embed are HARD constraints in config.zig --------
    ids = list(a.engram_layers)
    if not ids or len(ids) > 8 or any(i <= 0 or i > n for i in ids):
        raise SystemExit("--engram-layer-ids must be 1-based, within 1..layers, at most 8")
    t["engram_layer_ids"] = ids
    t["engram_layer_head_vocab_sizes"] = {
        str(i): [_SMALL_PRIMES[:8], _SMALL_PRIMES[8:16]] for i in ids
    }
    # small vocab: all special ids moved below vocab_size (see report)
    t["engram_vocab_size"] = [211, 223]
    t["engram_pad_id"] = 3

    # --- small token ids (< vocab_size) -------------------------------------
    id_map = dict(bos_token_id=1, eos_token_id=2, pad_token_id=3,
                  image_start_token_id=4, image_token_id=5, image_newline_token_id=6,
                  image_end_token_id=7, video_start_token_id=8, video_end_token_id=9,
                  video_token_id=10, vision_start_token_id=4)
    for k, val in id_map.items():
        if k in t:
            t[k] = val
        if k in cfg:
            cfg[k] = val
    cfg["dfly_mask_token_id"] = 11

    # --- drafter -------------------------------------------------------------
    cfg["speculative_config"]["num_speculative_tokens"] = a.spec_tokens
    cfg["dfly_num_hidden_layers"] = a.dfly_layers
    cfg["dfly_target_index_layer"] = [n]          # within 1..layers
    cfg["dfly_target_kv_layers"] = [n]
    cfg["dfly_target_index_topk"] = a.index_candidate_topk

    # --- vision tower --------------------------------------------------------
    v["num_hidden_layers"] = a.vision_layers
    v["hidden_size"] = a.vision_hidden
    v["num_attention_heads"] = 2
    v["num_key_value_heads"] = 2
    v["intermediate_size"] = a.vision_hidden * 2
    v["out_hidden_size"] = a.hidden
    v["text_hidden_size"] = a.hidden
    v["max_vit_seq_len"] = 256
    v["img_max_token_num"] = 64
    v["packed_seq_alignment"] = 8
    return cfg


# ---------------------------------------------------------------------------
def official_layer_classes(cfg):
    """(kind, moe, engram) for each official layer, 0-based."""
    t = cfg["text_config"]
    engram_ids = {int(i) - 1 for i in t["engram_layer_ids"]}
    out = []
    for i, kind in enumerate(t["layer_types"]):
        out.append((kind, t["mlp_layer_types"][i] == "sparse", i in engram_ids))
    return out


def derive_names(cfg, official_index):
    """Build the synthetic tensor-name list from official index templates."""
    names = list(official_index["weight_map"].keys())

    # global text tensors (lm_head / embed_tokens / hc_head / final norm)
    glob = sorted(n for n in names
                  if (n.startswith("model.") and not n.startswith("model.layers.")
                      and not n.startswith("model.dfly_layers."))
                  or n.startswith("lm_head."))
    # non-layer vision tensors
    vit_extra = sorted(n for n in names if n.startswith("vit.") and not n.startswith("vit.layers."))
    # non-layer drafter tensors
    dfly_extra = sorted(n for n in names
                        if n.startswith("model.dfly_layers.")
                        and not re.match(r"model[.]dfly_layers[.][0-9]+[.]", n))

    layer_map = defaultdict(dict)
    for n in names:
        m = re.match(r"model[.]layers[.]([0-9]+)[.](.*)$", n)
        if m:
            layer_map[int(m.group(1))][m.group(2)] = n
    dfly_map = defaultdict(dict)
    for n in names:
        m = re.match(r"model[.]dfly_layers[.]([0-9]+)[.](.*)$", n)
        if m:
            dfly_map[int(m.group(1))][m.group(2)] = n
    vit_map = defaultdict(dict)
    for n in names:
        m = re.match(r"vit[.]layers[.]([0-9]+)[.](.*)$", n)
        if m:
            vit_map[int(m.group(1))][m.group(2)] = n

    official = official_layer_classes(_load_json(CONFIG_FIXTURE))
    templates = {}
    for cls in set(official):
        for li, ocls in enumerate(official):
            if ocls == cls:
                templates[cls] = layer_map[li]
                break

    t = cfg["text_config"]
    out = list(glob)
    for i in range(t["num_hidden_layers"]):
        cls = (t["layer_types"][i], t["mlp_layer_types"][i] == "sparse",
               (i + 1) in set(int(x) for x in t["engram_layer_ids"]))
        if cls not in templates:
            raise SystemExit("no official template for layer class %s" % (cls,))
        for suffix in templates[cls]:
            out.append("model.layers.%d.%s" % (i, suffix))

    dlayer = next(iter(sorted(dfly_map)))
    for i in range(cfg["dfly_num_hidden_layers"]):
        for suffix in dfly_map[dlayer]:
            out.append("model.dfly_layers.%d.%s" % (i, suffix))
    out.extend(dfly_extra)

    vlayer = next(iter(sorted(vit_map)))
    for i in range(cfg["vision_config"]["num_hidden_layers"]):
        for suffix in vit_map[vlayer]:
            out.append("vit.layers.%d.%s" % (i, suffix))
    out.extend(vit_extra)

    if len(set(out)) != len(out):
        raise SystemExit("derived duplicate tensor names")
    return out


# ---------------------------------------------------------------------------
def dims(cfg):
    t, v = cfg["text_config"], cfg["vision_config"]
    return dict(
        H=t["hidden_size"], V=t["vocab_size"], L=t["num_hidden_layers"],
        NH=t["num_attention_heads"], SWAH=t["swa_num_attention_heads"],
        KVH=t["num_key_value_heads"],
        NOPE=t["qk_nope_head_dim"], ROPE=t["qk_rope_head_dim"], VDIM=t["v_head_dim"],
        SNOPE=t["swa_qk_nope_head_dim"], SROPE=t["swa_qk_rope_head_dim"],
        SVDIM=t["swa_v_head_dim"],
        QL=t["q_lora_rank"], KVL=t["kv_lora_rank"],
        I=t["intermediate_size"], MI=t["moe_intermediate_size"],
        LAT=t["routed_expert_hidden_size"], E=t["n_routed_experts"],
        TOPK=t["num_experts_per_tok"], HC=t["hc_mult"],
        ENG_HEADS=t["engram_n_head_per_ngram"], ENG_EMBED=t["engram_n_embed_per_ngram"],
        IDXH=t["compressed_indexer_num_attention_heads"],
        IDXD=t["compressed_indexer_head_dim"], IDXR=t["compressed_indexer_compress_ratio"],
        SPEC=cfg["speculative_config"]["num_speculative_tokens"],
        VH=v["hidden_size"], VI=v["intermediate_size"], VOUT=v["out_hidden_size"],
        VHEADS=v["num_attention_heads"], PATCH=v["patch_size"],
        CHANNELS=v["num_channels"], MVLS=v["max_vit_seq_len"],
    )


_HC_SHAPES = {
    "hc_attn_layer.hc_pre.hc_fn": lambda d: (2 * d["HC"], d["HC"] * d["H"]),
    "hc_attn_layer.hc_pre.hc_base": lambda d: (2 * d["HC"],),
    "hc_attn_layer.hc_pre.hc_scale": lambda d: (2,),
    "hc_mlp_layer.hc_pre.hc_fn": lambda d: (2 * d["HC"], d["HC"] * d["H"]),
    "hc_mlp_layer.hc_pre.hc_base": lambda d: (2 * d["HC"],),
    "hc_mlp_layer.hc_pre.hc_scale": lambda d: (2,),
}

_INDEXER_SHAPES = {
    "head_weight_projection.weight": lambda d: (d["IDXH"], d["H"]),
    "intra_group_ape": lambda d: (d["IDXR"], d["IDXD"]),
    "key_gate_projection.weight": lambda d: (d["IDXD"], d["IDXD"]),
    "key_norm.weight": lambda d: (d["IDXD"],),
    "key_value_projection.weight": lambda d: (d["IDXD"], d["IDXD"]),
    "query_projection.weight": lambda d: (d["IDXH"] * d["IDXD"], d["QL"]),
    "token_key_norm.weight": lambda d: (d["IDXD"],),
    "token_key_projection.weight": lambda d: (d["IDXD"], d["H"]),
}

_DFLY_SHAPES = {
    "input_layernorm.weight": lambda d: (d["H"],),
    "post_attention_layernorm.weight": lambda d: (d["H"],),
    "mlp.gate_proj.weight": lambda d: (3 * d["H"], d["H"]),
    "mlp.up_proj.weight": lambda d: (3 * d["H"], d["H"]),
    "mlp.down_proj.weight": lambda d: (d["H"], 3 * d["H"]),
    "self_attn.kv_a_layernorm.weight": lambda d: (d["KVL"],),
    "self_attn.kv_a_proj_with_mqa.weight": lambda d: (d["KVL"] + d["ROPE"], d["H"]),
    "self_attn.kv_b_proj.weight": lambda d: (d["SWAH"] * (d["SNOPE"] + d["SVDIM"]), d["KVL"]),
    "self_attn.learnable_sink_param": lambda d: (d["SWAH"],),
    "self_attn.linear_gate.weight": lambda d: (d["SWAH"] * d["SVDIM"], d["H"]),
    "self_attn.o_proj.weight": lambda d: (d["H"], d["SWAH"] * d["SVDIM"]),
    "self_attn.post_proj_norm.weight": lambda d: (d["H"],),
    "self_attn.q_a_layernorm.weight": lambda d: (d["QL"],),
    "self_attn.q_a_proj.weight": lambda d: (d["QL"], d["H"]),
    "self_attn.q_b_proj.weight": lambda d: (d["SWAH"] * (d["SNOPE"] + d["SROPE"]), d["QL"]),
}

_DFLY_EXTRA = {
    "confidence_head.proj.bias": lambda d: (1,),
    "confidence_head.proj.weight": lambda d: (1, d["H"]),
    "final_norm.weight": lambda d: (d["H"],),
    "hidden_correction.down_proj.weight": lambda d: (d["H"], d["H"]),
    "hidden_correction.embed_norm.weight": lambda d: (d["H"],),
    "hidden_correction.gate_proj.weight": lambda d: (d["H"], 2 * d["H"]),
    "hidden_correction.hidden_norm.weight": lambda d: (d["H"],),
    "hidden_correction.up_proj.weight": lambda d: (d["H"], 2 * d["H"]),
    "mtp_fusion.eh_proj.weight": lambda d: (d["H"], 2 * d["H"]),
    "mtp_fusion.enorm.weight": lambda d: (d["H"],),
    "mtp_fusion.gate": lambda d: (d["H"],),
    "mtp_fusion.hnorm.weight": lambda d: (d["H"],),
}


def engram_bucket_total(cfg, layer):
    t = cfg["text_config"]
    for eid in t["engram_layer_ids"]:
        if int(eid) - 1 == layer:
            table = t["engram_layer_head_vocab_sizes"][str(eid)]
            return sum(int(p) for grp in table for p in grp)
    return 0


def shape_for(name, cfg, d):
    """Map a checkpoint tensor name to the shape implied by the scaled config."""
    H, V = d["H"], d["V"]
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (V, H)
    if name == "model.norm.weight":
        return (H,)
    if name.startswith("model.hc_head.hc_head_fn"):
        return (d["HC"], d["HC"] * H)
    if name.startswith("model.hc_head.hc_head_base"):
        return (d["HC"],)
    if name.startswith("model.hc_head.hc_head_scale"):
        return (1,)

    m = re.match(r"model[.]layers[.]([0-9]+)[.](.*)$", name)
    if m:
        li, s = int(m.group(1)), m.group(2)
        if s in _HC_SHAPES:
            return _HC_SHAPES[s](d)
        if s in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            return (H,)
        if s.startswith("self_attn.compressed_indexer."):
            tail = s[len("self_attn.compressed_indexer."):]
            return _INDEXER_SHAPES[tail](d)
        if s == "self_attn.q_a_proj.weight":
            return (d["QL"], H)
        if s == "self_attn.q_a_layernorm.weight":
            return (d["QL"],)
        if s == "self_attn.q_b_proj.weight":
            return (d["NH"] * (d["NOPE"] + d["ROPE"]), d["QL"])
        if s == "self_attn.kv_a_proj_with_mqa.weight":
            return (d["KVL"] + d["ROPE"], H)
        if s == "self_attn.kv_a_layernorm.weight":
            return (d["KVL"],)
        if s == "self_attn.kv_b_proj.weight":
            return (d["NH"] * (d["NOPE"] + d["VDIM"]), d["KVL"])
        if s == "self_attn.o_proj.weight":
            return (H, d["NH"] * d["VDIM"])
        if s == "self_attn.post_proj_norm.weight":
            return (d["H"],)
        if s == "self_attn.learnable_sink_param":
            return (d["NH"],)
        if s == "self_attn.linear_gate.weight":
            return (d["NH"] * d["VDIM"], d["H"])
        if s == "mlp.gate_proj.weight":
            return (d["I"], H)
        if s == "mlp.up_proj.weight":
            return (d["I"], H)
        if s == "mlp.down_proj.weight":
            return (H, d["I"])
        if s == "mlp.post_fc2_norm.weight":
            return (d["H"],)
        if s == "mlp.gate.weight":
            return (d["E"], H)
        if s == "mlp.gate.e_score_correction_bias":
            return (d["E"],)
        if s == "mlp.experts.gate_up_proj":
            return (d["E"], H, d["LAT"])
        if s == "mlp.experts.down_proj":
            return (d["E"], d["LAT"], d["LAT"])
        if s == "mlp.routed_expert_up_proj.weight":
            return (H, d["LAT"])
        if s == "mlp.routed_expert_down_proj.weight":
            return (d["LAT"], H)
        if s == "mlp.shared_experts.gate_proj.weight":
            return (d["LAT"], H)
        if s == "mlp.shared_experts.up_proj.weight":
            return (d["LAT"], H)
        if s == "mlp.shared_experts.down_proj.weight":
            return (H, d["LAT"])
        if s == "mlp.combine_output_norm.weight":
            return (H,)
        if s.startswith("engram_layer."):
            head = d["ENG_EMBED"] // d["ENG_HEADS"]
            if s == "engram_layer.lookup_table":
                return (d["V"],)
            if s == "engram_layer.multi_head_embedding.embedding.weight":
                return (engram_bucket_total(cfg, li), d["ENG_EMBED"] // d["ENG_HEADS"])
            if s == "engram_layer.value_proj.weight":
                return (H, 2 * d["ENG_EMBED"])
            if s == "engram_layer.value_proj.bias":
                return (H,)
            if s == "engram_layer.unified_pre_gamma":
                return (H,)
            if s.startswith("engram_layer.unified_per_hc_scales."):
                return (H,)
            em = re.match(
                r"engram_layer[.](key_projs|norm_key|norm_query|lookup_per_head_norms|unified_per_hc_scales)[.]([0-9]+)[.](weight|bias)$", s)
            if em:
                kind, which = em.group(1), em.group(3)
                if kind == "key_projs":
                    return (H, 2 * d["ENG_EMBED"]) if which == "weight" else (H,)
                if kind == "unified_per_hc_scales":
                    return (H,)
                if kind == "lookup_per_head_norms":
                    return (head,)
                return (H,)
        raise ValueError("no shape rule for tensor: " + name)

    m = re.match(r"model[.]dfly_layers[.]([0-9]+)[.](.*)$", name)
    if m:
        s = m.group(2)
        if s in _DFLY_SHAPES:
            return _DFLY_SHAPES[s](d)
        raise ValueError("no shape rule for tensor: " + name)

    m = re.match(r"model[.]dfly_layers[.](.*)$", name)
    if m and m.group(1) in _DFLY_EXTRA:
        return _DFLY_EXTRA[m.group(1)](d)

    if name.startswith("vit.layers."):
        parts = name.split(".", 3)
        s = parts[3] if len(parts) > 3 else ""
        vh, vi = d["VH"], d["VI"]
        if s in ("input_layernorm.weight", "input_layernorm.bias",
                 "post_attention_layernorm.weight", "post_attention_layernorm.bias"):
            return (vh,)
        if re.match(r"self_attn[.][qkvo]_proj[.]weight$", s):
            return (vh, vh)
        if re.match(r"self_attn[.][qkvo]_proj[.]bias$", s):
            return (vh,)
        if s == "mlp.dense_h_to_4h.weight":
            return (vi, vh)
        if s == "mlp.dense_h_to_4h.bias":
            return (vi,)
        if s == "mlp.dense_4h_to_h.weight":
            return (vh, vi)
        if s == "mlp.dense_4h_to_h.bias":
            return (vh,)
        raise ValueError("no shape rule for tensor: " + name)

    if name == "vit.embeddings.patch_embedding.weight":
        return (d["VH"], d["CHANNELS"], d["PATCH"], d["PATCH"])
    if name == "vit.embeddings.patch_embedding.bias":
        return (d["VH"],)
    if name == "vit.embeddings.position_embedding.weight":
        return (d["MVLS"], d["VH"])
    if name == "vit.perceive.image_newline":
        return (d["VH"],)
    if name == "vit.perceive.norm.weight":
        return (d["VH"],)
    if name == "vit.perceive.proj.0.weight":
        return (d["VOUT"], d["VH"])
    if name == "vit.perceive.proj.0.bias":
        return (d["VOUT"],)
    if name == "vit.perceive.proj.2.weight":
        return (d["VOUT"], d["VOUT"])
    if name == "vit.perceive.proj.2.bias":
        return (d["VOUT"],)
    raise ValueError("no shape rule for tensor: " + name)


def value_for(name, shape):
    """Zeros for weights, ones for norms, small constants for scales/sinks."""
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return varied(shape)
    if name.endswith(".weight") and ("norm" in name or "layernorm" in name):
        return const(shape, 1.0)
    if name.endswith("hc_head_scale") or name.endswith("hc_scale"):
        return const(shape, 1.0)
    if name.endswith("learnable_sink_param"):
        return const(shape, 0.01)
    if "unified_per_hc_scales" in name:
        return const(shape, 0.02)
    if "intra_group_ape" in name:
        return const(shape, 0.02)
    return zeros(shape)


# ---------------------------------------------------------------------------
def write_shard(path, tensors, metadata=None):
    header = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs = []
    offset = 0
    for name, (shape, arr, dtype) in tensors.items():
        raw = arr.tobytes(order="C")
        header[name] = {
            "dtype": dtype,
            "shape": [int(x) for x in shape],
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    head += b" " * ((-len(head)) % 8)      # safetensors pads header to 8 bytes
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(head)))
        fh.write(head)
        for blob in blobs:
            fh.write(blob)


def shard_name(i, total):
    return "model-%05d-of-%05d.safetensors" % (i + 1, total)


# ---------------------------------------------------------------------------
