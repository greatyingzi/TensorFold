#!/usr/bin/env python3
"""Generate and verify a tiny, structurally identical Hy4-Flash checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hy4_synthetic_shapes import *
from hy4_synthetic_shapes import _load_json

def generate(a):
    if not CONFIG_FIXTURE.exists() or not INDEX_FIXTURE.exists():
        raise SystemExit("missing fixtures under %s" % FAM)
    cfg = build_config(a)
    d = dims(cfg)
    if a.heads % a.kv_heads or a.swa_heads % a.kv_heads:
        raise SystemExit("heads and swa_heads must be divisible by kv_heads")
    if a.dfly_layers > 8 or len(a.engram_layers) > 8:
        raise SystemExit("dfly layers / engram layers exceed 8")
    if a.latent > a.moe_intermediate or a.topk > a.experts:
        raise SystemExit("latent<=moe_intermediate and topk<=experts required")
    if a.index_group_topk > a.index_candidate_topk or a.index_ratio > a.sliding_window:
        raise SystemExit("index group_topk<=candidate_topk and ratio<=window required")

    names = derive_names(cfg, _load_json(INDEX_FIXTURE))
    tensors = {}
    total_bytes = 0
    for name in names:
        shape = shape_for(name, cfg, d)
        arr = value_for(name, shape)
        tensors[name] = (shape, arr)
        total_bytes += arr.nbytes

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write(chr(10))

    # deterministic round-robin split across shards
    shards = [{} for _ in range(a.shards)]
    wmap = {}
    for i, name in enumerate(names):
        si = i % a.shards
        shards[si][name] = tensors[name]
        wmap[name] = shard_name(si, a.shards)
    emeta = {"source_tokenizer_len": cfg["text_config"]["vocab_size"],
             "compressed_vocab_size": cfg["text_config"]["engram_vocab_size"][0],
             "engram_seed": cfg["text_config"].get("engram_seed", 0),
             "normalizer_signature": "nfkc-nfd-stripaccents-lower-ws-strip-addedbypass-v2",
             "lookup_sha256": "00" * 32}
    meta = {"format": "pt", "engram_vocab_map_metadata": json.dumps(emeta, separators=(",", ":"))}
    for si, bucket in enumerate(shards):
        if bucket:
            write_shard(out / shard_name(si, a.shards), bucket, meta)

    with open(out / "model.safetensors.index.json", "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": wmap},
                  fh, indent=2, sort_keys=True)
        fh.write(chr(10))

    print("[hy4-synthetic] %d tensors, %d shards, %d bytes (%.2f MiB) -> %s"
          % (len(names), sum(1 for s in shards if s), total_bytes,
             total_bytes / 2 ** 20, out))
    return 0


# ---------------------------------------------------------------------------
def validate_config(cfg, d):
    """Mirror config.zig parse() + graph.zig validate(); raise on mismatch."""
    t, v = cfg["text_config"], cfg["vision_config"]
    if cfg.get("model_type") != "hy_v4_flash":
        raise ValueError("model_type must be hy_v4_flash")
    for key in ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
                "swa_num_attention_heads", "num_key_value_heads", "qk_nope_head_dim",
                "qk_rope_head_dim", "v_head_dim", "swa_qk_nope_head_dim",
                "swa_qk_rope_head_dim", "swa_v_head_dim", "q_lora_rank", "kv_lora_rank",
                "intermediate_size", "moe_intermediate_size", "routed_expert_hidden_size",
                "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
                "first_k_dense_replace", "max_position_embeddings",
                "compressed_indexer_num_attention_heads", "compressed_indexer_head_dim",
                "compressed_indexer_compress_ratio", "compressed_indexer_group_topk",
                "compressed_indexer_candidate_topk", "hc_mult", "sliding_window",
                "engram_max_ngram_size", "engram_n_head_per_ngram",
                "engram_n_embed_per_ngram", "engram_pad_id"):
        val = t.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError("text_config.%s must be a positive int (got %r)" % (key, val))
    for key in ("enable_ihc", "sandwich_norm", "gated_mla", "learnable_sink",
                "engram_fp8_embedding"):
        if not isinstance(t.get(key), bool):
            raise ValueError("text_config.%s must be bool" % key)
    for key in ("rms_norm_eps", "hc_magnitude", "hc_eps", "routed_scaling_factor"):
        val = t.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError("text_config.%s must be numeric" % key)
    if t["num_hidden_layers"] > 64 or t["n_routed_experts"] > 512:
        raise ValueError("layers<=64 and experts<=512 required")
    if t["hc_mult"] != 4:
        raise ValueError("hc_mult must be 4")
    if not t["enable_ihc"]:
        raise ValueError("enable_ihc must be true")
    if t["num_experts_per_tok"] > t["n_routed_experts"]:
        raise ValueError("num_experts_per_tok > n_routed_experts")
    if t["routed_expert_hidden_size"] > t["moe_intermediate_size"]:
        raise ValueError("routed_expert_hidden_size > moe_intermediate_size")
    if (t["num_attention_heads"] % t["num_key_value_heads"] or
            t["swa_num_attention_heads"] % t["num_key_value_heads"]):
        raise ValueError("heads must be divisible by kv_heads")
    if t["first_k_dense_replace"] <= 0 or t["first_k_dense_replace"] >= t["num_hidden_layers"]:
        raise ValueError("0 < first_k_dense_replace < num_hidden_layers")
    if t["compressed_indexer_compress_ratio"] > t["sliding_window"]:
        raise ValueError("index ratio > sliding window")
    if t["compressed_indexer_group_topk"] > t["compressed_indexer_candidate_topk"]:
        raise ValueError("index group_topk > candidate_topk")
    for key in ("layer_types", "mlp_layer_types", "swa_pattern", "indexer_types"):
        if not isinstance(t.get(key), list) or len(t[key]) < t["num_hidden_layers"]:
            raise ValueError("text_config.%s too short" % key)
    ropes = t.get("rope_parameters")
    if not isinstance(ropes, dict):
        raise ValueError("missing rope_parameters")
    for kind in ("deepseek_sparse_attention", "sliding_attention"):
        r = ropes.get(kind)
        if not isinstance(r, dict) or not isinstance(r.get("rope_theta"), (int, float)):
            raise ValueError("missing rope theta for %s" % kind)
    for i in range(t["num_hidden_layers"]):
        if t["layer_types"][i] not in ("deepseek_sparse_attention",
                                       "indexed_attention", "sliding_attention"):
            raise ValueError("unsupported layer kind at %d" % i)
        if t["mlp_layer_types"][i] not in ("dense", "sparse"):
            raise ValueError("unsupported mlp kind at %d" % i)
        if i >= t["first_k_dense_replace"] and t["mlp_layer_types"][i] != "sparse":
            raise ValueError("layer %d must be MoE after first_k_dense_replace" % i)

    # engram (config.zig hard constraints)
    if (t["engram_max_ngram_size"] != 3 or t["engram_n_head_per_ngram"] != 8 or
            t["engram_n_embed_per_ngram"] != 8 * 96):
        raise ValueError("engram ngram/heads/embed must be 3/8/768")
    if not isinstance(t.get("engram_seed"), int) or isinstance(t["engram_seed"], bool) or t["engram_seed"] < 0:
        raise ValueError("bad engram_seed")
    ev = t.get("engram_vocab_size")
    if not isinstance(ev, list) or len(ev) != 2 or any(not isinstance(x, int) or x <= 0 for x in ev):
        raise ValueError("bad engram_vocab_size")
    ids = t.get("engram_layer_ids")
    if not isinstance(ids, list) or not (0 < len(ids) <= 8):
        raise ValueError("bad engram_layer_ids")
    sizes = t.get("engram_layer_head_vocab_sizes")
    if not isinstance(sizes, dict):
        raise ValueError("bad engram_layer_head_vocab_sizes")
    for eid in ids:
        if not isinstance(eid, int) or eid <= 0 or eid > t["num_hidden_layers"]:
            raise ValueError("engram layer id out of range")
        grps = sizes.get(str(eid))
        if not isinstance(grps, list) or len(grps) != 2:
            raise ValueError("engram %s: expected 2 prime groups" % eid)
        for g in grps:
            if not isinstance(g, list) or len(g) != 8 or any(not isinstance(p, int) or p <= 0 for p in g):
                raise ValueError("engram %s: each group needs 8 positive primes" % eid)

    # drafter
    spec = cfg.get("speculative_config")
    if not isinstance(spec, dict) or not isinstance(spec.get("num_speculative_tokens"), int) or spec["num_speculative_tokens"] <= 0:
        raise ValueError("bad speculative_config")
    dl = cfg.get("dfly_num_hidden_layers")
    if not isinstance(dl, int) or dl <= 0 or dl > 8:
        raise ValueError("bad dfly_num_hidden_layers")
    if not isinstance(cfg.get("dfly_target_index_topk"), int) or cfg["dfly_target_index_topk"] <= 0:
        raise ValueError("bad dfly_target_index_topk")
    dtargets = cfg.get("dfly_target_index_layer")
    if not isinstance(dtargets, list) or not (0 < len(dtargets) <= 8):
        raise ValueError("bad dfly_target_index_layer")
    for x in dtargets:
        if not isinstance(x, int) or x <= 0 or x > t["num_hidden_layers"]:
            raise ValueError("dfly target out of range")

    # vision (config.zig parse)
    for key in ("num_hidden_layers", "hidden_size", "patch_size", "spatial_merge_size", "out_hidden_size"):
        val = v.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError("vision_config.%s must be positive int" % key)


def expected_count(cfg):
    """Mirror graph.layerPlan + the constant per-family tensor groups."""
    t = cfg["text_config"]
    engrams = {int(i) - 1 for i in t["engram_layer_ids"]}
    total = 0
    for i in range(t["num_hidden_layers"]):
        moe = t["mlp_layer_types"][i] == "sparse"
        sparse_attn = t["layer_types"][i] != "sliding_attention"
        base = (36 if sparse_attn else 28) if moe else 30
        total += base + (41 if i in engrams else 0)
    total += 15 * cfg["dfly_num_hidden_layers"] + 12       # dfly layers + dfly extras
    total += 16 * cfg["vision_config"]["num_hidden_layers"] + 9   # vit layers + vit extras
    total += 6                                              # embed/lm_head/hc_head/norm
    return total


def read_shard_header(path):
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError("%s: truncated header length" % path.name)
        n = struct.unpack("<Q", raw_len)[0]
        hdr = json.loads(fh.read(n).decode("utf-8"))
    return hdr


def verify(directory):
    directory = Path(directory)
    cfg_path = directory / "config.json"
    idx_path = directory / "model.safetensors.index.json"
    if not cfg_path.is_file() or not idx_path.is_file():
        raise SystemExit("%s: missing config.json or index" % directory)

    cfg = _load_json(cfg_path, detect_dups=True)
    d = dims(cfg)
    validate_config(cfg, d)
    print("[verify] config.json: parse + graph constraints OK "
          "(layers=%d, hidden=%d, experts=%d)"
          % (cfg["text_config"]["num_hidden_layers"], cfg["text_config"]["hidden_size"],
             cfg["text_config"]["n_routed_experts"]))

    index = _load_json(idx_path, detect_dups=True)
    if ("metadata" not in index or "weight_map" not in index
            or not isinstance(index["weight_map"], dict)):
        raise SystemExit("bad index: missing metadata/weight_map")
    wmap = index["weight_map"]
    names = list(wmap.keys())
    if len(set(names)) != len(names):
        raise SystemExit("duplicate tensor names in index")
    want = expected_count(cfg)
    if len(names) != want:
        raise SystemExit("tensor count %d != layer-graph expectation %d" % (len(names), want))
    print("[verify] tensor names: %d (expected %d), no duplicates" % (len(names), want))

    headers = {}
    for shard in wmap.values():
        if shard not in headers:
            p = directory / shard
            if not p.is_file():
                raise SystemExit("shard referenced by index is missing: %s" % shard)
            headers[shard] = read_shard_header(p)
    print("[verify] shards: %d referenced, all present" % len(headers))

    seen = set()
    total_bytes = 0
    for name, shard in wmap.items():
        hdr = headers[shard]
        if name not in hdr or not isinstance(hdr[name], dict):
            raise SystemExit("%s: not present in %s header" % (name, shard))
        if name in seen:
            raise SystemExit("duplicate tensor across shards: %s" % name)
        seen.add(name)
        meta = hdr[name]
        if meta.get("dtype") != "BF16":
            raise SystemExit("%s: dtype %s is not BF16" % (name, meta.get("dtype")))
        shape = tuple(meta.get("shape", []))
        exp_shape = tuple(shape_for(name, cfg, d))
        if shape != exp_shape:
            raise SystemExit("%s: shape %s != config-derived %s" % (name, shape, exp_shape))
        a_off, b_off = meta.get("data_offsets")
        total_bytes += int(b_off) - int(a_off)
    print("[verify] all %d tensors are BF16 with config-consistent shapes" % len(names))

    declared = index["metadata"].get("total_size")
    if declared != total_bytes:
        raise SystemExit("index total_size %s != shard bytes %d" % (declared, total_bytes))
    print("[verify] total_size OK: %d bytes (%.2f MiB)" % (total_bytes, total_bytes / 2 ** 20))
    print("[verify] PASS")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", metavar="DIR", help="generate synthetic checkpoint here")
    p.add_argument("--verify", metavar="DIR", help="verify a generated checkpoint")
    for key, val in DEFAULTS.items():
        if key == "engram_layers":
            p.add_argument("--engram-layer-ids", dest="engram_layers", default=val,
                           help="comma-separated 1-based engram layer ids")
        else:
            p.add_argument("--" + key.replace("_", "-"), default=val, type=type(val),
                           help="default: %s" % (val,))
    a = p.parse_args(argv)
    if isinstance(a.engram_layers, str):
        a.engram_layers = tuple(int(x) for x in a.engram_layers.split(",") if x)
    if bool(a.out) == bool(a.verify):
        p.error("pass exactly one of --out DIR or --verify DIR")
    if a.verify:
        return verify(Path(a.verify))
    return generate(a)


if __name__ == "__main__":
    sys.exit(main())
