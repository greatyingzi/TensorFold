# Hy4 Flash serving integration plan (research receipts)

## Registration surface
- `core/registry.zig` is the quantized-kernel registry, NOT model families. Do not touch it unless adding kernels.
- Family contract: `core/engine_api.zig`: Family (:263-269), Open (:275-353), Opened (:370-377), Engine.VTable (:413-445).
- Lane backend: `core/lanes/backend.zig`: Backend (:75-87), VTable (:88-158). Mandatory: prefill, first, queue, read, verify, keep, draft, release.
- CUDA registration: `native/cuda_families.zig`: registry tuple (:17) + api.Family list (:18-28); dispatch in `native/cuda.zig`:159-181; `openWith` (:182+) requires model_type, formats, default_context, max_segments, Options, Loaded, open. Build modules in `zig/build/cuda_engines.zig`:80-92.
- Server is family-agnostic: `server/main.zig`:154 reads model_type, :182 starts tokenizer+engine; `server/engines.zig`:32-39 publishes capabilities, :53-69 opens engine. HTTP routes in `server/serve.zig`/`chat.zig`.

## Smallest templates
- Metal chat family: Flint — `zig/src/families/flint/` (12 files) + `native/flint.zig` + registration in `native/metal.zig`:70,171 + export in `zig/src/tensorfold.zig`.
- CUDA chat family: Dory — `zig/src/families/dory/` (18 files, cuda_native.zig:14-43,109+) + `build/dory.zig`:5 + `cuda_engines.zig`:91 + `cuda_families.zig`:17.

## Tokenizer
- Checkpoint already has tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json — serving can use the standard HfText path (`server/hf_text.zig`:68-75, template search :167-198). No builtIn hook needed.

## Build / launch
- CUDA: `zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc` → `zig-out/native/bin/tensorfold-native`.
- Serve: `tensorfold-native serve MODEL --name local-model --parallel N --context N --port P`.

## Hy4 gaps (as of this round)
- No api.Family entry; no backend.zig implementing lanes.backend.Backend.VTable; no native/hy4*.zig host open().
- Not exported in `zig/src/tensorfold.zig`, not in any platform family registry/build module table.
- session.zig is token-by-token CPU forward; no prefill/chunk/verify/keep/lane engine; sparse indexer recall not wired; vision tower not forward-wired.
- capability.zig admits only bf16; EXL3/affine4 readers not exposed through formats/registration.
- Minimal serving path: real EXL3 GPU forward (in progress) → backend VTable + native open → CUDA family registration + build export → (tokenizer already in pack) → build tensorfold-native → serve → quality/throughput validation.

## Progress 2026-11-19 (real 3bpw GPU)
- `hy4-gpu-bench exl3real PACK model.layers.1.self_attn.q_a_proj`: real trellis/suh/svh uploaded via exl3_upload, `exl3.Dense.decode` runs at 0.0114 ms/row; 1024/1024 finite, worst rel 0.162 vs the CPU dequantizer. The kept BF16 `.weight` in the pack is a different representation (182x off) and is NOT a valid reference.
- `hy4-gpu-bench exl3layer /pack 1`: nine dense projections of a real layer, 0.136 ms/token = 7379 tok/s dense-only. `kv_a` has n=576 (not a multiple of 128) and is refused by the kernel — needs a split/padding path.
- Codebook for the real pack is `mul1` per config; marker tensor names vary (`mcg` on q_a), so config is authoritative.

## Next implementation steps
1. Full 42-layer dense EXL3 load/decode (batch the per-projection loader; handle kv_a). 2. Wire routed experts from `model.layers.L.mlp.experts.{gate_up,down}.E.*` into `exl3_upload.layer` + shared `exl3_experts`. 3. Assemble MLA + HC + norms + lm_head + sampling into a decode loop. 4. Backend VTable + native open + CUDA family registration. 5. Build `tensorfold-native`, `serve` the pack, run quality + throughput gates.
