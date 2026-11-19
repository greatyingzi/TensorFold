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

## Progress round 26 (KV cache attempt reverted)
- A first KV-cache implementation (per-layer [42][4096][576] BF16 cache, latent absorb, roped tail, chunks/merge/expandV) compiled on host but **segfaulted in Runner.init on DGX** before the GPU decode log line. The working service was restored (container `hy4serve10`, 137.9 s load, `/health` OK, 33 tok/s).
- Root-cause candidates for the segfault: identity-decoding kv_b with 512 rows against scratch reserved for 128 rows; sink tensor is F32 [32] in the pack (not BF16), so any BF16 conversion is wrong. A corrected attempt is in progress.

## Progress round 25 (SERVICE RUNNING)
- `tensorfold-native serve /pack --name hy4flash --parallel 1 --context 2048 --port 8080` **starts successfully on DGX00** with the real 3bpw GPU decode path: `hy_v4_flash: 3bpw GPU decode active, 611360 bytes of scratch and 1 stream`; model 59.29 GiB; loaded in 97.4 s; `/v1/models` and `/v1/completions` respond.
- Measured serving throughput: 33.4 tok/s, TTFT 0.044 s, 8 tokens in 0.209 s (greedy).
- `/v1/chat/completions` was refused by the checkpoint's chat template (`str.format()` unsupported by the server's Jinja subset). Passing `--chat-template /chat/hy4_chat.jinja` with a minimal loop template fixes it; both `/v1/completions` and `/v1/chat/completions` now return 200 with 33.5 tok/s.
- Running service container on DGX00: `hy4serve8` (host network, port 8080), model name `hy4flash`.
- Generated text is still incoherent: the GPU step's attention is zeroed (valid only for the first token) and routed MoE experts are not yet called. Quality gate NOT passed.
- Registration hook that made admission fit: `native.zig textWeightBytes` returns 70 GiB (compressed budget) instead of the generic 464 GiB; GPU path loads only `gpu_pack` (59.29 GiB) and skips the host Bag.

## Progress round 24 (decode step unblocked, serving registered)
- Root cause of the 1-layer GPU step hang: IHC kernels called `__shfl_xor_sync` inside `if (lane == 0)`; only one lane of the warp participated. Fixed in both collapse and hc_head; added a gate-only kernel for correct MoE post gates.
- Full 42-layer GPU step now completes with real 3bpw weights: finite logits (min -10.1, max +9.4), ~24-41 ms/token. Attention is temporarily zeroed (correct only for the first token with an empty cache); KV cache write is still pending.
- lm_head in the pack is fp16 and norm tensors are bf16; both are converted at load (norms to fp32, lm_head to bf16) — without this logits were NaN.
- Hy4 family serving registration skeleton lands in native/cuda_families.zig, build/cuda_engines.zig, tensorfold.zig, with backend.zig/native.zig/contract.zig; host builds and `check-native-cuda-host` are green.
- `tensorfold-native serve` currently refuses the pack: the generic loader accounts 464 GiB against a 101 GiB GPU budget. The GPU path (gpu_pack) only uploads 63 GiB, so the serving backend must be switched to the streaming GPU loader before deployment.

## Progress round 21 (full pack load + IHC kernels)
- CUDA IHC kernels (collapse / stream_update / hc_head) implemented in `zig/kernels/cuda/hy4_ihc.cu`, bound in `zig/src/cuda/hy4_ihc.zig`, fatbin registered; compiles with nvcc.
- `zig/src/families/hy_v4_flash/gpu_pack.zig` loads the whole real pack into device weights: 42 layers, 6.55 GB of EXL3 dense weights + embed/lm_head + all HC/norm/sink/gate tensors. `hy4-gpu-bench packcheck /pack` passes.
- Real 512-expert routed layer runs in 0.062 ms/token. Full dense 42 layers run in 5.544 ms/token.

## Next implementation steps
1. Full 42-layer dense EXL3 load/decode (batch the per-projection loader; handle kv_a). 2. Wire routed experts from `model.layers.L.mlp.experts.{gate_up,down}.E.*` into `exl3_upload.layer` + shared `exl3_experts`. 3. Assemble MLA + HC + norms + lm_head + sampling into a decode loop. 4. Backend VTable + native open + CUDA family registration. 5. Build `tensorfold-native`, `serve` the pack, run quality + throughput gates.
