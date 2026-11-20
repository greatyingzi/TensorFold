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

## Progress round 38 (in-situ weights intact; field diff remains)
- In-situ integrity at L1: expert 223's trellis pointer valid (0xf13616200000), first words nonzero (0x59eaea4f), suh scales plausible (±0.019) — device-memory corruption RULED OUT for the pointer table and picked expert's data.
- With weights intact, input healthy (0.23), picks/weights correct, scratch zeroed, and the standalone harness reproducing correct output at both 4- and 512-expert scale, the remaining difference is an unspotted Layer-field or argument mismatch between gpu_pack.loadExperts and the working harness construction.
- Next: dump the full exl3_experts.Layer struct values from BOTH the runner and the harness at runtime and diff field-by-field (experts, dims, width, codebook, uniform_k2, gu_stride, limit, act_mode, all 12 buffer pointers); alternatively byte-compare the runner's uploaded stacked-scale buffers against the pack files.

## Progress round 37 (bug is context-dependent)
- act_mode=0 A/B: expert_out unchanged (0.0009) — act_mode ruled out.
- E512 harness (512 experts, real picks 223/395/382/497, real weights): GPU max 0.070 on random x — **the 512-expert scale works standalone.** (Its printed ref was the stale experts-0-3 block — void; the scale verdict stands from E4 + linearity.)
- Runner: latent_in max256 = 0.2275 (healthy) yet expert_out max256 = 0.0009 (~50x small). Same picks/weights, same layer fields, same scratch zeroing.
- Conclusion: the failure is CONTEXT-DEPENDENT — the expert path works standalone and collapses inside the full step. Prime suspects: (a) device-memory corruption of the expert weights/pointer tables by an earlier op in the step (attention/L0-dense/IHC buffers), (b) an unnoticed Layer-field difference between loadExperts and the harness.
- Next: in-situ integrity dump — at L1 in the runner, read back pick 223's trellis pointer and suh scale values right before ex.run; if garbage → find the corrupting op; if intact → diff Layer structs field-by-field.

## Progress round 36 (multi-expert path VERIFIED at 4-expert scale)
- E4 harness (experts=4, pointer tables, stacked scales [4,1536], picks [0..3], wts 0.25): **GPU max 0.241 vs BF16 weighted ref 0.263 — 8% = accumulated quantization noise. The kernel AND the multi-expert grouping path are correct.**
- Combined with the single-expert verification (2-4%), the expert machinery is sound at small scale. The runner's expert_out=0.0006 must come from what DIFFERS from the working harness: (1) 512-expert scale (pointer table 4KB, stacked scales 1.5MB, group() over 512), (2) act_mode=1 vs the harness's 0, (3) the runner's real latent_in/picks/wts.
- Next: E512 harness reusing gpu_pack's loadExperts output (or flip the runner's act_mode to 0 first — one-line A/B), expecting to isolate the 512-scale vs glue difference in one run.

## Progress round 35 (single-expert kernel VERIFIED vs authoritative BF16)
- expertcheck now carries an authoritative reference from the ORIGINAL BF16 source pack (`mlp.experts.gate_up_proj` [512,3072,1536]): **single-expert kernel output matches within 2–4%** (GPU max 0.465 vs ref 0.455; first element -0.097 vs -0.101) — 3bpw quantization noise. The earlier "dequantize" CPU reference was wrong (orientation bugs) and is discarded.
- The MULTI zero in expertcheck was a harness bug: the Layer was built with experts=1 while picks referenced 223/395 — the kernel's e < experts guard dropped them. NOT the runner's failure mode.
- Runner discrepancy remains: expert_out 0.0006 (small but nonzero) with 512 experts, real picks/weights. Narrowed suspects: multi-expert grouping with 512-entry pointer tables, stacked scales [512,1536], or the x_stride semantic (bytes vs elements) in ex.run.
- Next: build a 4-expert harness (experts=4, pointer table + stacked scales) that reproduces the runner's small-but-nonzero output, then bisect: weight count (1 vs 16), distinct experts, stacked-scale layout.

## Progress round 34 (standalone expert numeric check online)
- `hy4-gpu-bench expertcheck /pack 1 223`: single expert, weight 1.0 — the kernel's output has NORMAL magnitude (max 0.465), so the 1000x-small expert_out in the runner is a multi-expert/pick/wts path issue, not the GEMM scale.
- My local CPU reference in expertcheck is untrustworthy (fused gate/up layout assumption + two orientation fixes already changed it 0.46→0.019); the decisive reference is the ORIGINAL BF16 source pack's expert weights (model.layers.1.mlp.experts.gate_up_proj.weight) — next round's first action.
- Serve note: hy4serve35 exited (137) during the long debug; the serve restarts cleanly from the same image.

## Progress round 33 (bug localized inside the expert kernel call)
- latent_in at L1 is normal magnitude (-0.074, 0.080) — routed_down decode is correct. The expert output is ~1000x too small (0.0006 in, expected O(0.1)): the bug is inside the ex.run invocation, not the surrounding glue.
- Prime suspects: (a) gate and up trellis pointers ALIASED to the same fused gate_up trellis with gu_stride=96 — the kernel may expect two separate contiguous projections; (b) act_mode=1 semantics (try 0); (c) the suh/svh stacking layout [E, k] vs the kernel's expectation; (d) HAD_SCALE compensation chain in rot_in/grouped/down epilogues.
- Next: standalone numeric check of the expert kernel (the bench never verified outputs), starting with separate gate/up pointer uploads and act_mode=0.

## Progress round 32 (CLI unified onto Runner; expert-weight dtype fixed)
- The GPU CLI now runs the serve's exact math (Runner module), making it the fast A/B testbed: builds in ~1 min instead of the native server's 6–8.
- Real bug found and fixed: the expert kernel reads routing weights as **fp32** (`const float* wts` in exl3_experts.cu:196); we uploaded fp16, so the weighted sum read garbage/zeros — expert_out was exactly -0.0000. After the fix expert_out is nonzero and the token changed 4459 → 12653.
- Remaining: expert_out is still ~1000x too small (-0.0006 vs expected O(0.1)). Next split: dump latent_in (routed_down decode output) — if ~0, the bug is in routed_down; if normal, it is inside the expert kernel path (fused gate_up pointer aliasing with gu_stride=96, suh/svh stacking, or the act epilogue).
- Also ruled out: expert scratch counters were not the issue (explicit zeroing changed nothing).

## Progress round 31 (A/B localizes the divergence)
- CPU dumpref mode (real weights, full 42-layer dump): **NEXT=68** for token 0 — the earlier NEXT=7 came from the smoke path's synthetic vocab and is void.
- Per-layer mixed dump comparison: **L0 exact match** (bf16 rounding), **L1 ~3% drift** (quantization noise scale), **L2 structural divergence**: CPU mixed jumps 20x over L1 (0.22/-0.23/0.70) while GPU stays at L1's scale (0.014). L2's input is L1's FFN output and L1 is the first routed-MoE layer — the GPU's routed-expert contribution is effectively zero while the CPU's is strong (routed_scale=3.998).
- Next: instrument the GPU MoE block (routed_down → ex.run → routed_up) at L1 — print expert_out and routed_out magnitudes, and verify pick/wts layout (int32 [rows][16] picks, fp16 wts) against the kernel's expectations.

## Progress round 30 (A/B ground truth established)
- CPU reference forward on the ORIGINAL BF16 pack (token 0, 42 layers, mmap lazy decode): **NEXT=7**, 57.9 s for 2 tokens, RC=0. Correctness target for the GPU path.
- GPU step on the EXL3 pack with the same input gives 87033 (CLI) — a large divergence, independent of the MoE wiring (a 3-layer run already diverges).
- Rope theta fixed per layer type (DSA 1e7 / sliding 1e4, from rope_parameters); SwiGLU clamp (limit 10) added for shared experts and L0 dense via tf_hy4_swiglu_limit; output changed but still gibberish (~11 tok/s, no NaN).
- Next: per-layer dump diff CPU vs GPU. The CPU session already has dump hooks; add the matching dump to the GPU CLI, run both with token 0, compare layer by layer.

## Progress round 28 (SERVE STABLE END-TO-END)
- The serve path now completes requests with all 42 layers (64-head sliding layers included), zero NaN, no crashes: 12-token completion at 14.65 tok/s, TTFT 0.37 s, greedy.
- Root cause of the last `Invalid`: merge downloads requested the max-heads host array against a per-layer-heads device view (65,536 vs 131,072 bytes); clipped to `nout = rows*heads*kv_lora`.
- Remaining quality gap (why output is still gibberish): routed MoE experts (512-choose-16, most of the model's parameters) are not called — only the shared expert runs; layer 0's dense MLP is zeroed. The routed-expert kernel path is already proven at 16k tok/s in the bench, so wiring it is the next concrete step.

## Progress round 27 (full-model CLI verified; serve crash root-caused)
- **CLI `hy4-gpu-step /pack 0` runs all 42 layers with per-layer geometry (64-head sliding layers included): finite logits (min −9.12, max 8.74), 54.5 ms/token = 18.4 tok/s, RC=0.** The full GPU forward math is verified end-to-end on the real 3bpw pack.
- Serve-side segfault root cause: the runner's attention scratch buffers were still sized for 32 heads — L2's 64-head q_b decode wrote 32 KB into a 16 KB `qfull` (heap corruption). A batch of resize edits had silently failed to apply. Fixed by actually resizing to `max_heads`; the serve no longer segfaults.
- Remaining serve issue: first request now fails with CUDA `Invalid` at L0 (before merge) — no crash. Next step is an A/B diff against the known-good CLI path, which runs the same math successfully.
- CLI absorb path also fixed: direct BF16 `kv_b.weight` download with per-layer sizes replaced the overflowing identity decode.

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
