# Hy4 Flash 适配 receipts（hy4-flash-adapt）

## 已验收

- 分支基于上游 `origin/main` @ `8cbef9777`；`zig build test-hy4-host` 56 测试全绿；`tools/pr_check.py --base origin/main --pr 1` 0 error，lean 221 → 221。
- 真权重形状/dtype 按官方 safetensors header 全量对齐：q_b `nope+rope`、kv_a `kv_lora+rope`、expert down `[E,latent,latent]`、indexer F32、sink/router bias F32、draft MLP/confidence head 等。
- 真 BF16 2 层烟测（DGX00）：72 张量、有限 logits、墙钟约 21.8s；内部计时 4 token 纯前向 1.45s ≈ 2 tok/s（CPU matvec，不含加载）。
- 双机量化框架（改造版 exl3-dist-quant）真双机 DGX00↔DGX01 四项全过：端到端、跨机 bit-identical、`-r` 跨机续跑、tiny-qwen 全量 649MB 包两机各一份。关键约束：`docker import` 会丢 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，必须恢复才能得到 bit-identical。DGX01 磁盘 409G < 源 410G，按主人裁定不删数据，Hy4 双机全量维持阻塞。
- Hy4 exllamav3 adapter M1–M3 通过，真 440GB 权重 `--max_module 3` 通过；限制：packed 专家与 Engram 自定义 Module 未暴露给 EXL3 量化器（MoE 层 bpw 0.01–0.04），专家量化由自研 affine4 PTQ 覆盖。
- affine4 PTQ：82/82 专家张量、246 张量/82 分片、0 异常、78GiB；Engram FP8 真表探针 SQNR 31.77dB、1.92x、边界全绿。
- EXL3 规范 reader：Zig 侧解析 `quantization_config`、变体门控、`trellis/suh/svh` 标量解码器，已接入 loader；合成端到端 parity 通过。
- 完整 Q4 包：`Hy4-flash-Q4-Repaired` 87 分片 / 2128 张量 / 0 坏；坏分片可用 `repair_shard.py` 按索引从源包与 Q4 专家包精确重建。

## 未完成（明确的下一道坎）

1. ~~42 层完整前向 OOM~~ **已解决（提交 `12799651c`）**：非专家 BF16 张量 >100k 元素保留 mmap 字节（`Value.bf16raw`），`ops.matvecBf16` 在 matvec 内流式解码，全部 matvec 调用点经 `matvecW` 分发（含 router 的 `gate.weight`）。DGX00 全 42 层 Q4 前向通过：1272 张量、4 token、有限 logits，86.0s（约 0.046 tok/s，CPU 懒解码基线）。
2. EXL3 未覆盖 packed 专家与 Engram；需要把这两个自定义 Module 暴露给 exllamav3 量化器。
3. 视觉 tower：已解析真实 `vision_config` 并建模全部 441 个 ViT 张量形状（含 rank-4 patch/perceive 投影），提交 `6a907c4f9`；真实视觉前向与多模态 token 拼接仍未接。
4. 完整模型（42 层）性能基线已有 0.046 tok/s（CPU 懒解码）；decode/prefill/concurrency 的 GPU 路径 receipts 仍未收。

## 相关工具

- `zig/hy4_runner.zig`：按层加载 + 注入小词表的烟测 runner，支持内部计时。
- `tools/zig/hy4_synthetic.py`：合成检查点生成器（形状规则与官方 header 对齐）。
- `zig/src/families/hy_v4_flash/exl3.zig`、`affine4.zig`、`weights.zig`：两条量化 reader。
- `hy4_assemble.py`、`repair_shard.py`、`verify_assembled.py`（在 `/Users/liheng/work/hy4-quant/`）：Q4 完整包组装/修复/校验。

## 2026-11-19 night: GPU performance campaign + dual-node quantization

### GPU throughput on DGX00 (hy4-gpu-bench, affine4 kernels, Hy4 geometry)
- experts 41L x 16E x 2P, single launches: 16.91 ms/token = 59.1 tok/s
- experts grouped (gemvMany x4, 8 launches/layer): **11.05 ms/token = 90.5 tok/s**, 262 GB/s = 96% of HBM peak
- attn/dense projections (42L x 6P): 11.15 ms/token = 89.7 tok/s, 230 GB/s = 84%
- MLA attention math (absorb_q<192>, chunks, merge, expand_v<256>, gate; 4K cache): 7.3 ms/token
- total affine4-5bpw estimate: 29.5 ms = 34 tok/s; at 3bpw projected 48-65 tok/s
- all 9 supported dense projections of one real layer (`hy4-gpu-bench exl3layer /pack 1`): 0.136 ms/token = **7379 tok/s** dense-only. kv_a (n=576) is not a multiple of 128 and needs a separate split/padding path; the other nine load and run directly.
- real EXL3 3bpw projection on GPU (`hy4-gpu-bench exl3real PACK model.layers.1.self_attn.q_a_proj`): direct safetensors read of trellis/suh/svh/marker, `exl3_upload.Uploader` + `exl3.Dense.decode`, **0.0114 ms/row = 87.8k tok/s** for one projection; 1024/1024 finite outputs, worst rel 0.162 against the CPU dequantizer (the pack's kept BF16 `.weight` is a different representation, 182x off, so it is not a reference). This is the first real-weight EXL3 kernel receipt.
- full 42-layer step assembly with grouped expert launches (gemvMany, 8 launches/layer): **20.86 ms/token = 47.9 tok/s** (random affine4, all projections + MLA + sink merge). Earlier 32-separate-launches build was 37.45 ms/token = 26.7 tok/s under quant contention; the launch-count cut is the main win. Root cause of an intermediate `InputMismatch`: gemvMany rejects duplicate output pointers; each of the 16 experts now owns a distinct output buffer.
- real-weight numeric witness: layer1 expert0 down_proj GPU vs CPU, worst rel 3.890e-3 = bf16 output precision (0.391%), zero missing rows

### Dual-node quantization (DGX00 head + DGX01 peer over CX7 10.0.1.x)
- layer 7 (first dual): quantize 500.7 -> 246.3 s = **2.03x**, 528 peer / 506 local tensors, xfer 0.66 GiB/layer
- layer 8: 574.95 s total vs 833 s single = 1.45x per layer
- source copy to DGX01: HTTP parallel fetch 1.23 GB/s over CX7 (rsync 191 MB/s); disk-read bound at 660 MB/s
### DFly2 draft correctness (2026-11-19 noon)

- The checkpoint contains more than the 87 per-layer dfly tensors: `final_norm`, `hidden_correction.{embed_norm,hidden_norm,gate_proj,up_proj,down_proj}`, `mtp_fusion.{enorm,hnorm,eh_proj,gate}`, `confidence_head`. The first drafter ignored all of them and wrongly used the target model's `norm.weight`, which made its hidden nearly orthogonal to the target (cos 0.06, top-10 overlap 0/10).
- Wired the full draft route: hidden_correction (SwiGLU on concat(norm(embed), norm(target hidden))) -> 5 gated MLA draft layers -> mtp_fusion (enorm/hnorm + eh_proj + gate) -> dfly final_norm -> shared lm_head.
- Native DGX00 probe with the corrected binary, seed 120000: `sum_chain` agreement 2/5 = 40% (was 0% with the stale binary that lacked these modules), `target_only` 0%. The official-style MTP recipe in the repo's ling oracle confirms enorm/hnorm -> concat -> eh_proj -> draft block -> final_norm. sum_chain is the融合变体 to use in the spec-decode ring.
- The spec-decode loop is wired into `hy4-runner ... spec TOKEN STEPS` via `draft.proposeDfly`; synthetic fixture smoke 2/2 accepted. A double-step bug (target advanced once at loop head and again in verify) was fixed.
- 20-step native probe (seed 120000, 564.8s): `sum_chain` 4/20 = 20%, `target_only` 0/20 = 0%. Baselines: embed-vs-final cosine ~0.01, layer-41 hidden-vs-final ~0.75. The draft vector is above the embed baseline but below the target hidden; acceptance is real but modest. sum_chain is the fusion to use.
- End-to-end `hy4-runner ... spec 120000 10` on the real checkpoint: 2/10 accepted (20%), 182.2s for 10 tokens on CPU. The draft ring is live in the host runner; GPU-side batched verification is the remaining lever for a decode-speed win.

### EXL3 3.0bpw quantization (2026-11-19 12:33)

- `-- All done` + `CONVERT_EXIT:0`; 42 layers, per-layer bpw ~3.02, sqnr ~16.7; 8 shards written into `NewModels/Hy4-flash-EXL3` (~59 GB before graft).
- `post_quant.sh` chain is running: graft BF16 extras (dfly/vit/hc_head and every non-routed tensor) into the pack, then `verify_grafted.py` structural check.

