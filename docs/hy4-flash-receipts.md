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

1. **42 层完整前向 OOM**。DGX00 111GiB available 下，加载 25 秒 RSS 已 67GB 并继续增长。专家已走 affine4 按需解码，但非专家 BF16 张量（`o_proj`、`linear_gate`、`q_b/kv_b`、`shared_experts` 等）仍全量转 f32 常驻。修法：非专家也改 mmap + 按需 matvec / 逐层流式，不能靠调参绕过。
2. EXL3 未覆盖 packed 专家与 Engram；需要把这两个自定义 Module 暴露给 exllamav3 量化器。
3. 视觉 tower / 多模态未接。
4. 完整模型（42 层）decode/prefill/concurrency 性能 receipts 未收；当前只有 2 层 CPU 基线。

## 相关工具

- `zig/hy4_runner.zig`：按层加载 + 注入小词表的烟测 runner，支持内部计时。
- `tools/zig/hy4_synthetic.py`：合成检查点生成器（形状规则与官方 header 对齐）。
- `zig/src/families/hy_v4_flash/exl3.zig`、`affine4.zig`、`weights.zig`：两条量化 reader。
- `hy4_assemble.py`、`repair_shard.py`、`verify_assembled.py`（在 `/Users/liheng/work/hy4-quant/`）：Q4 完整包组装/修复/校验。
