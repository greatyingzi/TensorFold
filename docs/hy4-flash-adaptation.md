# Hy4 Flash（tencent/Hy4-flash）TensorFold 适配方案

状态：调研完成，代码尚未实现。本文是本分支（`hy4-flash-adapt`）的适配依据。

## 0. 分支基线

- 本地仓库：`~/work/self/test/glm5.3-flash/TensorFold`
- 新分支：`hy4-flash-adapt`，从 `origin/main` @ `8cbef9777` 切出（上游主线，晚于 tag `v1.2.9`）
- 已推送到 `fork`（`git@github.com:greatyingzi/TensorFold.git`）
- 引擎：**Zig 主线**。Python 线冻结在 v0.6.6 / 分支 `python-0.6`，本次不做 Python 适配。
- 回滚：删除远端/本地同名分支即可，不影响 `main` 与任何现有分支。

## 1. 参考的官方与社区适配（事实来源）

| 来源 | 内容 | 链接 |
| --- | --- | --- |
| Transformers PR #50903（open） | 官方架构实现 `HYV4Flash`、权重重命名、配置解析 | https://github.com/huggingface/transformers/pull/50903 |
| llama.cpp PR #32517（open） | `src/models/hy-v4-flash.cpp`、`conversion/hy_v4_flash.py`、索引页缓存 `llama-memory-idx-iswa` | https://github.com/ggml-org/llama.cpp/pull/32517 |
| vLLM 插件 FL PR #773（open） | 8 个私有算子、W8A8 INT8 MoE、Engram 使用 126.9 GB pinned host + UVA | https://github.com/flagos-ai/vllm-plugin-FL/pull/773 |
| 官方模型卡 / config / 权重索引 | 2026-11-18 发布，bf16 总大小 440.5 GB | https://huggingface.co/tencent/Hy4-flash |
| 官方仓库 | 部署、DFly2、量化说明 | https://github.com/Tencent-Hunyuan/Hy4-flash |

**结论：上游 TensorFold 至今没有任何 hy/hunyuan 分支、issue 或模型族，这是全新适配；但 Hy4 Flash 的构件与现有 `flashnext` / `deepseek_v41` 两族高度重合，可大量复用。**

## 2. 模型事实（来自 config.json 与官方模型卡）

- `model_type`: `hy_v4_flash`；`architectures`: `HYV4FlashForConditionalGeneration`
- 总参 220B（backbone 155.5B + Engram 63.5B + ViT 0.43B + DFly2 0.85B），每 token 激活 11.3B
- 42 层（1 dense + 41 MoE），hidden 3072，词表 120832，上下文 1,048,576
- **混合注意力**：12 层 `deepseek_sparse_attention`（32 头，rope θ=1e7）+ 30 层 `sliding_attention`（64 头，rope θ=1e4，window [511,0]，SWA 头维度 192+64/256），按 `layer_types` 交错
- **MLA**：q_lora_rank 1024，kv_lora_rank 512，qk_nope 192 + qk_rope 64，v_dim 256，gated_mla，learnable sink
- **压缩索引器**：每 4 token 池化为一个组键（gated pooling），组 topk 512，候选 topk 2048，16 头 × 128 维；短上下文不裁剪
- **LatentMoE**：512 路由专家 + 1 共享，top16，专家中间维 1536（hidden 3072 → latent 1536），sigmoid 路由 + e_score_correction_bias
- **Engram**：层 1/11/21/31（0-based；config 的 `engram_layer_ids` 是 1-based 的 2/12/22/32）；2/3-gram，每阶 8 个素数哈希表，每头 96 维（共 768/阶），FP8 embedding
- **iHC**：4 条残差流，`hc_mult=4`，`hc_magnitude=2.0`；sandwich norm
- **DFly2 drafter**：5 层，0.85B，复用目标 KV 与稀疏索引；`num_speculative_tokens=5`，目标层 25/29/33/37/41
- **视觉**：SigLIP2 派生的 27 层 tower（hidden 1152），patch 16，spatial_merge 2，MLP projector，1M 位置

## 3. 与 TensorFold 现有块的映射

| 构件 | Hy4 参数 | 现有候选复用点 | 判断 |
| --- | --- | --- | --- |
| 配置解析 | 见上 | `zig/src/families/flashnext/config.zig`（已有 hc_count / experts / topk / ngram / ple / index_* / mrope / mtp 字段） | **复用为主**，需加 sliding kind、per-layer rope、latent expert 宽、Engram 嵌套素数表 |
| 权重映射 | 见 §4 | `zig/src/families/deepseek_v41/cuda_weights.zig`、flashnext 的 tensor reader | 需新增 Hy4 命名到规范名的映射 |
| MLA + iHC + sink | gated MLA、4 流 iHC | `flashnext/cuda_batch_hc.zig`、`deepseek_v41/block_prefill_mhc.zig` | 复用，核对流数与 scale 语义 |
| 稀疏索引器 | 4:1 组池化 + topk 512 | `deepseek_v41/engram_*` 与 flashnext 的 indexer / `zig/src/core/dspine.zig` | 需改池化比率与 topk；注意 llama.cpp 专门新增了 ISWA 页缓存 |
| SWA 层 | window 512，64 头 | `zig/src/families/step5/config.zig`、`kolibri1`、`lighton_ocr` 的 sliding 支持 | 有现成先例，需支持两类层不同头数/rope |
| LatentMoE | 1536  latent，512 专家 | `zig/src/kernels/cuda/experts_rows.cu`、flashnext 的 MoE lane kernel | 需在路由后加 down/up 投影路径 |
| Engram 表 | 63.4B，FP8，4 层 | `deepseek_v41/engram_tables.zig` / `engram_hash.zig` / `engram_rows_*.zig` | 复用表/哈希基础设施；重建 Hy4 素数表布局与哈希乘子生成 |
| Drafter | DFly2 5 层 | `zig/src/core/lanes/drafter.zig`、`qwen3_5` 的 DFlash2 / MTP | 需新 glue；注意 DFly2 复用目标 KV 与索引 |
| 视觉 tower | SigLIP2 派生 | 现有 OCR/VL 族的 vision 路径 | 文本优先，视觉列为 Stage 3 |

## 4. 加载陷阱（官方实现踩过的坑）

1. `layer_types` 里的 `deepseek_sparse_attention` 在 transformers 中需 remap 成 `indexed_attention`；`rope_parameters` 按层类型键控，必须一起 remap，否则回落到默认 θ。
2. 权重重命名（官方 `conversion_mapping.py`，保存时会反向应用）：
   - `.self_attn.learnable_sink_param` → `.self_attn.sinks`
   - `.self_attn.linear_gate.weight` → `.self_attn.gate_proj.weight`
   - `.hc_attn_layer.hc_pre.hc_fn/hc_base/hc_scale` → `.attn_hc.fn/base/scale`
   - `.hc_mlp_layer.hc_pre.hc_fn/hc_base/hc_scale` → `.ffn_hc.fn/base/scale`
   - `.hc_head.hc_head_fn/hc_head_base/hc_head_scale` → `.hc_head.hc_fn/hc_base/hc_scale`
   - 视觉：`mlp.dense_h_to_4h/4h_to_h` → `mlp.fc1/fc2`；`vit.layers.N.input_layernorm` → `layer_norm1`；`post_attention_layernorm` → `layer_norm2`；`perceive.*` → `patch_merger.*`；`^vit` → `model.vision_tower`
   - 注意官方特意不用 HY4 的短目标 `.hc_` / `.hc_scale`，否则会误伤 `unified_per_hc_scales`。
3. Engram 哈希乘子**不在权重里**：由 `engram_seed=0` + 1-based 层号推导（llama.cpp 用 `seed + 10007 * layer_id` 的 NumPy 默认 RNG，奇数值）。配置里的 `engram_layer_head_vocab_sizes` 是 [ngram][head] 嵌套的素数桶数，要按 [bi,tri] × 8 展平。
4. Engram token map：HF 仓库没有单独的 `engram_vocab_map.npz`。实测官方 safetensors 的 `__metadata__` 里带 `engram_vocab_map_metadata`：`source_tokenizer_len=120818`、`compressed_vocab_size=92620`、`engram_seed=0`、`normalizer_signature=nfkc-nfd-stripaccents-lower-ws-strip-addedbypass-v2`、`lookup_sha256=1188e7ca8ca10b7d3328ce1f0d177b9df660b04b81e2deb053619ace67ba6a1d`。即 token→压缩 id 映射要由 tokenizer 按该归一化签名在dev 期生成，并用 sha256 校验；不能指望有现成文件。
5. 权重实数验证（HTTP Range 读 safetensors header，无需下载 440GB）：`model.layers.1.mlp.experts.gate_up_proj` 是裸 BF16 `[512,3072,1536]`（4.83 GB），不是 EXL3 打包格式。官方 BF16 检查点没有任何量化；PTQ 必须从 BF16 原始权重开始。
6. 哈希乘子的生成公式已用 numpy 2.0.2 验证并与官方 pin 值对齐：`bound=(int64_max // 92620)//2 = 49791470723681`，`rng=np.random.default_rng(seed + 10007*layer_id)`，`multipliers = rng.integers(0,bound,size=3)*2+1`；layer_id=2 得到 `[81853575799317, 39690085502883, 24229915018165]`。
7. `dfly_layers.*` 在 transformers 中被忽略；我们要做 drafter，必须保留（llama.cpp 把它导出成一个额外 block，`skip_mtp=False`）。
8. 容量：bf16 权重 440.5 GB。双 DGX Spark（2×128 GB）必须走 4-bit 或 W8A8；vLLM 实测 INT8 检查点 268.6 GiB、Engram 另有 126.9 GB pinned host，KV 60.5 GB/卡。按我们的容量红线，4-bit 下账面 ~110–130 GB，可行但要先确认 Engram 表的宿主内存策略。

## 5. 工作分解与验收

按 `docs/recipes/adding-a-zig-family.md` 的阶段执行：

- **Stage 0（配置/权重）**：新增 `hy_v4_flash` 配置解析与张量映射，能加载并列出全部张量，无 missing/unexpected key。验收：对照官方 PR 的“loads as published”清单。
- **Stage 1（文本前向）**：复用 MLA/iHC/MoE/Engram，实现混合注意力与 LatentMoE。验收：首 token 一致、8k/32k 后续 48 token 一致（见 recipe 的 exactness checks）。
- **Stage 2（推测解码）**：DFly2 5 层 drafter 接入 `lanes.drafter`。验收：drafted 与 plain 逐 token 相等；记录接受率与加速比。
- **Stage 3（多模态）**：SigLIP2 tower + projector + 图像/视频预处理。
- **Stage 4（性能）**：逐项套用 recipe 的 step 2–15，每项 A-B-A 并留 receipts。

## 6. 待拍板的问题

1. 目标硬件：双 DGX Spark（GB10，我们已有的生产口径）还是 Mac？两者 kernel 路径不同。
2. 量化档位：4-bit（EXL3/MLX affine）还是 W8A8 INT8？决定 weight reader 和 Engram 是否驻留宿主。
3. 范围：先文本-only（丢弃视觉，像 llama.cpp 那样单独导出 mmproj），还是一次性含视觉？
4. 是否沿用 qwen3.8-flash-next 的 CJK 草稿词表优化（PR#416）？Hy4 的 tokenizer 不同，需要重新统计。
5. Engram token map 的来源（见 §4.4）。

## 7. 参考链接

- 模型卡：https://huggingface.co/tencent/Hy4-flash
- 配置文件：https://huggingface.co/tencent/Hy4-flash/raw/main/config.json
- 权重索引：https://huggingface.co/tencent/Hy4-flash/raw/main/model.safetensors.index.json
- Transformers 实现 PR：https://github.com/huggingface/transformers/pull/50903
- llama.cpp 实现 PR：https://github.com/ggml-org/llama.cpp/pull/32517
- vLLM 插件实现 PR：https://github.com/flagos-ai/vllm-plugin-FL/pull/773
