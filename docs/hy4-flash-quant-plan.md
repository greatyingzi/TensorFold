# Hy4 Flash 量化方案（4-bit + Q8 Engram）

状态：方案已定，代码未实现。本文件是 P2 的合约。

## 1. 源与档位

- **源检查点只能是官方原始精度**：`tencent/Hy4-flash`（BF16，440.5 GB）。已用 HTTP Range 读 safetensors header 实证：专家张量 `model.layers.N.mlp.experts.gate_up_proj` 是裸 BF16 `[512,3072,1536]`，官方检查点**没有任何量化**。
- **不从官方 FP8 或社区 W8A8 二次量化**：二次量化会叠加误差，而且不尊重源 scale。
- **不接受 W8A8/INT8 激活作为默认路径**：违反 TensorFold 的精度策略；若将来支持，必须是单独标注的 checkpoint-defined mode 并有自己的保真证据。

## 2. 目标格式

| 张量族 | 目标 | 理由 |
| --- | --- | --- |
| 路由专家 `mlp.experts.gate_up_proj` / `down_proj` | EXL3 K4（trellis/suh/svh，复用 `zig/src/core/exl3_layout.zig`） | 占权重绝大部分；已有布局与 kernel 基础设施 |
| attention 投影、router、shared expert、norm、sink、hc_* | 多精度 core view：original BF16 / K8 / K6 三选一，加载时按设备内存选择 | 小矩阵对 4-bit 敏感；参考社区 MC（Multi-Core）检查点思路 |
| Engram `multi_head_embedding.embedding.weight`（4 层，BF16 `[~165M,96]`/层，共约 127 GB） | 一次性压到 FP8/E4M3 + 每行 F32 scale；保留 `engram_fp8_embedding` 语义 | 不压则总权重约 205 GB，双 Spark 贴线 |
| Engram 其余小投影/norm、lm_head、embed_tokens | BF16 | 体积占比小，保真优先 |

名字沿用官方 checkpoint，不另造一套；档位由 `quantization_config` 与 safetensors header 元数据声明。

## 3. 容量估算（双 DGX Spark，2×128 GB，TP=2）

- backbone 155.5B × 0.5 B ≈ 77.8 GB
- Engram 63.5B × 1 B（FP8 + scale）≈ 63.5 GB
- ViT 0.43B + DFly2 0.85B ≈ 0.7 GB
- 权重合计 ≈ 142 GB → 单 rank ≈ 71 GB
- KV/buffer 单 rank 约 15–20 GB → 单 rank 峰值约 90 GB，128 GB 内有余量

## 4. 保真与验收

1. 同一合成/真实检查点，BF16 plain 为 oracle；4-bit 路径必须报告逐 token 一致率与 top-token 差异/margin，不能用平均误差掩盖最差列。
2. 任何改变求和顺序的 kernel 需要独立 FP64 参考，按投影类别分别给出误差；prompt 与 decode 分开比。
3. drafted 与 plain 逐 token 相等是 drafter 的硬门禁，量化不能成为关闭 drafting 的理由。
4. Engram QDQ 要覆盖 E4M3FN 中点、subnormal、1e-12 scale 下限与 ±448 端点（参考官方测试的边界用例）。
5. 性能 receipts 按 `docs/recipes/adding-a-zig-family.md`：同机同检查点同设置，A-B-A，记录 GPU/驱动/提交/量化档位。

## 5. 开发期工具

- `tools/zig/hy4_synthetic.py`：生成小尺寸同构检查点，供 loader/前向单测使用，不依赖 440 GB 权重。
- `tools/zig/hy4_ptq.py`（待写）：从 BF16 源生成 K4 专家 + FP8 Engram 的检查点，并输出按张量的量化误差报告。
- Engram token map 已在 checkpoint 里：`engram_layer.lookup_table` 是 I32 [120818] 的映射表（每层一份）；safetensors header 的 `engram_vocab_map_metadata` 带 `lookup_sha256` 可用于校验来源，不需要外部 npz 或从 tokenizer 重建。
- 哈希乘子：用 NumPy PCG64 复现（见 `engram_rng.zig`），layer_id 为 1-based，seed 0。
