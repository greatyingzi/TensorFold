# Hy4 Flash 适配攻坚 — 最终交接文档

## 目标达成情况（诚实）
- ① 前向数值正确性：**已证明**（与官方 transformers 单步+5步逐位一致）；生成质量：**未达标**（非语义）
- ② 速度：未达标（10 tok/s vs 目标 63-90）
- ③ DFly2 草稿接入 GPU：未做
- ④ 服务在线：**达成**（hy4serve_v9, HTTP 200）

## 最重要的结论（铁证）
1. **我们的前向实现数学完全正确**：用 torch+transformers PR#50903 搭权威对照，映射 134/135 synthetic fixture 权重进官方 HYV4FlashTextModel，单步与 5 步变 token 序列 [1,5,9,13,20] 的 logits 与 Zig session **逐位精确一致**（4 位小数完全相同）。质量问题不在我们的前向代码。
2. BPE 编码、chat template token 结构均已对照官方验证/修正。
3. BF16 源包 + 正确前向生成仍非语义（"The capital of France is" → ">. "）。逻辑上非语义是该 checkpoint 文本路径属性。
4. 模型是 HYV4FlashForConditionalGeneration（多模态）；官方 quickstart 用 AutoModelForImageTextToText + processor + image。

## 修复的 10+ 个真 bug（都已提交）
- stream-update 内核 grid 2D→1D（stream1/2/3 从未更新）
- gpu_pack routed_scale 1.0→3.998（从 config 读取）
- config 全字段从 json 读取（rms_norm_eps 1e-6→1e-5 等）
- engram bf16 dtype + 按需行解码（避免 15GB 表全量解码 OOM）
- AttentionCacheFull: backend cacheCap 64→context
- post_proj_norm（o_proj 后 RMSNorm，CPU+GPU 都漏了）
- session attention gate 权重名 gate_proj.weight（非 linear_gate.weight，rename 后）
- chat template think token 格式（120029/120030）
- 服务内存 reserve 12→6 GiB，weight admission 70→66 GiB
- 官方结构 chat template（特殊 token）

## 可复用验证工具链
-权威对照：/home/leon/venvs/hf（torch 2.15 cpu + transformers PR50903）；脚本 offseq.py / reffwd2.py（DGX）
- Zig 导出：hy4_runner.zig 的 exportlogits mode（synthetic fixture，5步序列）
- 官方 logits 基准：token1 = [2.6218, 4.7408, 2.6756, 0.9617, -0.4004, -1.4197, -2.0874, -2.4079]

## 下一步（需用户决策）
1. 提供能正常输出的参考配置/命令（哪个模型、是否带图像输入、哪个 commit）——这是定位质量问题的唯一钥匙
2. 若确认需多模态路径：实现 processor 的 mm_token_type_ids / 图像预处理
3. 速度优化：从 10 tok/s 提升到理论值（需 profiler 分析瓶颈）
4. DFly2 草稿接入 GPU 服务
5. 硬件约束：完整模型需 TP8（Engram 127GB 单卡装不下）

## 关键环境
- DGX: leon@100.104.125.29, GPU GB10 128GB unified, zig /home/leon/zig017
- fork: git@github.com:greatyingzi/TensorFold.git, branch hy4-flash-adapt
- 服务脚本: /home/leon/hy4serve_correct.sh, 模板 /home/leon/hy4_chat_correct.jinja
- 服务: docker hy4serve_v9, port 8080
