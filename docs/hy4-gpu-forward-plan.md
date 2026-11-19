# Hy4 GPU full-layer forward — assembly plan

Verified components (DGX00 receipts, 2026-11-19):
- affine4 pack + decode/gemvMany: real-weight numerics 3.890e-3 = bf16 output precision; experts grouped 90.5 tok/s (96% HBM peak)
- mla kernels with Hy4 heads: absorb_q<192>, rope, chunks, mergePart, expand_v<256>, gate — 0.32-0.39 ms/layer at 4K cache
- embed rows, norms, route kernels exist shared (cuda.embed_rows / cuda.norms / cuda.route_rows)
- exl3_linear + exl3_experts: shared kernels, GLM production path — swap in when the 3bpw pack lands

Decode step assembly (per layer, single row):
1. embed row -> streams[hc_mult] all equal (embed_rows)
2. attn_hc: ihc.gates + collapse — GAP: no kernel; math = mixes=(flat@fn)*unweightedRMS(flat), pre=sigmoid(pre_l*pre_s+pre_b)+eps, post=mag*sigmoid(post_l*post_s+post_b)+eps, out=sum(pre*streams); small custom kernel or norms-based
3. input_layernorm (norms kernel)
4. q_a (affine4 decode) -> q_a_norm (norms) -> q_b -> split nope/rope -> rope (mla.rope)
5. kv_a -> split (kv_lora | k_rope) -> kv_a_norm -> kv_b -> k_nope|v; k_rope MQA expand
6. append to paged latent cache (latentWrite pattern from glm cuda_latent)
7. chunks + mergePart (mla) with sink support — VERIFY: mla chunks take sinks? glm attn has sinks; check tf_mla chunks kernel args
8. expand_v<256> -> gate (mla.gate) -> o_proj (affine4) -> post_proj_norm
9. streams update: stream_i = post_i * branch + stream_i (elementwise; small kernel)
10. ffn_hc collapse -> post_attention_layernorm -> MoE: router (route_rows or affine4 gemv on gate.weight) -> topk -> 16 experts gemvMany x4 -> shared experts affine4 -> combine norm
11. final: hc_head collapse (ihc-like, GAP) -> model.norm -> lm_head (affine4 decode)

Open gaps: ihc collapse kernel, hc_head collapse kernel, stream-update elementwise kernel, sink confirmation in chunks kernel.
Pack order: assemble on affine4 Q4 pack first (all pieces verified), then swap exl3 kernels for the 3bpw pack.
