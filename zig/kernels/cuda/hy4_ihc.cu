// Hy4 Flash identity-hyper-connections (iHC) on CUDA.
//
// Host reference: zig/src/families/hy_v4_flash/ihc.zig with ops.zig.
//   ops.matvec and the unweighted RMS accumulate in f64 and round once to f32;
//   ops.sigmoid evaluates exp in f64 and returns f32;
//   collapse accumulates gate * stream in ascending stream order in f32.
// Stream tensors are bf16 [rows][HC][H]; the flattened width is N = HC * H
// (HC = 4, H = 3072).  Every fn row has stride N -- 2*HC rows for the gate
// matvec of collapse, HC rows for hc_head (the zig hc_head.hc_fn is
// [hc, hc*hidden]).  One block handles one row with 256 threads.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace tf_hy4_ihc_detail {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;   // 8
constexpr int MAX_HC = 8;
constexpr int MAX_MIX = 2 * MAX_HC;   // 16

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// ops.sigmoid(float): f64 exponentiation, then a single round back to f32.
__device__ __forceinline__ float sigmoid_f32(float x) {
    const double y = static_cast<double>(x);
    return static_cast<float>(1.0 / (1.0 + exp(-y)));
}

// A fixed-order butterfly; f64 partials make the tree order irrelevant at
// bf16 output precision, while keeping the host matvec / norm accumulation.
__device__ __forceinline__ float s_mix_f(double s_mix[MAX_MIX][WARPS], int i) {
    double sum = 0.0;
#pragma unroll
    for (int k = 0; k < WARPS; ++k) sum += s_mix[i][k];
    return static_cast<float>(sum);
}

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

}  // namespace tf_hy4_ihc_detail

using namespace tf_hy4_ihc_detail;

// Pre-gate collapse.  fn holds 2*HC rows of N = HC*H floats; base/scale hold
// 2*HC and 2 entries respectively.  post == 0 selects the pre gates, any
// nonzero value the magnitude-scaled post gates.
extern "C" __global__ void __launch_bounds__(256)
tf_hy4_ihc_collapse(const __nv_bfloat16* streams, const float* fn, const float* base,
                    const float* scale, __nv_bfloat16* out, int rows, int H, int HC,
                    float eps, float magnitude, int post) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int N = HC * H;
    const int nmix = 2 * HC;
    const size_t soff = static_cast<size_t>(row) * N;

    __shared__ double s_mix[MAX_MIX][WARPS];   // per-warp mix partials
    __shared__ double s_ss[WARPS];             // per-warp sum of squares
    __shared__ float s_mixf[MAX_MIX];          // rounded f32 mixes
    __shared__ float s_gate[MAX_HC];           // gate for each stream

    double mix[MAX_MIX];
#pragma unroll
    for (int i = 0; i < MAX_MIX; ++i) mix[i] = 0.0;
    double ss = 0.0;

    for (int j = tid; j < N; j += THREADS) {
        const double d = static_cast<double>(bf2f(streams[soff + j]));
        ss += d * d;
#pragma unroll
        for (int i = 0; i < MAX_MIX; ++i) {
            if (i < nmix) mix[i] += static_cast<double>(fn[static_cast<size_t>(i) * N + j]) * d;
        }
    }

    // __shfl_xor_sync needs every lane of the warp to participate; compute the
    // reductions on all lanes first, then let lane 0 publish the warp's partials.
    const double ss_w = warp_sum(ss);
    double mix_w[MAX_MIX];
#pragma unroll
    for (int i = 0; i < MAX_MIX; ++i) mix_w[i] = warp_sum(mix[i]);
    if (lane == 0) {
        s_ss[warp] = ss_w;
#pragma unroll
        for (int i = 0; i < MAX_MIX; ++i) s_mix[i][warp] = mix_w[i];
    }
    __syncthreads();

    if (tid == 0) {
        double total = 0.0;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) total += s_ss[w];
        const float norm = static_cast<float>(
            1.0 / sqrt(total / static_cast<double>(N) + static_cast<double>(eps)));
#pragma unroll
        for (int i = 0; i < MAX_MIX; ++i) {
            double sum = 0.0;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) sum += s_mix[i][w];
            s_mixf[i] = static_cast<float>(sum);
        }
#pragma unroll
        for (int c = 0; c < MAX_HC; ++c) {
            if (c < HC) {
                const float pre_logit = __fmul_rn(s_mixf[c], norm);
                const float pre_arg = __fadd_rn(__fmul_rn(pre_logit, scale[0]), base[c]);
                const float pre = __fadd_rn(sigmoid_f32(pre_arg), eps);
                const float post_logit = __fmul_rn(s_mixf[HC + c], norm);
                const float post_arg = __fadd_rn(__fmul_rn(post_logit, scale[1]), base[HC + c]);
                const float pg = __fadd_rn(__fmul_rn(magnitude, sigmoid_f32(post_arg)), eps);
                s_gate[c] = post ? pg : pre;
            }
        }
    }
    __syncthreads();

    for (int h = tid; h < H; h += THREADS) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < MAX_HC; ++c) {
            if (c < HC) {
                const float v = bf2f(streams[soff + static_cast<size_t>(c) * H + h]);
                acc = __fadd_rn(acc, __fmul_rn(s_gate[c], v));
            }
        }
        out[static_cast<size_t>(row) * H + h] = __float2bfloat16_rn(acc);
    }
}


// Gate-only helper: write pre[HC] and post[HC] for the streams of each row.
// Used by the MoE residual update, which needs the post gates rather than a collapsed vector.
extern "C" __global__ void __launch_bounds__(256)
tf_hy4_ihc_gates(const __nv_bfloat16* streams, const float* fn, const float* base,
                 const float* scale, float* pre_out, float* post_out,
                 int rows, int H, int HC, float eps, float magnitude) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int N = HC * H;
    const int nmix = 2 * HC;
    const size_t soff = static_cast<size_t>(row) * N;

    __shared__ double s_mix[MAX_MIX][WARPS];
    __shared__ double s_ss[WARPS];
    __shared__ float s_pre[MAX_HC];
    __shared__ float s_post[MAX_HC];

    double mix[MAX_MIX];
#pragma unroll
    for (int i = 0; i < MAX_MIX; ++i) mix[i] = 0.0;
    double ss = 0.0;
    for (int j = tid; j < N; j += THREADS) {
        const double d = static_cast<double>(bf2f(streams[soff + j]));
        ss += d * d;
#pragma unroll
        for (int i = 0; i < MAX_MIX; ++i) {
            if (i < nmix) mix[i] += static_cast<double>(fn[static_cast<size_t>(i) * N + j]) * d;
        }
    }
    const double ss_w = warp_sum(ss);
    double mix_w[MAX_MIX];
#pragma unroll
    for (int i = 0; i < MAX_MIX; ++i) mix_w[i] = warp_sum(mix[i]);
    if (lane == 0) {
        s_ss[warp] = ss_w;
#pragma unroll
        for (int i = 0; i < MAX_MIX; ++i) s_mix[i][warp] = mix_w[i];
    }
    __syncthreads();
    if (tid == 0) {
        double total = 0.0;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) total += s_ss[w];
        const float norm = static_cast<float>(1.0 / sqrt(total / static_cast<double>(N) + static_cast<double>(eps)));
#pragma unroll
        for (int c = 0; c < MAX_HC; ++c) {
            if (c < HC) {
                const float pl = __fmul_rn(s_mix_f(s_mix, c), norm);
                const float pa = __fadd_rn(__fmul_rn(pl, scale[0]), base[c]);
                s_pre[c] = __fadd_rn(sigmoid_f32(pa), eps);
                const float ql = __fmul_rn(s_mix_f(s_mix, HC + c), norm);
                const float qa = __fadd_rn(__fmul_rn(ql, scale[1]), base[HC + c]);
                s_post[c] = __fadd_rn(__fmul_rn(magnitude, sigmoid_f32(qa)), eps);
            }
        }
    }
    __syncthreads();
    if (tid < HC) {
        pre_out[static_cast<size_t>(row) * HC + tid] = s_pre[tid];
        post_out[static_cast<size_t>(row) * HC + tid] = s_post[tid];
    }
}


// Elementwise output gate: o[i] = bf16(bf16(o[i]) * sigmoid(g[i])), g fp32.
extern "C" __global__ void __launch_bounds__(256)
tf_hy4_gate_sigmoid(__nv_bfloat16* o, const float* g, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[i] = __float2bfloat16_rn(__fmul_rn(bf2f(o[i]), sigmoid_f32(g[i])));
}

// In-place residual write-back: streams[r][c][h] += post[c] * branch[r][h].
extern "C" __global__ void __launch_bounds__(256)
tf_hy4_stream_update(__nv_bfloat16* streams, const __nv_bfloat16* branch, const float* post,
                     int rows, int H, int HC) {
    const long long total = static_cast<long long>(rows) * HC * H;
    const long long idx = static_cast<long long>(blockIdx.x) * THREADS + threadIdx.x;
    if (idx >= total) return;
    const int h = static_cast<int>(idx % H);
    const int c = static_cast<int>((idx / H) % HC);
    const long long r = idx / (static_cast<long long>(H) * HC);
    const float added = __fmul_rn(post[c], bf2f(branch[static_cast<size_t>(r) * H + h]));
    const float v = __fadd_rn(added, bf2f(streams[idx]));
    streams[idx] = __float2bfloat16_rn(v);
}

// Final head collapse.  fn holds HC rows of N = HC*H floats (hc_head.hc_fn),
// base has HC entries and scale has one entry.
extern "C" __global__ void __launch_bounds__(256)
tf_hy4_hc_head(const __nv_bfloat16* streams, const float* fn, const float* base,
               const float* scale, __nv_bfloat16* out, int rows, int H, int HC, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int N = HC * H;
    const size_t soff = static_cast<size_t>(row) * N;

    __shared__ double s_mix[MAX_HC][WARPS];
    __shared__ double s_ss[WARPS];
    __shared__ float s_mixf[MAX_HC];
    __shared__ float s_gate[MAX_HC];

    double mix[MAX_HC];
#pragma unroll
    for (int i = 0; i < MAX_HC; ++i) mix[i] = 0.0;
    double ss = 0.0;

    for (int j = tid; j < N; j += THREADS) {
        const double d = static_cast<double>(bf2f(streams[soff + j]));
        ss += d * d;
#pragma unroll
        for (int i = 0; i < MAX_HC; ++i) {
            if (i < HC) mix[i] += static_cast<double>(fn[static_cast<size_t>(i) * N + j]) * d;
        }
    }

    const double ss_w = warp_sum(ss);
    double mix_w[MAX_HC];
#pragma unroll
    for (int i = 0; i < MAX_HC; ++i) mix_w[i] = warp_sum(mix[i]);
    if (lane == 0) {
        s_ss[warp] = ss_w;
#pragma unroll
        for (int i = 0; i < MAX_HC; ++i) s_mix[i][warp] = mix_w[i];
    }
    __syncthreads();

    if (tid == 0) {
        double total = 0.0;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) total += s_ss[w];
        const float norm = static_cast<float>(
            1.0 / sqrt(total / static_cast<double>(N) + static_cast<double>(eps)));
#pragma unroll
        for (int i = 0; i < MAX_HC; ++i) {
            double sum = 0.0;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) sum += s_mix[i][w];
            s_mixf[i] = static_cast<float>(sum);
        }
#pragma unroll
        for (int c = 0; c < MAX_HC; ++c) {
            if (c < HC) {
                const float logit = __fmul_rn(s_mixf[c], norm);
                const float arg = __fadd_rn(__fmul_rn(logit, scale[0]), base[c]);
                s_gate[c] = __fadd_rn(sigmoid_f32(arg), eps);
            }
        }
    }
    __syncthreads();

    for (int h = tid; h < H; h += THREADS) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < MAX_HC; ++c) {
            if (c < HC) {
                const float v = bf2f(streams[soff + static_cast<size_t>(c) * H + h]);
                acc = __fadd_rn(acc, __fmul_rn(s_gate[c], v));
            }
        }
        out[static_cast<size_t>(row) * H + h] = __float2bfloat16_rn(acc);
    }
}
