// elementwise.mu — 逐元素算子（ADR L2-2）
//
// 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回。
// 所有指针 __restrict__（由 ops 层 alias 检测保证）。
// ABI 版本嵌入符号名：musapy_<op>_<dtype>_v<abi>（ADR L2-1）
//
// ABI 版本：
//   _v1: flat contiguous（v0.1-alpha，保留兼容）
//   _v2: stride-aware N-dimensional（v0.2-alpha, ADR-002-D2）

#include "include/common.h"
#include <math.h>

// ── 公共结构 ─────────────────────────────────────────────────

#define MUSAPY_MAX_NDIM 32

/// Binary kernel 参数（两个输入 strides）。
struct NdMeta {
    int ndim;
    size_t shape[MUSAPY_MAX_NDIM];
    ssize_t a_strides[MUSAPY_MAX_NDIM];
    ssize_t b_strides[MUSAPY_MAX_NDIM];
};

/// Unary kernel 参数（单输入 strides）。
struct NdMetaUnary {
    int ndim;
    size_t shape[MUSAPY_MAX_NDIM];
    ssize_t a_strides[MUSAPY_MAX_NDIM];
};

// ── v1: flat contiguous（v0.1-alpha，保留）──────────────────────

template <typename T>
__global__ void musapy_add_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ── v2 Binary kernels（ADR-002-D2）────────────────────────────

template <typename T>
__global__ void musapy_add_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = a[a_off] + b[b_off];
    }
}

template <typename T>
__global__ void musapy_sub_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = a[a_off] - b[b_off];
    }
}

template <typename T>
__global__ void musapy_mul_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = a[a_off] * b[b_off];
    }
}

template <typename T>
__global__ void musapy_div_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = a[a_off] / b[b_off];
    }
}

template <typename T>
__global__ void musapy_pow_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = pow(a[a_off], b[b_off]);
    }
}

// ── v2 Unary kernels ─────────────────────────────────────────

template <typename T>
__global__ void musapy_sin_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx] = sin(a[a_off]);
    }
}

template <typename T>
__global__ void musapy_cos_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx] = cos(a[a_off]);
    }
}

template <typename T>
__global__ void musapy_exp_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx] = exp(a[a_off]);
    }
}

template <typename T>
__global__ void musapy_log_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx] = log(a[a_off]);
    }
}

template <typename T>
__global__ void musapy_abs_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx] = fabs(a[a_off]);
    }
}

template <typename T>
__global__ void musapy_sign_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        T v = a[a_off];
        c[idx] = (v > T(0)) - (v < T(0));
    }
}

// ── v2 Neg kernel ────────────────────────────────────────────

template <typename T>
__global__ void musapy_neg_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx] = -a[a_off];
    }
}

// ── v2 Clamp kernel ──────────────────────────────────────────

template <typename T>
__global__ void musapy_clamp_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, T lo, T hi,
    NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        T v = a[a_off];
        c[idx] = v < lo ? lo : (v > hi ? hi : v);
    }
}

// ── v2 Cast kernel ───────────────────────────────────────────

template <typename Src, typename Dst>
__global__ void musapy_cast_kernel_v2(
    const Src* __restrict__ a, Dst* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx] = static_cast<Dst>(a[a_off]);
    }
}

// ── v2 Comparison kernels（Phase 3）──────────────────────────
// 输入 T（f32/f64），输出 uint8_t（bool: 0/1）

template <typename T>
__global__ void musapy_eq_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b,
    uint8_t* __restrict__ c, NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = (uint8_t)(a[a_off] == b[b_off]);
    }
}

template <typename T>
__global__ void musapy_ne_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b,
    uint8_t* __restrict__ c, NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = (uint8_t)(a[a_off] != b[b_off]);
    }
}

template <typename T>
__global__ void musapy_lt_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b,
    uint8_t* __restrict__ c, NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = (uint8_t)(a[a_off] < b[b_off]);
    }
}

template <typename T>
__global__ void musapy_gt_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b,
    uint8_t* __restrict__ c, NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = (uint8_t)(a[a_off] > b[b_off]);
    }
}

template <typename T>
__global__ void musapy_le_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b,
    uint8_t* __restrict__ c, NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = (uint8_t)(a[a_off] <= b[b_off]);
    }
}

template <typename T>
__global__ void musapy_ge_kernel_v2(
    const T* __restrict__ a, const T* __restrict__ b,
    uint8_t* __restrict__ c, NdMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);
        c[idx] = (uint8_t)(a[a_off] >= b[b_off]);
    }
}

// ── v2 Flat kernels（contiguous fast-path）─────────────────────
// 当所有输入 strides 为 C-contiguous 时，跳过 offset_nd 直接索引。

// Flat binary
template <typename T>
__global__ void musapy_add_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

template <typename T>
__global__ void musapy_sub_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

template <typename T>
__global__ void musapy_mul_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

template <typename T>
__global__ void musapy_div_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / b[idx];
}

template <typename T>
__global__ void musapy_pow_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = pow(a[idx], b[idx]);
}

// Flat unary
template <typename T>
__global__ void musapy_sin_flat_v2(const T* __restrict__ a, T* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = sin(a[idx]);
}

template <typename T>
__global__ void musapy_cos_flat_v2(const T* __restrict__ a, T* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = cos(a[idx]);
}

template <typename T>
__global__ void musapy_exp_flat_v2(const T* __restrict__ a, T* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = exp(a[idx]);
}

template <typename T>
__global__ void musapy_log_flat_v2(const T* __restrict__ a, T* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = log(a[idx]);
}

template <typename T>
__global__ void musapy_abs_flat_v2(const T* __restrict__ a, T* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = fabs(a[idx]);
}

template <typename T>
__global__ void musapy_sign_flat_v2(const T* __restrict__ a, T* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { T v = a[idx]; c[idx] = (v > T(0)) - (v < T(0)); }
}

template <typename T>
__global__ void musapy_neg_flat_v2(const T* __restrict__ a, T* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = -a[idx];
}

// Flat clamp
template <typename T>
__global__ void musapy_clamp_flat_v2(
    const T* __restrict__ a, T* __restrict__ c, T lo, T hi, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { T v = a[idx]; c[idx] = v < lo ? lo : (v > hi ? hi : v); }
}

// Flat cast
template <typename Src, typename Dst>
__global__ void musapy_cast_flat_v2(
    const Src* __restrict__ a, Dst* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = static_cast<Dst>(a[idx]);
}

// Flat comparison
template <typename T>
__global__ void musapy_eq_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (uint8_t)(a[idx] == b[idx]);
}

template <typename T>
__global__ void musapy_ne_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (uint8_t)(a[idx] != b[idx]);
}

template <typename T>
__global__ void musapy_lt_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (uint8_t)(a[idx] < b[idx]);
}

template <typename T>
__global__ void musapy_gt_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (uint8_t)(a[idx] > b[idx]);
}

template <typename T>
__global__ void musapy_le_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (uint8_t)(a[idx] <= b[idx]);
}

template <typename T>
__global__ void musapy_ge_flat_v2(
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (uint8_t)(a[idx] >= b[idx]);
}

// ── v2 Flat float4 向量化（P3）───────────────────────────────
// f32 flat-contiguous + 16B 对齐 + n%4==0 + n≥1M 时启用（wrapper 内自检，
// 无 FFI/ABI 变化）。每线程 4 元素单指令读写；mp_22 实测 +3.6~5.3%
// @4M/16M（kernel 已接近内存带宽上限，余量有限）。P2 教训：float4
// 不与 warp shuffle 同函数即可安全使用。

struct VecOpAdd { __device__ static float apply(float x, float y) { return x + y; } };
struct VecOpSub { __device__ static float apply(float x, float y) { return x - y; } };
struct VecOpMul { __device__ static float apply(float x, float y) { return x * y; } };
struct VecOpDiv { __device__ static float apply(float x, float y) { return x / y; } };
struct VecOpPow { __device__ static float apply(float x, float y) { return powf(x, y); } };
struct VecOpAbs { __device__ static float apply(float x) { return fabsf(x); } };
struct VecOpNeg { __device__ static float apply(float x) { return -x; } };
struct VecOpExp { __device__ static float apply(float x) { return expf(x); } };
struct VecOpLog { __device__ static float apply(float x) { return logf(x); } };
struct VecOpSin { __device__ static float apply(float x) { return sinf(x); } };
struct VecOpCos { __device__ static float apply(float x) { return cosf(x); } };
struct VecOpSign {
    __device__ static float apply(float x) { return (x > 0.0f) - (x < 0.0f); }
};

/// 向量化路径统一门槛（host wrapper 内检查）。
#define VEC4_THRESHOLD 1000000

/// 二进制 vec4 kernel（每线程 4 元素）。
template <typename Op>
__global__ void musapy_binary_vec4_kernel(
    const float4* __restrict__ a, const float4* __restrict__ b,
    float4* __restrict__ c, size_t n4
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i];
        float4 y = b[i];
        c[i] = make_float4(Op::apply(x.x, y.x), Op::apply(x.y, y.y),
                           Op::apply(x.z, y.z), Op::apply(x.w, y.w));
    }
}

/// 一元 vec4 kernel（每线程 4 元素）。
template <typename Op>
__global__ void musapy_unary_vec4_kernel(
    const float4* __restrict__ a, float4* __restrict__ c, size_t n4
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i];
        c[i] = make_float4(Op::apply(x.x), Op::apply(x.y),
                           Op::apply(x.z), Op::apply(x.w));
    }
}

// ── extern "C" 稳定 ABI ────────────────────────────────────────

extern "C" {

// ── v1 符号（保留，L4-3 兼容性）──

void musapy_add_f32_v1(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, size_t n, musaStream_t stream
) {
    musapy_add_kernel<float><<<grid_size_1d(n), 256, 0, stream>>>(a, b, c, n);
}

void musapy_add_f64_v1(
    const double* __restrict__ a, const double* __restrict__ b,
    double* __restrict__ c, size_t n, musaStream_t stream
) {
    musapy_add_kernel<double><<<grid_size_1d(n), 256, 0, stream>>>(a, b, c, n);
}

// ── v2 Binary 符号 ──
// 宏：生成 binary op 的 f32/f64 wrapper（含 contiguous fast-path + P3 vec4）
#define BINARY_V2(OP, VECOP)                                                  \
void musapy_##OP##_f32_v2(                                                   \
    const float* __restrict__ a, const float* __restrict__ b,                \
    float* __restrict__ c, int ndim, const size_t* shape,                    \
    const ssize_t* a_strides, const ssize_t* b_strides, musaStream_t stream  \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                           \
    if (is_contiguous_strides(shape, a_strides, ndim) &&                     \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        if (n >= VEC4_THRESHOLD && (n & 3) == 0 &&                           \
            (((uintptr_t)a & 15) == 0) && (((uintptr_t)b & 15) == 0) &&      \
            (((uintptr_t)c & 15) == 0)) {                                    \
            musapy_binary_vec4_kernel<VECOP><<<grid_size_1d(n >> 2), 256,    \
                0, stream>>>((const float4*)a, (const float4*)b,             \
                (float4*)c, n >> 2);                                         \
        } else {                                                             \
            musapy_##OP##_flat_v2<float><<<grid_size_1d(n), 256, 0, stream>>>(\
                a, b, c, n);                                                 \
        }                                                                    \
    } else {                                                                 \
        NdMeta meta;                                                         \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
            meta.b_strides[i] = b_strides[i];                                \
        }                                                                    \
        musapy_##OP##_kernel_v2<float><<<grid_size_1d(n), 256, 0, stream>>>( \
            a, b, c, meta, n);                                               \
    }                                                                        \
}                                                                            \
void musapy_##OP##_f64_v2(                                                   \
    const double* __restrict__ a, const double* __restrict__ b,              \
    double* __restrict__ c, int ndim, const size_t* shape,                   \
    const ssize_t* a_strides, const ssize_t* b_strides, musaStream_t stream  \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                           \
    if (is_contiguous_strides(shape, a_strides, ndim) &&                     \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_flat_v2<double><<<grid_size_1d(n), 256, 0, stream>>>(  \
            a, b, c, n);                                                     \
    } else {                                                                 \
        NdMeta meta;                                                         \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
            meta.b_strides[i] = b_strides[i];                                \
        }                                                                    \
        musapy_##OP##_kernel_v2<double><<<grid_size_1d(n), 256, 0, stream>>>(\
            a, b, c, meta, n);                                               \
    }                                                                        \
}

BINARY_V2(add, VecOpAdd)
BINARY_V2(sub, VecOpSub)
BINARY_V2(mul, VecOpMul)
BINARY_V2(div, VecOpDiv)
BINARY_V2(pow, VecOpPow)

#undef BINARY_V2

// ── v2 Unary 符号 ──
#define UNARY_V2(OP, VECOP)                                                   \
void musapy_##OP##_f32_v2(                                                   \
    const float* __restrict__ a, float* __restrict__ c,                      \
    int ndim, const size_t* shape, const ssize_t* a_strides,                 \
    musaStream_t stream                                                      \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                           \
    if (is_contiguous_strides(shape, a_strides, ndim)) {                     \
        if (n >= VEC4_THRESHOLD && (n & 3) == 0 &&                           \
            (((uintptr_t)a & 15) == 0) && (((uintptr_t)c & 15) == 0)) {      \
            musapy_unary_vec4_kernel<VECOP><<<grid_size_1d(n >> 2), 256,     \
                0, stream>>>((const float4*)a, (float4*)c, n >> 2);          \
        } else {                                                             \
            musapy_##OP##_flat_v2<float><<<grid_size_1d(n), 256, 0, stream>>>(\
                a, c, n);                                                    \
        }                                                                    \
    } else {                                                                 \
        NdMetaUnary meta;                                                    \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
        }                                                                    \
        musapy_##OP##_kernel_v2<float><<<grid_size_1d(n), 256, 0, stream>>>( \
            a, c, meta, n);                                                  \
    }                                                                        \
}                                                                            \
void musapy_##OP##_f64_v2(                                                   \
    const double* __restrict__ a, double* __restrict__ c,                    \
    int ndim, const size_t* shape, const ssize_t* a_strides,                 \
    musaStream_t stream                                                      \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                           \
    if (is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_##OP##_flat_v2<double><<<grid_size_1d(n), 256, 0, stream>>>(  \
            a, c, n);                                                        \
    } else {                                                                 \
        NdMetaUnary meta;                                                    \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
        }                                                                    \
        musapy_##OP##_kernel_v2<double><<<grid_size_1d(n), 256, 0, stream>>>(\
            a, c, meta, n);                                                  \
    }                                                                        \
}

UNARY_V2(sin, VecOpSin)
UNARY_V2(cos, VecOpCos)
UNARY_V2(exp, VecOpExp)
UNARY_V2(log, VecOpLog)
UNARY_V2(abs, VecOpAbs)
UNARY_V2(sign, VecOpSign)
UNARY_V2(neg, VecOpNeg)

#undef UNARY_V2

// ── v2 Clamp 符号 ──

void musapy_clamp_f32_v2(
    const float* __restrict__ a, float* __restrict__ c,
    float lo, float hi,
    int ndim, const size_t* shape, const ssize_t* a_strides,
    musaStream_t stream
) {
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    if (is_contiguous_strides(shape, a_strides, ndim)) {
        musapy_clamp_flat_v2<float><<<grid_size_1d(n), 256, 0, stream>>>(
            a, c, lo, hi, n);
    } else {
        NdMetaUnary meta;
        meta.ndim = ndim;
        for (int i = 0; i < ndim; i++) {
            meta.shape[i] = shape[i];
            meta.a_strides[i] = a_strides[i];
        }
        musapy_clamp_kernel_v2<float><<<grid_size_1d(n), 256, 0, stream>>>(
            a, c, lo, hi, meta, n);
    }
}

void musapy_clamp_f64_v2(
    const double* __restrict__ a, double* __restrict__ c,
    double lo, double hi,
    int ndim, const size_t* shape, const ssize_t* a_strides,
    musaStream_t stream
) {
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    if (is_contiguous_strides(shape, a_strides, ndim)) {
        musapy_clamp_flat_v2<double><<<grid_size_1d(n), 256, 0, stream>>>(
            a, c, lo, hi, n);
    } else {
        NdMetaUnary meta;
        meta.ndim = ndim;
        for (int i = 0; i < ndim; i++) {
            meta.shape[i] = shape[i];
            meta.a_strides[i] = a_strides[i];
        }
        musapy_clamp_kernel_v2<double><<<grid_size_1d(n), 256, 0, stream>>>(
            a, c, lo, hi, meta, n);
    }
}

// ── v2 Cast 符号 ──
// 宏：生成 cast 的 wrapper（Src → Dst，含 contiguous fast-path）
#define CAST_V2(SRC_C, SRC_T, DST_C, DST_T)                                  \
void musapy_cast_##SRC_C##_##DST_C##_v2(                                     \
    const SRC_T* __restrict__ a, DST_T* __restrict__ c,                      \
    int ndim, const size_t* shape, const ssize_t* a_strides,                 \
    musaStream_t stream                                                      \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                           \
    if (is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_cast_flat_v2<SRC_T, DST_T><<<grid_size_1d(n), 256, 0, stream>>>(\
            a, c, n);                                                        \
    } else {                                                                 \
        NdMetaUnary meta;                                                    \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
        }                                                                    \
        musapy_cast_kernel_v2<SRC_T, DST_T><<<grid_size_1d(n), 256, 0, stream>>>(\
            a, c, meta, n);                                                  \
    }                                                                        \
}

// → float32
CAST_V2(i8,  int8_t,   f32, float)
CAST_V2(i16, int16_t,  f32, float)
CAST_V2(i32, int32_t,  f32, float)
CAST_V2(i64, int64_t,  f32, float)
CAST_V2(u8,  uint8_t,  f32, float)
CAST_V2(u16, uint16_t, f32, float)
CAST_V2(u32, uint32_t, f32, float)
CAST_V2(u64, uint64_t, f32, float)
CAST_V2(f64, double,   f32, float)

// → float64
CAST_V2(i8,  int8_t,   f64, double)
CAST_V2(i16, int16_t,  f64, double)
CAST_V2(i32, int32_t,  f64, double)
CAST_V2(i64, int64_t,  f64, double)
CAST_V2(u8,  uint8_t,  f64, double)
CAST_V2(u16, uint16_t, f64, double)
CAST_V2(u32, uint32_t, f64, double)
CAST_V2(u64, uint64_t, f64, double)
CAST_V2(f32, float,    f64, double)

// → int64（Phase 4 reduction 整数累加用）
CAST_V2(i8,  int8_t,   i64, int64_t)
CAST_V2(i16, int16_t,  i64, int64_t)
CAST_V2(i32, int32_t,  i64, int64_t)
CAST_V2(u8,  uint8_t,  i64, int64_t)
CAST_V2(u16, uint16_t, i64, int64_t)
CAST_V2(u32, uint32_t, i64, int64_t)
CAST_V2(u64, uint64_t, i64, int64_t)

#undef CAST_V2

// extern "C" wrapper 宏（输入 T，输出 uint8_t，含 contiguous fast-path）
#define COMPARE_V2(OP)                                                        \
void musapy_##OP##_f32_v2(                                                   \
    const float* __restrict__ a, const float* __restrict__ b,                \
    uint8_t* __restrict__ c,                                                 \
    int ndim, const size_t* shape,                                           \
    const ssize_t* a_strides, const ssize_t* b_strides,                      \
    musaStream_t stream                                                      \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                           \
    if (is_contiguous_strides(shape, a_strides, ndim) &&                     \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_flat_v2<float><<<grid_size_1d(n), 256, 0, stream>>>(   \
            a, b, c, n);                                                     \
    } else {                                                                 \
        NdMeta meta;                                                         \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
            meta.b_strides[i] = b_strides[i];                                \
        }                                                                    \
        musapy_##OP##_kernel_v2<float><<<grid_size_1d(n), 256, 0, stream>>>( \
            a, b, c, meta, n);                                               \
    }                                                                        \
}                                                                            \
void musapy_##OP##_f64_v2(                                                   \
    const double* __restrict__ a, const double* __restrict__ b,              \
    uint8_t* __restrict__ c,                                                 \
    int ndim, const size_t* shape,                                           \
    const ssize_t* a_strides, const ssize_t* b_strides,                      \
    musaStream_t stream                                                      \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                           \
    if (is_contiguous_strides(shape, a_strides, ndim) &&                     \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_flat_v2<double><<<grid_size_1d(n), 256, 0, stream>>>(  \
            a, b, c, n);                                                     \
    } else {                                                                 \
        NdMeta meta;                                                         \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
            meta.b_strides[i] = b_strides[i];                                \
        }                                                                    \
        musapy_##OP##_kernel_v2<double><<<grid_size_1d(n), 256, 0, stream>>>(\
            a, b, c, meta, n);                                               \
    }                                                                        \
}

COMPARE_V2(eq)
COMPARE_V2(ne)
COMPARE_V2(lt)
COMPARE_V2(gt)
COMPARE_V2(le)
COMPARE_V2(ge)

#undef COMPARE_V2

} // extern "C"
