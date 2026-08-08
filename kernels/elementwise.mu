// elementwise.mu — 逐元素算子（ADR L2-2）
//
// 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回。
// 所有指针 __restrict__（由 ops 层 alias 检测保证）。
// ABI 版本嵌入符号名：musapy_<op>_<dtype>_v<abi>（ADR L2-1）
//
// ABI 版本：
//   _v2: stride-aware N-dimensional（v0.2-alpha, ADR-002-D2）
//   （v1 flat 符号于 P6 清理删除——Rust 侧从未调用，_flat_v2 已覆盖）

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

// ── 标量广播 fast-path（P1）───────────────────────────────────
// 广播标量操作数（全 0 strides：0-dim 或全 1 shape）不再走 offset_nd
// 双操作数路径——mp_22 上 64 位 div/mod 为软件仿真，实测标量广播比
// contiguous 慢 2-4×（mul f32×标量 96 vs add f32 447 GB/s，2026-08-07
// 基准）。标量读入寄存器一次（同地址广播读，L2 命中），另一操作数
// 连续访问。只启用「标量 + C-contiguous」组合；其余仍走 nd 路径。
// AEXPR/BEXPR 为 c[i] 表达式（分别对应 a 标量 / b 标量），
// 可用变量：av/bv（标量值）、a[i]/b[i]（连续操作数）。

#define SCALAR_BINARY_KERNELS(OP, AEXPR, BEXPR)                                \
template <typename T>                                                          \
__global__ void musapy_##OP##_scalar_a_kernel(                                 \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,       \
    size_t n                                                                    \
) {                                                                            \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (i < n) {                                                               \
        T av = a[0];                                                           \
        c[i] = AEXPR;                                                          \
    }                                                                          \
}                                                                              \
template <typename T>                                                          \
__global__ void musapy_##OP##_scalar_b_kernel(                                 \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,       \
    size_t n                                                                    \
) {                                                                            \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (i < n) {                                                               \
        T bv = b[0];                                                           \
        c[i] = BEXPR;                                                          \
    }                                                                          \
}

SCALAR_BINARY_KERNELS(add, (av + b[i]), (a[i] + bv))
SCALAR_BINARY_KERNELS(sub, (av - b[i]), (a[i] - bv))
SCALAR_BINARY_KERNELS(mul, (av * b[i]), (a[i] * bv))
SCALAR_BINARY_KERNELS(div, (av / b[i]), (a[i] / bv))
SCALAR_BINARY_KERNELS(pow, pow(av, b[i]), pow(a[i], bv))

#undef SCALAR_BINARY_KERNELS

// ── 标量广播 fast-path（P1）——comparison 版（输出 uint8_t）──
// 与 SCALAR_BINARY_KERNELS 同因：bernoulli（rand < p）等标量比较走
// offset_nd 双操作数路径受 64 位 div/mod 仿真拖累。

#define SCALAR_COMPARE_KERNELS(OP, AEXPR, BEXPR)                              \
template <typename T>                                                          \
__global__ void musapy_##OP##_scalar_a_kernel(                                 \
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c, \
    size_t n                                                                    \
) {                                                                            \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (i < n) {                                                               \
        T av = a[0];                                                           \
        c[i] = (uint8_t)(AEXPR);                                               \
    }                                                                          \
}                                                                              \
template <typename T>                                                          \
__global__ void musapy_##OP##_scalar_b_kernel(                                 \
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c, \
    size_t n                                                                    \
) {                                                                            \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (i < n) {                                                               \
        T bv = b[0];                                                           \
        c[i] = (uint8_t)(BEXPR);                                               \
    }                                                                          \
}

SCALAR_COMPARE_KERNELS(eq, (av == b[i]), (a[i] == bv))
SCALAR_COMPARE_KERNELS(ne, (av != b[i]), (a[i] != bv))
SCALAR_COMPARE_KERNELS(lt, (av < b[i]), (a[i] < bv))
SCALAR_COMPARE_KERNELS(gt, (av > b[i]), (a[i] > bv))
SCALAR_COMPARE_KERNELS(le, (av <= b[i]), (a[i] <= bv))
SCALAR_COMPARE_KERNELS(ge, (av >= b[i]), (a[i] >= bv))

#undef SCALAR_COMPARE_KERNELS

// ── extern "C" 稳定 ABI ────────────────────────────────────────

extern "C" {

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
    if (is_scalar_strides(a_strides, ndim) &&                                \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_scalar_a_kernel<float><<<grid_size_1d(n), 256, 0,      \
            stream>>>(a, b, c, n);                                           \
    } else if (is_scalar_strides(b_strides, ndim) &&                         \
        is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_##OP##_scalar_b_kernel<float><<<grid_size_1d(n), 256, 0,      \
            stream>>>(a, b, c, n);                                           \
    } else if (is_contiguous_strides(shape, a_strides, ndim) &&              \
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
    if (is_scalar_strides(a_strides, ndim) &&                                \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_scalar_a_kernel<double><<<grid_size_1d(n), 256, 0,     \
            stream>>>(a, b, c, n);                                           \
    } else if (is_scalar_strides(b_strides, ndim) &&                         \
        is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_##OP##_scalar_b_kernel<double><<<grid_size_1d(n), 256, 0,     \
            stream>>>(a, b, c, n);                                           \
    } else if (is_contiguous_strides(shape, a_strides, ndim) &&              \
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
        musapy_##OP##_kernel_v2<double><<<grid_size_1d(n), 256, 0, stream>>>( \
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
    if (is_scalar_strides(a_strides, ndim) &&                                \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_scalar_a_kernel<float><<<grid_size_1d(n), 256, 0,      \
            stream>>>(a, b, c, n);                                           \
    } else if (is_scalar_strides(b_strides, ndim) &&                         \
        is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_##OP##_scalar_b_kernel<float><<<grid_size_1d(n), 256, 0,      \
            stream>>>(a, b, c, n);                                           \
    } else if (is_contiguous_strides(shape, a_strides, ndim) &&              \
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
    if (is_scalar_strides(a_strides, ndim) &&                                \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_scalar_a_kernel<double><<<grid_size_1d(n), 256, 0,     \
            stream>>>(a, b, c, n);                                           \
    } else if (is_scalar_strides(b_strides, ndim) &&                         \
        is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_##OP##_scalar_b_kernel<double><<<grid_size_1d(n), 256, 0,     \
            stream>>>(a, b, c, n);                                           \
    } else if (is_contiguous_strides(shape, a_strides, ndim) &&              \
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

// ════════════════════════════════════════════════════════════════
// ── complex 支持（v0.3 Phase 5，ADR-003 003-D5）────────────────
//
// complex64/128 的 elementwise 实例化：binary add/sub/mul/div +
// unary neg/abs + comparison eq/ne。语义规则（003-D5）：
//   - 逐元素按 re/im 分量公式计算（mcc 3.1.0 已实测支持 struct kernel，
//     2026-08-08 冒烟通过；不用 C++ 运算符重载，显式公式更稳）。
//   - abs(complex) 输出 **real**（NumPy：np.abs(complex) → float32/64）。
//   - eq/ne 支持 complex（re 与 im 全等才相等）；lt/gt/le/ge 对 complex
//     永久拒绝（complex 无全序）——由 Rust 侧白名单把关，本文件不实例化。
//   - 不启用 vec4/scalar fast-path（complex 为新面，收敛实现范围）。
//
// ABI：complex buffer 的 interleaved re/im 布局与 C 的
// `struct { T re; T im; }` 一一对应（muComplex/muDoubleComplex，
// musa_x_ffi.rs:65-78），wrapper 直接透传 buffer 指针，无打包/解包。

// ── complex 标量类型 ────────────────────────────────────────────
typedef struct c64 { float re; float im; } c64;
typedef struct c128 { double re; double im; } c128;

// ── complex binary kernel 模板（nd + flat）─────────────────────
// RE_EXPR / IM_EXPR 用局部变量 av/bv（避免 nd/flat 索引变量名差异）。

#define CPLX_BINARY_TEMPLATES(OP, RE_EXPR, IM_EXPR)                         \
template <typename T>                                                       \
__global__ void musapy_##OP##_cplx_kernel_v2(                               \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,    \
    NdMeta meta, size_t n                                                    \
) {                                                                         \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (idx < n) {                                                          \
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);\
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim);\
        T av = a[a_off];                                                    \
        T bv = b[b_off];                                                    \
        c[idx].re = (RE_EXPR);                                              \
        c[idx].im = (IM_EXPR);                                              \
    }                                                                       \
}                                                                           \
template <typename T>                                                       \
__global__ void musapy_##OP##_cplx_flat_v2(                                 \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,    \
    size_t n                                                                 \
) {                                                                         \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (idx < n) {                                                          \
        T av = a[idx];                                                      \
        T bv = b[idx];                                                      \
        c[idx].re = (RE_EXPR);                                              \
        c[idx].im = (IM_EXPR);                                              \
    }                                                                       \
}

CPLX_BINARY_TEMPLATES(add, (av.re + bv.re), (av.im + bv.im))
CPLX_BINARY_TEMPLATES(sub, (av.re - bv.re), (av.im - bv.im))
CPLX_BINARY_TEMPLATES(mul,
    (av.re * bv.re - av.im * bv.im),
    (av.re * bv.im + av.im * bv.re))
CPLX_BINARY_TEMPLATES(div,
    ((av.re * bv.re + av.im * bv.im) / (bv.re * bv.re + bv.im * bv.im)),
    ((av.im * bv.re - av.re * bv.im) / (bv.re * bv.re + bv.im * bv.im)))

#undef CPLX_BINARY_TEMPLATES

// ── complex unary：neg（T 泛型，输出同 complex）────────────────

template <typename T>
__global__ void musapy_neg_cplx_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx].re = -a[a_off].re;
        c[idx].im = -a[a_off].im;
    }
}

template <typename T>
__global__ void musapy_neg_cplx_flat_v2(
    const T* __restrict__ a, T* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx].re = -a[idx].re;
        c[idx].im = -a[idx].im;
    }
}

// ── complex unary：abs（输出 real，c64→float / c128→double）────

#define CPLX_ABS_KERNEL(CT, RT, SQRTFN)                                       \
__global__ void musapy_abs_cplx_kernel_v2_##CT(                               \
    const CT* __restrict__ a, RT* __restrict__ c, NdMetaUnary meta, size_t n  \
) {                                                                           \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < n) {                                                            \
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim); \
        RT re = (RT)a[a_off].re;                                              \
        RT im = (RT)a[a_off].im;                                              \
        c[idx] = SQRTFN(re * re + im * im);                                   \
    }                                                                         \
}                                                                             \
__global__ void musapy_abs_cplx_flat_v2_##CT(                                 \
    const CT* __restrict__ a, RT* __restrict__ c, size_t n                    \
) {                                                                           \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < n) {                                                            \
        RT re = (RT)a[idx].re;                                                \
        RT im = (RT)a[idx].im;                                                \
        c[idx] = SQRTFN(re * re + im * im);                                   \
    }                                                                         \
}

CPLX_ABS_KERNEL(c64, float, sqrtf)
CPLX_ABS_KERNEL(c128, double, sqrt)

#undef CPLX_ABS_KERNEL

// ── complex comparison eq/ne（输出 uint8_t；re 与 im 全等才相等）─

#define CPLX_COMPARE_TEMPLATES(OP, EXPR)                                      \
template <typename T>                                                         \
__global__ void musapy_##OP##_cplx_kernel_v2(                                 \
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c,\
    NdMeta meta, size_t n                                                      \
) {                                                                           \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < n) {                                                            \
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim); \
        size_t b_off = offset_nd(idx, meta.shape, meta.b_strides, meta.ndim); \
        T av = a[a_off];                                                      \
        T bv = b[b_off];                                                      \
        c[idx] = (uint8_t)(EXPR);                                             \
    }                                                                         \
}                                                                             \
template <typename T>                                                         \
__global__ void musapy_##OP##_cplx_flat_v2(                                   \
    const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ c,\
    size_t n                                                                   \
) {                                                                           \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < n) {                                                            \
        T av = a[idx];                                                        \
        T bv = b[idx];                                                        \
        c[idx] = (uint8_t)(EXPR);                                             \
    }                                                                         \
}

CPLX_COMPARE_TEMPLATES(eq, (av.re == bv.re && av.im == bv.im))
CPLX_COMPARE_TEMPLATES(ne, (av.re != bv.re || av.im != bv.im))

#undef CPLX_COMPARE_TEMPLATES

// ── extern "C" complex 符号 ────────────────────────────────────
// wrapper：contiguous → flat fast-path；否则 → stride-aware nd。
// 无 scalar/vec4 fast-path（收敛实现面，complex 场景低频）。

extern "C" {

// binary wrapper（c64/c128 各一份）
#define CPLX_BINARY_WRAPPER(OP, CT)                                          \
void musapy_##OP##_##CT##_v2(                                                \
    const CT* __restrict__ a, const CT* __restrict__ b,                     \
    CT* __restrict__ c, int ndim, const size_t* shape,                      \
    const ssize_t* a_strides, const ssize_t* b_strides, musaStream_t stream  \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                          \
    if (is_contiguous_strides(shape, a_strides, ndim) &&                     \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_cplx_flat_v2<CT><<<grid_size_1d(n), 256, 0,             \
            stream>>>(a, b, c, n);                                           \
    } else {                                                                 \
        NdMeta meta;                                                         \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
            meta.b_strides[i] = b_strides[i];                                \
        }                                                                    \
        musapy_##OP##_cplx_kernel_v2<CT><<<grid_size_1d(n), 256, 0,           \
            stream>>>(a, b, c, meta, n);                                     \
    }                                                                        \
}

#define CPLX_BINARY_V2(OP) \
    CPLX_BINARY_WRAPPER(OP, c64) \
    CPLX_BINARY_WRAPPER(OP, c128)

CPLX_BINARY_V2(add)
CPLX_BINARY_V2(sub)
CPLX_BINARY_V2(mul)
CPLX_BINARY_V2(div)

#undef CPLX_BINARY_WRAPPER
#undef CPLX_BINARY_V2

// neg wrapper（c64/c128 各一份，输出同 complex）
#define CPLX_NEG_WRAPPER(CT)                                                  \
void musapy_neg_##CT##_v2(                                                    \
    const CT* __restrict__ a, CT* __restrict__ c,                            \
    int ndim, const size_t* shape, const ssize_t* a_strides,                  \
    musaStream_t stream                                                       \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                          \
    if (is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_neg_cplx_flat_v2<CT><<<grid_size_1d(n), 256, 0, stream>>>(     \
            a, c, n);                                                        \
    } else {                                                                 \
        NdMetaUnary meta;                                                    \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
        }                                                                    \
        musapy_neg_cplx_kernel_v2<CT><<<grid_size_1d(n), 256, 0, stream>>>(   \
            a, c, meta, n);                                                  \
    }                                                                        \
}

CPLX_NEG_WRAPPER(c64)
CPLX_NEG_WRAPPER(c128)

#undef CPLX_NEG_WRAPPER

// abs wrapper（输出 real：c64→float / c128→double）
#define CPLX_ABS_WRAPPER(CT, RT)                                              \
void musapy_abs_##CT##_v2(                                                    \
    const CT* __restrict__ a, RT* __restrict__ c,                            \
    int ndim, const size_t* shape, const ssize_t* a_strides,                  \
    musaStream_t stream                                                       \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                          \
    if (is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_abs_cplx_flat_v2_##CT<<<grid_size_1d(n), 256, 0, stream>>>(    \
            a, c, n);                                                        \
    } else {                                                                 \
        NdMetaUnary meta;                                                    \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
        }                                                                    \
        musapy_abs_cplx_kernel_v2_##CT<<<grid_size_1d(n), 256, 0, stream>>>(  \
            a, c, meta, n);                                                  \
    }                                                                        \
}

CPLX_ABS_WRAPPER(c64, float)
CPLX_ABS_WRAPPER(c128, double)

#undef CPLX_ABS_WRAPPER

// comparison wrapper（c64/c128 各一份，输出 uint8_t）
#define CPLX_COMPARE_WRAPPER(OP, CT)                                         \
void musapy_##OP##_##CT##_v2(                                                \
    const CT* __restrict__ a, const CT* __restrict__ b,                     \
    uint8_t* __restrict__ c, int ndim, const size_t* shape,                 \
    const ssize_t* a_strides, const ssize_t* b_strides, musaStream_t stream  \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                          \
    if (is_contiguous_strides(shape, a_strides, ndim) &&                     \
        is_contiguous_strides(shape, b_strides, ndim)) {                     \
        musapy_##OP##_cplx_flat_v2<CT><<<grid_size_1d(n), 256, 0,             \
            stream>>>(a, b, c, n);                                           \
    } else {                                                                 \
        NdMeta meta;                                                         \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
            meta.b_strides[i] = b_strides[i];                                \
        }                                                                    \
        musapy_##OP##_cplx_kernel_v2<CT><<<grid_size_1d(n), 256, 0,           \
            stream>>>(a, b, c, meta, n);                                     \
    }                                                                        \
}

CPLX_COMPARE_WRAPPER(eq, c64)
CPLX_COMPARE_WRAPPER(eq, c128)
CPLX_COMPARE_WRAPPER(ne, c64)
CPLX_COMPARE_WRAPPER(ne, c128)

#undef CPLX_COMPARE_WRAPPER

// ── complex cast（real → complex，Phase 5）────────────────────
// 4 对：f32→c64 / f32→c128 / f64→c64 / f64→c128（re=src, im=0）。
// fft real 输入扩展 + 混合运算类型提升（real + complex → 宽 complex）共用。

#define CPLX_CAST_KERNELS(SRC_T, CT, SUFFIX)                                  \
__global__ void musapy_cast_##SUFFIX##_cplx_kernel_v2(                        \
    const SRC_T* __restrict__ a, CT* __restrict__ c, NdMetaUnary meta,        \
    size_t n                                                                    \
) {                                                                           \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < n) {                                                            \
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim); \
        c[idx].re = a[a_off];                                                 \
        c[idx].im = (SRC_T)0;                                                 \
    }                                                                         \
}                                                                             \
__global__ void musapy_cast_##SUFFIX##_cplx_flat_v2(                          \
    const SRC_T* __restrict__ a, CT* __restrict__ c, size_t n                 \
) {                                                                           \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < n) {                                                            \
        c[idx].re = a[idx];                                                   \
        c[idx].im = (SRC_T)0;                                                 \
    }                                                                         \
}

CPLX_CAST_KERNELS(float, c64, f32_c64)
CPLX_CAST_KERNELS(float, c128, f32_c128)
CPLX_CAST_KERNELS(double, c64, f64_c64)
CPLX_CAST_KERNELS(double, c128, f64_c128)

#undef CPLX_CAST_KERNELS

// ── complex 宽度提升（c64 → c128，Phase 5）────────────────────
// 跨类别提升（f64+c64→c128 等）时窄 complex 需扩宽；re/im 各 f32→f64，无精度损失。

__global__ void musapy_cast_c64_c128_cplx_kernel_v2(
    const c64* __restrict__ a, c128* __restrict__ c, NdMetaUnary meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t a_off = offset_nd(idx, meta.shape, meta.a_strides, meta.ndim);
        c[idx].re = (double)a[a_off].re;
        c[idx].im = (double)a[a_off].im;
    }
}

__global__ void musapy_cast_c64_c128_cplx_flat_v2(
    const c64* __restrict__ a, c128* __restrict__ c, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx].re = (double)a[idx].re;
        c[idx].im = (double)a[idx].im;
    }
}

// ── real → complex cast wrapper（f32/f64 → c64/c128）──────────
#define CPLX_CAST_WRAPPER(SRC_T, CT, SUFFIX)                                  \
void musapy_cast_##SUFFIX##_v2(                                               \
    const SRC_T* __restrict__ a, CT* __restrict__ c,                         \
    int ndim, const size_t* shape, const ssize_t* a_strides,                  \
    musaStream_t stream                                                       \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                          \
    if (is_contiguous_strides(shape, a_strides, ndim)) {                     \
        musapy_cast_##SUFFIX##_cplx_flat_v2<<<grid_size_1d(n), 256, 0,        \
            stream>>>(a, c, n);                                              \
    } else {                                                                 \
        NdMetaUnary meta;                                                    \
        meta.ndim = ndim;                                                    \
        for (int i = 0; i < ndim; i++) {                                    \
            meta.shape[i] = shape[i];                                        \
            meta.a_strides[i] = a_strides[i];                                \
        }                                                                    \
        musapy_cast_##SUFFIX##_cplx_kernel_v2<<<grid_size_1d(n), 256, 0,      \
            stream>>>(a, c, meta, n);                                        \
    }                                                                        \
}

// 符号名对齐既有惯例：musapy_cast_<src>_<dst>_v2
CPLX_CAST_WRAPPER(float, c64, f32_c64)
CPLX_CAST_WRAPPER(float, c128, f32_c128)
CPLX_CAST_WRAPPER(double, c64, f64_c64)
CPLX_CAST_WRAPPER(double, c128, f64_c128)

#undef CPLX_CAST_WRAPPER

// c64→c128 wrapper（complex 宽度提升）
void musapy_cast_c64_c128_v2(
    const c64* __restrict__ a, c128* __restrict__ c,
    int ndim, const size_t* shape, const ssize_t* a_strides,
    musaStream_t stream
) {
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    if (is_contiguous_strides(shape, a_strides, ndim)) {
        musapy_cast_c64_c128_cplx_flat_v2<<<grid_size_1d(n), 256, 0, stream>>>(
            a, c, n);
    } else {
        NdMetaUnary meta;
        meta.ndim = ndim;
        for (int i = 0; i < ndim; i++) {
            meta.shape[i] = shape[i];
            meta.a_strides[i] = a_strides[i];
        }
        musapy_cast_c64_c128_cplx_kernel_v2<<<grid_size_1d(n), 256, 0, stream>>>(
            a, c, meta, n);
    }
}

// ── complex resize（截断/补零，Phase 5 fft 的 n 参数）──────────
// 输入 complex 数组 shape=[..., n_in]（stride-aware），输出连续
// shape=[..., n_out]；k < n_in 拷贝，否则补零。逐输出元素。
#define CPLX_RESIZE_KERNEL(CT, ZERO_T)                                        \
__global__ void musapy_resize_##CT##_kernel_v2(                               \
    const CT* __restrict__ a, CT* __restrict__ c,                             \
    NdMetaUnary meta, size_t n_in, size_t n_out, size_t outer                  \
) {                                                                           \
    size_t total = outer * n_out;                                             \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < total) {                                                        \
        size_t outer_idx = idx / n_out;                                       \
        size_t k = idx % n_out;                                               \
        if (k < n_in) {                                                       \
            size_t in_linear = outer_idx * n_in + k;                          \
            size_t a_off = offset_nd(in_linear, meta.shape, meta.a_strides,   \
                meta.ndim);                                                   \
            c[idx] = a[a_off];                                                \
        } else {                                                              \
            c[idx].re = (ZERO_T)0;                                            \
            c[idx].im = (ZERO_T)0;                                            \
        }                                                                     \
    }                                                                         \
}

CPLX_RESIZE_KERNEL(c64, float)
CPLX_RESIZE_KERNEL(c128, double)

#undef CPLX_RESIZE_KERNEL

// ── real resize（截断/补零，Phase 5 rfft 的 n 参数）────────────
// 与 complex resize 同构，但输入输出为 real（R2C/D2Z 的输入必须是 real buffer）。

#define REAL_RESIZE_KERNEL(RT)                                                \
__global__ void musapy_resize_##RT##_real_kernel_v2(                          \
    const RT* __restrict__ a, RT* __restrict__ c,                             \
    NdMetaUnary meta, size_t n_in, size_t n_out, size_t outer                  \
) {                                                                           \
    size_t total = outer * n_out;                                             \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < total) {                                                        \
        size_t outer_idx = idx / n_out;                                       \
        size_t k = idx % n_out;                                               \
        if (k < n_in) {                                                       \
            size_t in_linear = outer_idx * n_in + k;                          \
            size_t a_off = offset_nd(in_linear, meta.shape, meta.a_strides,   \
                meta.ndim);                                                   \
            c[idx] = a[a_off];                                                \
        } else {                                                              \
            c[idx] = (RT)0;                                                   \
        }                                                                     \
    }                                                                         \
}

REAL_RESIZE_KERNEL(float)
REAL_RESIZE_KERNEL(double)

#undef REAL_RESIZE_KERNEL

// ── real → complex cast + resize 合并（P-FFT-2，2026-08-08）──
// fft/ifft 的 real 输入：一次 kernel 完成「扩 complex（re=x, im=0）+ 截断/补零」，
// 省去原先 cast_array（[.,n_in]）与 resize（[.,n_in]→[.,n_out]）两次传递/launch。
// 读 real stride-aware shape=[..., n_in]，写 complex 连续 [..., n_out]。

#define CPLX_CAST_RESIZE_KERNELS(SRC_T, CT, SUFFIX)                           \
__global__ void musapy_cast_resize_##SUFFIX##_cplx_kernel_v2(                 \
    const SRC_T* __restrict__ a, CT* __restrict__ c, NdMetaUnary meta,        \
    size_t n_in, size_t n_out, size_t outer                                    \
) {                                                                           \
    size_t total = outer * n_out;                                             \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < total) {                                                        \
        size_t outer_idx = idx / n_out;                                       \
        size_t k = idx % n_out;                                               \
        if (k < n_in) {                                                       \
            size_t in_linear = outer_idx * n_in + k;                          \
            size_t a_off = offset_nd(in_linear, meta.shape, meta.a_strides,   \
                meta.ndim);                                                   \
            c[idx].re = a[a_off];                                             \
            c[idx].im = (SRC_T)0;                                             \
        } else {                                                              \
            c[idx].re = (SRC_T)0;                                             \
            c[idx].im = (SRC_T)0;                                             \
        }                                                                     \
    }                                                                         \
}

CPLX_CAST_RESIZE_KERNELS(float, c64, f32_c64)
CPLX_CAST_RESIZE_KERNELS(double, c128, f64_c128)

#undef CPLX_CAST_RESIZE_KERNELS

// ── complex 就地缩放（real 标量，Phase 5 fft 归一化）───────────
// 输出 buffer 恒连续（fft 骨架保证），无需 stride。
#define CPLX_SCALE_KERNEL(CT)                                                 \
__global__ void musapy_scale_##CT##_kernel_v2(                                \
    CT* __restrict__ c, double factor, size_t n                               \
) {                                                                           \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (idx < n) {                                                            \
        c[idx].re *= (float)factor;                                           \
        c[idx].im *= (float)factor;                                           \
    }                                                                         \
}

CPLX_SCALE_KERNEL(c64)
CPLX_SCALE_KERNEL(c128)

#undef CPLX_SCALE_KERNEL

// ── extern "C"：resize / scale ────────────────────────────────

extern "C" {

#define CPLX_RESIZE_WRAPPER(CT)                                               \
void musapy_resize_##CT##_v2(                                                 \
    const CT* __restrict__ a, CT* __restrict__ c,                             \
    int ndim, const size_t* shape, const ssize_t* a_strides,                  \
    size_t n_in, size_t n_out, musaStream_t stream                             \
) {                                                                           \
    size_t outer = 1;                                                         \
    for (int i = 0; i < ndim - 1; i++) outer *= shape[i];                    \
    NdMetaUnary meta;                                                         \
    meta.ndim = ndim;                                                         \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.shape[i] = shape[i];                                             \
        meta.a_strides[i] = a_strides[i];                                     \
    }                                                                         \
    musapy_resize_##CT##_kernel_v2<<<grid_size_1d(outer * n_out), 256, 0,     \
        stream>>>(a, c, meta, n_in, n_out, outer);                            \
}

CPLX_RESIZE_WRAPPER(c64)
CPLX_RESIZE_WRAPPER(c128)

#undef CPLX_RESIZE_WRAPPER

// real resize wrapper（f32/f64，rfft n 参数用）
#define REAL_RESIZE_WRAPPER(RT, SUFFIX)                                       \
void musapy_resize_##SUFFIX##_real_v2(                                        \
    const RT* __restrict__ a, RT* __restrict__ c,                             \
    int ndim, const size_t* shape, const ssize_t* a_strides,                  \
    size_t n_in, size_t n_out, musaStream_t stream                             \
) {                                                                           \
    size_t outer = 1;                                                         \
    for (int i = 0; i < ndim - 1; i++) outer *= shape[i];                    \
    NdMetaUnary meta;                                                         \
    meta.ndim = ndim;                                                         \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.shape[i] = shape[i];                                             \
        meta.a_strides[i] = a_strides[i];                                     \
    }                                                                         \
    musapy_resize_##RT##_real_kernel_v2<<<grid_size_1d(outer * n_out), 256,   \
        0, stream>>>(a, c, meta, n_in, n_out, outer);                         \
}

REAL_RESIZE_WRAPPER(float, f32)
REAL_RESIZE_WRAPPER(double, f64)

#undef REAL_RESIZE_WRAPPER

// cast+resize wrapper（P-FFT-2：real→complex 扩 + 截断/补零一步）
#define CPLX_CAST_RESIZE_WRAPPER(SRC_T, CT, SUFFIX)                           \
void musapy_cast_resize_##SUFFIX##_v2(                                        \
    const SRC_T* __restrict__ a, CT* __restrict__ c,                          \
    int ndim, const size_t* shape, const ssize_t* a_strides,                  \
    size_t n_in, size_t n_out, musaStream_t stream                             \
) {                                                                           \
    size_t outer = 1;                                                         \
    for (int i = 0; i < ndim - 1; i++) outer *= shape[i];                    \
    NdMetaUnary meta;                                                         \
    meta.ndim = ndim;                                                         \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.shape[i] = shape[i];                                             \
        meta.a_strides[i] = a_strides[i];                                     \
    }                                                                         \
    musapy_cast_resize_##SUFFIX##_cplx_kernel_v2<<<                           \
        grid_size_1d(outer * n_out), 256, 0, stream>>>(                       \
        a, c, meta, n_in, n_out, outer);                                      \
}

CPLX_CAST_RESIZE_WRAPPER(float, c64, f32_c64)
CPLX_CAST_RESIZE_WRAPPER(double, c128, f64_c128)

#undef CPLX_CAST_RESIZE_WRAPPER

// scale wrapper（输出 buffer 恒连续；c64/c128 为 .mu 内 typedef，
// ABI ≡ musa_x_ffi.rs 的 muComplex/muDoubleComplex）
void musapy_scale_c64_v2(
    c64* __restrict__ c, double factor, size_t n, musaStream_t stream
) {
    musapy_scale_c64_kernel_v2<<<grid_size_1d(n), 256, 0, stream>>>(c, factor, n);
}

void musapy_scale_c128_v2(
    c128* __restrict__ c, double factor, size_t n, musaStream_t stream
) {
    musapy_scale_c128_kernel_v2<<<grid_size_1d(n), 256, 0, stream>>>(c, factor, n);
}

} // extern "C" (resize/scale)

} // extern "C" (complex)
