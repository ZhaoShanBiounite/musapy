// reduction.mu — 缩减算子（ADR-002-D3）
//
// 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回。
// 策略：one-thread-per-output-element，每线程循环 axis_len 次累加。
// ABI 版本：_v2（stride-aware N-dimensional）
//
// 符号命名：musapy_<op>_<dtype>_v2
// 类型变体：i64 / f32 / f64（与 elementwise 保持一致）

#include "include/common.h"
#include <math.h>
#include <stdint.h>
#include <limits.h>

// ── Reduction 参数结构（按值传递给 kernel）────────────────────

#define MUSAPY_MAX_NDIM 32

/// Reduction kernel 参数（按值传递，避免 host 指针问题）。
struct NdMetaReduce {
    int ndim;
    size_t in_shape[MUSAPY_MAX_NDIM];
    ssize_t in_strides[MUSAPY_MAX_NDIM];
    int axis;
    size_t axis_len;
};

/// 在 kernel 内计算 reduction input offset（使用 NdMetaReduce）。
__device__ static inline size_t reduce_offset(
    size_t out_idx, const NdMetaReduce& meta, size_t k
) {
    int ndim = meta.ndim;
    int axis = meta.axis;
    // 展开 out_idx 到非 axis 维坐标
    size_t coords[MUSAPY_MAX_NDIM];
    int ci = 0;
    size_t tmp = out_idx;
    for (int i = ndim - 1; i >= 0; i--) {
        if (i == axis) continue;
        coords[ci++] = tmp % meta.in_shape[i];
        tmp /= meta.in_shape[i];
    }
    // 计算 offset
    size_t offset = 0;
    ci = 0;
    for (int i = ndim - 1; i >= 0; i--) {
        size_t coord = (i == axis) ? k : coords[ci++];
        offset += coord * (size_t)meta.in_strides[i];
    }
    return offset;
}

// ── Reduction kernel 模板 ────────────────────────────────────

template <typename T>
__global__ void musapy_sum_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c,
    NdMetaReduce meta, size_t out_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    size_t base = reduce_offset(idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    T acc = (T)0;
    for (size_t k = 0; k < meta.axis_len; k++) {
        acc += a[base + (size_t)((ssize_t)k * axis_stride)];
    }
    c[idx] = acc;
}

template <typename T>
__global__ void musapy_prod_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c,
    NdMetaReduce meta, size_t out_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    size_t base = reduce_offset(idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    T acc = (T)1;
    for (size_t k = 0; k < meta.axis_len; k++) {
        acc *= a[base + (size_t)((ssize_t)k * axis_stride)];
    }
    c[idx] = acc;
}

template <typename T>
__global__ void musapy_max_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c,
    NdMetaReduce meta, size_t out_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    size_t base = reduce_offset(idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    T acc = a[base];
    for (size_t k = 1; k < meta.axis_len; k++) {
        T val = a[base + (size_t)((ssize_t)k * axis_stride)];
        if (val > acc) acc = val;
    }
    c[idx] = acc;
}

template <typename T>
__global__ void musapy_min_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c,
    NdMetaReduce meta, size_t out_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    size_t base = reduce_offset(idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    T acc = a[base];
    for (size_t k = 1; k < meta.axis_len; k++) {
        T val = a[base + (size_t)((ssize_t)k * axis_stride)];
        if (val < acc) acc = val;
    }
    c[idx] = acc;
}

template <typename T>
__global__ void musapy_mean_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c,
    NdMetaReduce meta, size_t out_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    size_t base = reduce_offset(idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    T acc = (T)0;
    for (size_t k = 0; k < meta.axis_len; k++) {
        acc += a[base + (size_t)((ssize_t)k * axis_stride)];
    }
    c[idx] = acc / (T)meta.axis_len;
}

// ── Argmax / Argmin kernel 模板（输入 T，输出 int64_t）────────

template <typename T>
__global__ void musapy_argmax_kernel_v2(
    const T* __restrict__ a, int64_t* __restrict__ c,
    NdMetaReduce meta, size_t out_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    size_t base = reduce_offset(idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    T best_val = a[base];
    int64_t best_idx = 0;
    for (size_t k = 1; k < meta.axis_len; k++) {
        T val = a[base + (size_t)((ssize_t)k * axis_stride)];
        if (val > best_val) {
            best_val = val;
            best_idx = (int64_t)k;
        }
    }
    c[idx] = best_idx;
}

template <typename T>
__global__ void musapy_argmin_kernel_v2(
    const T* __restrict__ a, int64_t* __restrict__ c,
    NdMetaReduce meta, size_t out_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    size_t base = reduce_offset(idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    T best_val = a[base];
    int64_t best_idx = 0;
    for (size_t k = 1; k < meta.axis_len; k++) {
        T val = a[base + (size_t)((ssize_t)k * axis_stride)];
        if (val < best_val) {
            best_val = val;
            best_idx = (int64_t)k;
        }
    }
    c[idx] = best_idx;
}

// ── Cumsum kernel 模板（输出同 shape，prefix sum）─────────────

/// Cumsum 参数结构。
struct NdMetaCumsum {
    int ndim;
    size_t in_shape[MUSAPY_MAX_NDIM];
    ssize_t in_strides[MUSAPY_MAX_NDIM];
    int axis;
};

template <typename T>
__global__ void musapy_cumsum_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c,
    NdMetaCumsum meta, size_t out_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int ndim = meta.ndim;
    int axis = meta.axis;

    // 展开 idx 得到 axis 坐标
    size_t tmp = idx;
    size_t axis_coord = 0;
    for (int i = ndim - 1; i >= 0; i--) {
        size_t coord = tmp % meta.in_shape[i];
        tmp /= meta.in_shape[i];
        if (i == axis) axis_coord = coord;
    }

    // 计算 axis=0 时的 base offset
    size_t base = 0;
    tmp = idx;
    for (int i = ndim - 1; i >= 0; i--) {
        size_t coord = tmp % meta.in_shape[i];
        tmp /= meta.in_shape[i];
        if (i != axis) {
            base += coord * (size_t)meta.in_strides[i];
        }
    }

    ssize_t axis_stride = meta.in_strides[axis];
    T acc = (T)0;
    for (size_t k = 0; k <= axis_coord; k++) {
        acc += a[base + (size_t)((ssize_t)k * axis_stride)];
    }
    c[idx] = acc;
}

// ── extern "C" wrapper ───────────────────────────────────────

extern "C" {

/// 标准 reduction wrapper（sum/prod/max/min）：输入 T，输出 T
#define REDUCE_V2(OP)                                                          \
void musapy_##OP##_i64_v2(                                                    \
    const int64_t* __restrict__ a, int64_t* __restrict__ c,                   \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, musaStream_t stream           \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    meta.axis_len = axis_len;                                                 \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    musapy_##OP##_kernel_v2<int64_t>                                          \
        <<<grid_size_1d(out_size), 256, 0, stream>>>(a, c, meta, out_size);   \
}                                                                             \
void musapy_##OP##_f32_v2(                                                    \
    const float* __restrict__ a, float* __restrict__ c,                       \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, musaStream_t stream           \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    meta.axis_len = axis_len;                                                 \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    musapy_##OP##_kernel_v2<float>                                            \
        <<<grid_size_1d(out_size), 256, 0, stream>>>(a, c, meta, out_size);   \
}                                                                             \
void musapy_##OP##_f64_v2(                                                    \
    const double* __restrict__ a, double* __restrict__ c,                     \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, musaStream_t stream           \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    meta.axis_len = axis_len;                                                 \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    musapy_##OP##_kernel_v2<double>                                           \
        <<<grid_size_1d(out_size), 256, 0, stream>>>(a, c, meta, out_size);   \
}

REDUCE_V2(sum)
REDUCE_V2(prod)
REDUCE_V2(max)
REDUCE_V2(min)
#undef REDUCE_V2

/// mean wrapper：只有 f32/f64（整数输入在 Rust 层先 cast 到 f64）
#define MEAN_V2(T, SUFFIX)                                                    \
void musapy_mean_##SUFFIX##_v2(                                               \
    const T* __restrict__ a, T* __restrict__ c,                               \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, musaStream_t stream           \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    meta.axis_len = axis_len;                                                 \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    musapy_mean_kernel_v2<T>                                                  \
        <<<grid_size_1d(out_size), 256, 0, stream>>>(a, c, meta, out_size);   \
}

MEAN_V2(float, f32)
MEAN_V2(double, f64)
#undef MEAN_V2

/// argmax/argmin wrapper：输入 T，输出 int64_t*
#define ARGREDUCE_V2(OP)                                                      \
void musapy_##OP##_i64_v2(                                                    \
    const int64_t* __restrict__ a, int64_t* __restrict__ c,                   \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, musaStream_t stream           \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    meta.axis_len = axis_len;                                                 \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    musapy_##OP##_kernel_v2<int64_t>                                          \
        <<<grid_size_1d(out_size), 256, 0, stream>>>(a, c, meta, out_size);   \
}                                                                             \
void musapy_##OP##_f32_v2(                                                    \
    const float* __restrict__ a, int64_t* __restrict__ c,                     \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, musaStream_t stream           \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    meta.axis_len = axis_len;                                                 \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    musapy_##OP##_kernel_v2<float>                                            \
        <<<grid_size_1d(out_size), 256, 0, stream>>>(a, c, meta, out_size);   \
}                                                                             \
void musapy_##OP##_f64_v2(                                                    \
    const double* __restrict__ a, int64_t* __restrict__ c,                    \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, musaStream_t stream           \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    meta.axis_len = axis_len;                                                 \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    musapy_##OP##_kernel_v2<double>                                           \
        <<<grid_size_1d(out_size), 256, 0, stream>>>(a, c, meta, out_size);   \
}

ARGREDUCE_V2(argmax)
ARGREDUCE_V2(argmin)
#undef ARGREDUCE_V2

/// cumsum wrapper：输入 T，输出 T（同 shape）
#define CUMSUM_V2(T, SUFFIX)                                                  \
void musapy_cumsum_##SUFFIX##_v2(                                             \
    const T* __restrict__ a, T* __restrict__ c,                               \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t out_size, musaStream_t stream                            \
) {                                                                           \
    NdMetaCumsum meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    musapy_cumsum_kernel_v2<T>                                                \
        <<<grid_size_1d(out_size), 256, 0, stream>>>(a, c, meta, out_size);   \
}

CUMSUM_V2(int64_t, i64)
CUMSUM_V2(float, f32)
CUMSUM_V2(double, f64)
#undef CUMSUM_V2

} // extern "C"
