// indexing.mu — 索引算子 GPU kernel（Phase 6, ADR 002-D4）
//
// gather/scatter：按 axis + indices 取/写元素（copy 语义，分配新 buffer）。
// copy：stride-aware identity（视图物化为连续布局）。
//
// 约定（与 elementwise.mu 一致）：
// - 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回
// - 输入指针由 ops 层按 layout.offset 预调整（common.h offset_nd 的
//   无符号回绕语义要求基指针指向逻辑首元素）
// - indices 固定 int64；越界由 ops 层校验
//
// ABI 版本嵌入符号名：musapy_<op>_<dtype>（Phase 6 首版）

#include "include/common.h"

#define MUSAPY_MAX_NDIM 32

// ── 元数据结构 ──────────────────────────────────────────────

/// gather 参数：output shape（axis 维 = n_indices）+ input strides。
struct GatherMeta {
    int ndim;
    int axis;
    size_t out_shape[MUSAPY_MAX_NDIM];
    ssize_t in_strides[MUSAPY_MAX_NDIM];
};

/// scatter 参数：values shape（axis 维 = n_indices）+ values strides +
/// output 各维的连续 stride（row-major）。output 为连续布局。
struct ScatterMeta {
    int ndim;
    int axis;
    size_t val_shape[MUSAPY_MAX_NDIM];
    ssize_t val_strides[MUSAPY_MAX_NDIM];
    size_t out_strides[MUSAPY_MAX_NDIM];
};

/// copy 参数（stride-aware identity，与 NdMetaUnary 同构）。
struct CopyMeta {
    int ndim;
    size_t shape[MUSAPY_MAX_NDIM];
    ssize_t in_strides[MUSAPY_MAX_NDIM];
};

// ── Kernels ─────────────────────────────────────────────────

/// gather：out[idx] = input[offset]，其中 axis 维坐标取自 indices。
template <typename T>
__global__ void musapy_gather_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    const int64_t* __restrict__ indices, GatherMeta meta, size_t n_out
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_out) {
        size_t tmp = idx;
        ssize_t off = 0;
        for (int i = meta.ndim - 1; i >= 0; i--) {
            size_t coord = tmp % meta.out_shape[i];
            tmp /= meta.out_shape[i];
            size_t k = (i == meta.axis) ? (size_t)indices[coord] : coord;
            off += (ssize_t)k * meta.in_strides[i];
        }
        output[idx] = input[(size_t)off];
    }
}

/// scatter：output[out_offset] = values[idx]，axis 维坐标经 indices 映射。
/// 每个线程处理一个 values 元素；重复 indices 的写序未定义（与 PyTorch 一致）。
template <typename T>
__global__ void musapy_scatter_kernel(
    T* __restrict__ output, const T* __restrict__ values,
    const int64_t* __restrict__ indices, ScatterMeta meta, size_t n_values
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_values) {
        size_t tmp = idx;
        size_t out_off = 0;
        ssize_t val_off = 0;
        for (int i = meta.ndim - 1; i >= 0; i--) {
            size_t coord = tmp % meta.val_shape[i];
            tmp /= meta.val_shape[i];
            val_off += (ssize_t)coord * meta.val_strides[i];
            size_t k = (i == meta.axis) ? (size_t)indices[coord] : coord;
            out_off += k * meta.out_strides[i];
        }
        output[out_off] = values[(size_t)val_off];
    }
}

/// copy：out[idx] = input[offset_nd(idx)]（视图物化为连续布局）。
template <typename T>
__global__ void musapy_copy_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    CopyMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t off = offset_nd(idx, meta.shape, meta.in_strides, meta.ndim);
        output[idx] = input[off];
    }
}

// ── extern "C" 稳定 ABI ────────────────────────────────────────

extern "C" {

// ── gather ──

#define GATHER_WRAPPER(T, SUFFIX)                                             \
void musapy_gather_##SUFFIX(                                                 \
    const T* __restrict__ input, T* __restrict__ output,                     \
    const int64_t* __restrict__ indices,                                     \
    int ndim, int axis, const size_t* out_shape, const ssize_t* in_strides,  \
    size_t n_out, musaStream_t stream                                        \
) {                                                                          \
    if (n_out == 0) return;                                                  \
    GatherMeta meta;                                                         \
    meta.ndim = ndim;                                                        \
    meta.axis = axis;                                                        \
    for (int i = 0; i < ndim; i++) {                                        \
        meta.out_shape[i] = out_shape[i];                                    \
        meta.in_strides[i] = in_strides[i];                                  \
    }                                                                        \
    musapy_gather_kernel<T><<<grid_size_1d(n_out), 256, 0, stream>>>(        \
        input, output, indices, meta, n_out);                                \
}

GATHER_WRAPPER(float, f32)
GATHER_WRAPPER(double, f64)
GATHER_WRAPPER(int32_t, i32)
GATHER_WRAPPER(int64_t, i64)

// ── scatter ──

#define SCATTER_WRAPPER(T, SUFFIX)                                            \
void musapy_scatter_##SUFFIX(                                                \
    T* __restrict__ output, const T* __restrict__ values,                    \
    const int64_t* __restrict__ indices,                                     \
    int ndim, int axis, const size_t* val_shape, const ssize_t* val_strides, \
    const size_t* out_strides, size_t n_values, musaStream_t stream          \
) {                                                                          \
    if (n_values == 0) return;                                               \
    ScatterMeta meta;                                                        \
    meta.ndim = ndim;                                                        \
    meta.axis = axis;                                                        \
    for (int i = 0; i < ndim; i++) {                                        \
        meta.val_shape[i] = val_shape[i];                                    \
        meta.val_strides[i] = val_strides[i];                                \
        meta.out_strides[i] = out_strides[i];                                \
    }                                                                        \
    musapy_scatter_kernel<T><<<grid_size_1d(n_values), 256, 0, stream>>>(    \
        output, values, indices, meta, n_values);                            \
}

SCATTER_WRAPPER(float, f32)
SCATTER_WRAPPER(double, f64)
SCATTER_WRAPPER(int32_t, i32)
SCATTER_WRAPPER(int64_t, i64)

// ── copy（stride-aware identity，视图物化）──

#define COPY_WRAPPER(T, SUFFIX)                                               \
void musapy_copy_##SUFFIX(                                                   \
    const T* __restrict__ input, T* __restrict__ output,                     \
    int ndim, const size_t* shape, const ssize_t* in_strides,                \
    musaStream_t stream                                                      \
) {                                                                          \
    size_t n = 1;                                                            \
    for (int i = 0; i < ndim; i++) n *= shape[i];                           \
    if (n == 0) return;                                                      \
    CopyMeta meta;                                                           \
    meta.ndim = ndim;                                                        \
    for (int i = 0; i < ndim; i++) {                                        \
        meta.shape[i] = shape[i];                                            \
        meta.in_strides[i] = in_strides[i];                                  \
    }                                                                        \
    musapy_copy_kernel<T><<<grid_size_1d(n), 256, 0, stream>>>(              \
        input, output, meta, n);                                             \
}

COPY_WRAPPER(float, f32)
COPY_WRAPPER(double, f64)
COPY_WRAPPER(int32_t, i32)
COPY_WRAPPER(int64_t, i64)

} // extern "C"
