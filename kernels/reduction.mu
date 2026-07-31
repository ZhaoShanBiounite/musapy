// reduction.mu — 缩减算子（ADR-002-D3）
//
// 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回。
// ABI 版本：_v2（stride-aware N-dimensional）
//
// 两代 kernel 共存：
//   naive（one-thread-per-output）：保留，用于 mock 和小 axis_len
//   parallel（block-cooperative）：axis_len > 阈值时使用，两阶段缩减
//
// 符号命名：musapy_<op>_<dtype>_v2（naive）
//           musapy_<op>_partial_<dtype>_v2（parallel phase 1）
//           musapy_<op>_final_<dtype>_v2（parallel phase 2）

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

// ── Naive kernel 模板（one-thread-per-output，保留兼容）────────

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

// ── Argmax / Argmin naive kernel（输入 T，输出 int64_t）────────

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

// ═══════════════════════════════════════════════════════════════
// Parallel block-cooperative reduction kernels
// ═══════════════════════════════════════════════════════════════
//
// Phase 1 (partial): 多 block 协作缩减一个 output 的 axis 维度。
//   grid = out_size * tiles_per_output blocks
//   每 block 256 threads 缩减 axis 的一段 → 写 1 个 partial
//
// Phase 2 (final): 缩减 partials → 最终输出。
//   grid = out_size blocks
//   每 block 256 threads 缩减 tiles_per_output 个 partials → 1 个输出
//
// 当 tiles_per_output == 1 时，Rust 侧直接将 partials 指向 output，
// 跳过 Phase 2。

// ── Phase 1: partial reduction kernels ────────────────────────

template <typename T>
__global__ void musapy_sum_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;

    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    size_t base = reduce_offset(out_idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t total_threads = tiles_per_output * blockDim.x;
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;

    T acc = (T)0;
    for (size_t k = global_tid; k < meta.axis_len; k += total_threads) {
        acc += a[base + (size_t)((ssize_t)k * axis_stride)];
    }

    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = sdata[0];
    }
}

template <typename T>
__global__ void musapy_prod_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;

    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    size_t base = reduce_offset(out_idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t total_threads = tiles_per_output * blockDim.x;
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;

    T acc = (T)1;
    for (size_t k = global_tid; k < meta.axis_len; k += total_threads) {
        acc *= a[base + (size_t)((ssize_t)k * axis_stride)];
    }

    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] *= sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = sdata[0];
    }
}

template <typename T>
__global__ void musapy_max_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;

    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    size_t base = reduce_offset(out_idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t total_threads = tiles_per_output * blockDim.x;
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;

    // 初始化为最小可能值
    T acc = a[base];  // 至少有一个元素
    for (size_t k = global_tid; k < meta.axis_len; k += total_threads) {
        T val = a[base + (size_t)((ssize_t)k * axis_stride)];
        if (val > acc) acc = val;
    }

    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = sdata[0];
    }
}

template <typename T>
__global__ void musapy_min_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;

    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    size_t base = reduce_offset(out_idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t total_threads = tiles_per_output * blockDim.x;
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;

    T acc = a[base];
    for (size_t k = global_tid; k < meta.axis_len; k += total_threads) {
        T val = a[base + (size_t)((ssize_t)k * axis_stride)];
        if (val < acc) acc = val;
    }

    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] < sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = sdata[0];
    }
}

// mean partial = sum partial（final 阶段除以 axis_len）
template <typename T>
__global__ void musapy_mean_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;

    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    size_t base = reduce_offset(out_idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t total_threads = tiles_per_output * blockDim.x;
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;

    T acc = (T)0;
    for (size_t k = global_tid; k < meta.axis_len; k += total_threads) {
        acc += a[base + (size_t)((ssize_t)k * axis_stride)];
    }

    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = sdata[0];
    }
}

// ── Phase 2: final reduction kernels ──────────────────────────

template <typename T>
__global__ void musapy_sum_final_kernel_v2(
    const T* __restrict__ partials, T* __restrict__ c,
    size_t num_partials, size_t out_size
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size) return;

    const T* src = partials + out_idx * num_partials;
    T acc = (T)0;
    for (size_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        acc += src[i];
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) c[out_idx] = sdata[0];
}

template <typename T>
__global__ void musapy_prod_final_kernel_v2(
    const T* __restrict__ partials, T* __restrict__ c,
    size_t num_partials, size_t out_size
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size) return;

    const T* src = partials + out_idx * num_partials;
    T acc = (T)1;
    for (size_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        acc *= src[i];
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] *= sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) c[out_idx] = sdata[0];
}

template <typename T>
__global__ void musapy_max_final_kernel_v2(
    const T* __restrict__ partials, T* __restrict__ c,
    size_t num_partials, size_t out_size
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size) return;

    const T* src = partials + out_idx * num_partials;
    T acc = src[0];
    for (size_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        if (src[i] > acc) acc = src[i];
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) c[out_idx] = sdata[0];
}

template <typename T>
__global__ void musapy_min_final_kernel_v2(
    const T* __restrict__ partials, T* __restrict__ c,
    size_t num_partials, size_t out_size
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size) return;

    const T* src = partials + out_idx * num_partials;
    T acc = src[0];
    for (size_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        if (src[i] < acc) acc = src[i];
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] < sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) c[out_idx] = sdata[0];
}

template <typename T>
__global__ void musapy_mean_final_kernel_v2(
    const T* __restrict__ partials, T* __restrict__ c,
    size_t num_partials, size_t out_size, size_t axis_len
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size) return;

    const T* src = partials + out_idx * num_partials;
    T acc = (T)0;
    for (size_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        acc += src[i];
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) c[out_idx] = sdata[0] / (T)axis_len;
}

// ── Argmax/Argmin parallel kernels ────────────────────────────
// Phase 1: 每 block 输出 (best_val, best_idx) 到 partials_val / partials_idx

template <typename T>
__global__ void musapy_argmax_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials_val,
    int64_t* __restrict__ partials_idx,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    extern __shared__ char smem[];
    T* sval = (T*)smem;
    int64_t* sidx = (int64_t*)(sval + blockDim.x);

    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    size_t base = reduce_offset(out_idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t total_threads = tiles_per_output * blockDim.x;
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;

    T best_val = a[base];
    int64_t best_idx = 0;
    for (size_t k = global_tid; k < meta.axis_len; k += total_threads) {
        T val = a[base + (size_t)((ssize_t)k * axis_stride)];
        if (val > best_val) {
            best_val = val;
            best_idx = (int64_t)k;
        }
    }

    sval[threadIdx.x] = best_val;
    sidx[threadIdx.x] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials_val[out_idx * tiles_per_output + tile_idx] = sval[0];
        partials_idx[out_idx * tiles_per_output + tile_idx] = sidx[0];
    }
}

template <typename T>
__global__ void musapy_argmin_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials_val,
    int64_t* __restrict__ partials_idx,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    extern __shared__ char smem[];
    T* sval = (T*)smem;
    int64_t* sidx = (int64_t*)(sval + blockDim.x);

    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    size_t base = reduce_offset(out_idx, meta, 0);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t total_threads = tiles_per_output * blockDim.x;
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;

    T best_val = a[base];
    int64_t best_idx = 0;
    for (size_t k = global_tid; k < meta.axis_len; k += total_threads) {
        T val = a[base + (size_t)((ssize_t)k * axis_stride)];
        if (val < best_val) {
            best_val = val;
            best_idx = (int64_t)k;
        }
    }

    sval[threadIdx.x] = best_val;
    sidx[threadIdx.x] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] < sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials_val[out_idx * tiles_per_output + tile_idx] = sval[0];
        partials_idx[out_idx * tiles_per_output + tile_idx] = sidx[0];
    }
}

// Phase 2: argmax/argmin final — 从 partials 中选最终 index
template <typename T>
__global__ void musapy_argmax_final_kernel_v2(
    const T* __restrict__ partials_val, const int64_t* __restrict__ partials_idx,
    int64_t* __restrict__ c, size_t num_partials, size_t out_size
) {
    extern __shared__ char smem[];
    T* sval = (T*)smem;
    int64_t* sidx = (int64_t*)(sval + blockDim.x);
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size) return;

    const T* vsrc = partials_val + out_idx * num_partials;
    const int64_t* isrc = partials_idx + out_idx * num_partials;

    T best_val = vsrc[0];
    int64_t best_idx = isrc[0];
    for (size_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        if (vsrc[i] > best_val) {
            best_val = vsrc[i];
            best_idx = isrc[i];
        }
    }
    sval[threadIdx.x] = best_val;
    sidx[threadIdx.x] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) c[out_idx] = sidx[0];
}

template <typename T>
__global__ void musapy_argmin_final_kernel_v2(
    const T* __restrict__ partials_val, const int64_t* __restrict__ partials_idx,
    int64_t* __restrict__ c, size_t num_partials, size_t out_size
) {
    extern __shared__ char smem[];
    T* sval = (T*)smem;
    int64_t* sidx = (int64_t*)(sval + blockDim.x);
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size) return;

    const T* vsrc = partials_val + out_idx * num_partials;
    const int64_t* isrc = partials_idx + out_idx * num_partials;

    T best_val = vsrc[0];
    int64_t best_idx = isrc[0];
    for (size_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        if (vsrc[i] < best_val) {
            best_val = vsrc[i];
            best_idx = isrc[i];
        }
    }
    sval[threadIdx.x] = best_val;
    sidx[threadIdx.x] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] < sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) c[out_idx] = sidx[0];
}

// ═══════════════════════════════════════════════════════════════
// extern "C" wrappers
// ═══════════════════════════════════════════════════════════════

extern "C" {

// ── Naive wrappers（保留，用于 mock 和小 axis_len）──

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

/// mean wrapper：只有 f32/f64
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

/// argmax/argmin naive wrapper
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

/// cumsum wrapper
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

// ── Parallel wrappers（Phase 1: partial）──
// 签名：(a, partials, ndim, in_shape, in_strides, axis, axis_len,
//         out_size, tiles_per_output, stream)
// shared mem = 256 * sizeof(T)

#define REDUCE_PARTIAL_V2(OP)                                                  \
void musapy_##OP##_partial_i64_v2(                                            \
    const int64_t* __restrict__ a, int64_t* __restrict__ partials,            \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size,                               \
    size_t tiles_per_output, musaStream_t stream                              \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = out_size * tiles_per_output;                                \
    size_t smem = 256 * sizeof(int64_t);                                      \
    musapy_##OP##_partial_kernel_v2<int64_t>                                  \
        <<<grid, 256, smem, stream>>>(a, partials, meta, out_size, tiles_per_output); \
}                                                                             \
void musapy_##OP##_partial_f32_v2(                                            \
    const float* __restrict__ a, float* __restrict__ partials,                \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size,                               \
    size_t tiles_per_output, musaStream_t stream                              \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = out_size * tiles_per_output;                                \
    size_t smem = 256 * sizeof(float);                                        \
    musapy_##OP##_partial_kernel_v2<float>                                    \
        <<<grid, 256, smem, stream>>>(a, partials, meta, out_size, tiles_per_output); \
}                                                                             \
void musapy_##OP##_partial_f64_v2(                                            \
    const double* __restrict__ a, double* __restrict__ partials,              \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size,                               \
    size_t tiles_per_output, musaStream_t stream                              \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = out_size * tiles_per_output;                                \
    size_t smem = 256 * sizeof(double);                                       \
    musapy_##OP##_partial_kernel_v2<double>                                   \
        <<<grid, 256, smem, stream>>>(a, partials, meta, out_size, tiles_per_output); \
}

REDUCE_PARTIAL_V2(sum)
REDUCE_PARTIAL_V2(prod)
REDUCE_PARTIAL_V2(max)
REDUCE_PARTIAL_V2(min)
REDUCE_PARTIAL_V2(mean)
#undef REDUCE_PARTIAL_V2

// ── Parallel wrappers（Phase 2: final）──
// 签名：(partials, c, num_partials, out_size, stream)
// mean final 额外需要 axis_len

#define REDUCE_FINAL_V2(OP)                                                    \
void musapy_##OP##_final_i64_v2(                                              \
    const int64_t* __restrict__ partials, int64_t* __restrict__ c,            \
    size_t num_partials, size_t out_size, musaStream_t stream                 \
) {                                                                           \
    size_t smem = 256 * sizeof(int64_t);                                      \
    musapy_##OP##_final_kernel_v2<int64_t>                                    \
        <<<out_size, 256, smem, stream>>>(partials, c, num_partials, out_size); \
}                                                                             \
void musapy_##OP##_final_f32_v2(                                              \
    const float* __restrict__ partials, float* __restrict__ c,                \
    size_t num_partials, size_t out_size, musaStream_t stream                 \
) {                                                                           \
    size_t smem = 256 * sizeof(float);                                        \
    musapy_##OP##_final_kernel_v2<float>                                      \
        <<<out_size, 256, smem, stream>>>(partials, c, num_partials, out_size); \
}                                                                             \
void musapy_##OP##_final_f64_v2(                                              \
    const double* __restrict__ partials, double* __restrict__ c,              \
    size_t num_partials, size_t out_size, musaStream_t stream                 \
) {                                                                           \
    size_t smem = 256 * sizeof(double);                                       \
    musapy_##OP##_final_kernel_v2<double>                                     \
        <<<out_size, 256, smem, stream>>>(partials, c, num_partials, out_size); \
}

REDUCE_FINAL_V2(sum)
REDUCE_FINAL_V2(prod)
REDUCE_FINAL_V2(max)
REDUCE_FINAL_V2(min)
#undef REDUCE_FINAL_V2

// mean final 需要 axis_len 参数
#define MEAN_FINAL_V2(T, SUFFIX)                                              \
void musapy_mean_final_##SUFFIX##_v2(                                         \
    const T* __restrict__ partials, T* __restrict__ c,                        \
    size_t num_partials, size_t out_size, size_t axis_len, musaStream_t stream\
) {                                                                           \
    size_t smem = 256 * sizeof(T);                                            \
    musapy_mean_final_kernel_v2<T>                                            \
        <<<out_size, 256, smem, stream>>>(partials, c, num_partials, out_size, axis_len); \
}

MEAN_FINAL_V2(float, f32)
MEAN_FINAL_V2(double, f64)
#undef MEAN_FINAL_V2

// ── Argmax/Argmin parallel wrappers ──
// Phase 1 签名：(a, partials_val, partials_idx, ndim, in_shape, in_strides,
//                axis, axis_len, out_size, tiles_per_output, stream)
// smem = 256 * (sizeof(T) + sizeof(int64_t))

#define ARGREDUCE_PARTIAL_V2(OP)                                               \
void musapy_##OP##_partial_i64_v2(                                            \
    const int64_t* __restrict__ a, int64_t* __restrict__ partials_val,        \
    int64_t* __restrict__ partials_idx,                                       \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size,                               \
    size_t tiles_per_output, musaStream_t stream                              \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = out_size * tiles_per_output;                                \
    size_t smem = 256 * (sizeof(int64_t) + sizeof(int64_t));                  \
    musapy_##OP##_partial_kernel_v2<int64_t>                                  \
        <<<grid, 256, smem, stream>>>(a, partials_val, partials_idx, meta, out_size, tiles_per_output); \
}                                                                             \
void musapy_##OP##_partial_f32_v2(                                            \
    const float* __restrict__ a, float* __restrict__ partials_val,            \
    int64_t* __restrict__ partials_idx,                                       \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size,                               \
    size_t tiles_per_output, musaStream_t stream                              \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = out_size * tiles_per_output;                                \
    size_t smem = 256 * (sizeof(float) + sizeof(int64_t));                    \
    musapy_##OP##_partial_kernel_v2<float>                                    \
        <<<grid, 256, smem, stream>>>(a, partials_val, partials_idx, meta, out_size, tiles_per_output); \
}                                                                             \
void musapy_##OP##_partial_f64_v2(                                            \
    const double* __restrict__ a, double* __restrict__ partials_val,          \
    int64_t* __restrict__ partials_idx,                                       \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size,                               \
    size_t tiles_per_output, musaStream_t stream                              \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = out_size * tiles_per_output;                                \
    size_t smem = 256 * (sizeof(double) + sizeof(int64_t));                   \
    musapy_##OP##_partial_kernel_v2<double>                                   \
        <<<grid, 256, smem, stream>>>(a, partials_val, partials_idx, meta, out_size, tiles_per_output); \
}

ARGREDUCE_PARTIAL_V2(argmax)
ARGREDUCE_PARTIAL_V2(argmin)
#undef ARGREDUCE_PARTIAL_V2

// Phase 2 签名：(partials_val, partials_idx, c, num_partials, out_size, stream)
#define ARGREDUCE_FINAL_V2(OP)                                                 \
void musapy_##OP##_final_i64_v2(                                              \
    const int64_t* __restrict__ partials_val,                                 \
    const int64_t* __restrict__ partials_idx,                                 \
    int64_t* __restrict__ c,                                                  \
    size_t num_partials, size_t out_size, musaStream_t stream                 \
) {                                                                           \
    size_t smem = 256 * (sizeof(int64_t) + sizeof(int64_t));                  \
    musapy_##OP##_final_kernel_v2<int64_t>                                    \
        <<<out_size, 256, smem, stream>>>(partials_val, partials_idx, c, num_partials, out_size); \
}                                                                             \
void musapy_##OP##_final_f32_v2(                                              \
    const float* __restrict__ partials_val,                                   \
    const int64_t* __restrict__ partials_idx,                                 \
    int64_t* __restrict__ c,                                                  \
    size_t num_partials, size_t out_size, musaStream_t stream                 \
) {                                                                           \
    size_t smem = 256 * (sizeof(float) + sizeof(int64_t));                    \
    musapy_##OP##_final_kernel_v2<float>                                      \
        <<<out_size, 256, smem, stream>>>(partials_val, partials_idx, c, num_partials, out_size); \
}                                                                             \
void musapy_##OP##_final_f64_v2(                                              \
    const double* __restrict__ partials_val,                                  \
    const int64_t* __restrict__ partials_idx,                                 \
    int64_t* __restrict__ c,                                                  \
    size_t num_partials, size_t out_size, musaStream_t stream                 \
) {                                                                           \
    size_t smem = 256 * (sizeof(double) + sizeof(int64_t));                   \
    musapy_##OP##_final_kernel_v2<double>                                     \
        <<<out_size, 256, smem, stream>>>(partials_val, partials_idx, c, num_partials, out_size); \
}

ARGREDUCE_FINAL_V2(argmax)
ARGREDUCE_FINAL_V2(argmin)
#undef ARGREDUCE_FINAL_V2

} // extern "C"
