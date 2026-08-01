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
    // 單遍計算：提取非-axis 維坐標並直接累加 offset。
    // 加法交換律允許在分解 out_idx 的同一循環中累加，無需中間存儲。
    // 這避免了 stack array（mcc 會 spill 到 local memory），
    // 也避免了固定數量寄存器變量的 ndim 上限問題。
    size_t offset = k * (size_t)meta.in_strides[axis];
    size_t tmp = out_idx;
    for (int i = ndim - 1; i >= 0; i--) {
        if (i == axis) continue;
        size_t coord = tmp % meta.in_shape[i];
        tmp /= meta.in_shape[i];
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

// ── Cumsum（work-efficient parallel prefix sum）──────────────
//
// 原 naive 实现是 O(N²) work（每个 thread 从头重算前缀），在长 axis 上
// 退化灾难性（1M 元素 1D 需 3920ms）。这里改为经典三阶段 inclusive scan：
//
//   Phase 1 (block-local):  每 block 用 Blelloch 树形 scan 算出本 block 内
//                            的 inclusive prefix，写回输出；同时记录 block 总和。
//   Phase 2 (scan sums):    单 block 对所有 block 总和做 inclusive scan，
//                            得到「每个 block 之前的累加前缀」。
//   Phase 3 (add prefix):   每 block 把「自身之前的累加前缀」加到本 block
//                            所有输出上。
//
// 总 work O(N)（含 3 遍线性扫描），steps O(log B)（B = block 数）。
// 每一「行」（共享同一组非-axis 坐标的 L=axis_len 个元素）独立 scan。
//
// 约定：输入/输出共享 NdMetaReduce 结构（axis_len 即每行长度）。
// 一次 grid 同时处理所有行：grid = num_rows × blocks_per_row。

/// Cumsum 行 base offset 计算：给定行号 row（非 axis 维的展平索引），
/// 返回该行 axis_coord=0 处的输入偏移。
__device__ static inline size_t cumsum_row_base(
    size_t row, const NdMetaReduce& meta
) {
    int ndim = meta.ndim;
    int axis = meta.axis;
    size_t base = 0;
    // 从最低非-axis 维展开 row
    for (int i = ndim - 1; i >= 0; i--) {
        if (i == axis) continue;
        size_t coord = row % meta.in_shape[i];
        row /= meta.in_shape[i];
        base += coord * (size_t)meta.in_strides[i];
    }
    return base;
}

// ── Phase 1: block-local inclusive scan ───────────────────────
// grid: num_rows × blocks_per_row,  block: 256 threads
// 每 block 处理一行中连续 256 个元素（tile）。
// smem = 256 * sizeof(T)
template <typename T>
__global__ void musapy_cumsum_block_kernel_v2(
    const T* __restrict__ a, T* __restrict__ c, T* __restrict__ block_sums,
    NdMetaReduce meta, size_t num_rows, size_t blocks_per_row
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    int tid = threadIdx.x;

    size_t row = blockIdx.x / blocks_per_row;
    size_t block_in_row = blockIdx.x % blocks_per_row;
    if (row >= num_rows) return;

    size_t base = cumsum_row_base(row, meta);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t axis_len = meta.axis_len;
    size_t k0 = block_in_row * blockDim.x;  // 本 tile 起始 axis 坐标

    // 载入本 tile（边界外补 0，确保 scan 正确）
    T v = (T)0;
    size_t k = k0 + tid;
    if (k < axis_len) {
        v = a[base + (size_t)((ssize_t)k * axis_stride)];
    }
    sdata[tid] = v;
    __syncthreads();

    // Blelloch work-efficient inclusive scan（uphill + downhill）
    for (int s = 1; s < blockDim.x; s <<= 1) {
        int idx = (tid * 2 * s) + (2 * s - 1);
        if (idx < blockDim.x) {
            sdata[idx] += sdata[idx - s];
        }
        __syncthreads();
    }
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        int idx = (tid * 2 * s) + (2 * s - 1);
        if (idx + s < blockDim.x) {
            sdata[idx + s] += sdata[idx];
        }
        __syncthreads();
    }

    // 写回（边界外不写）
    if (k < axis_len) {
        c[base + (size_t)((ssize_t)k * axis_stride)] = sdata[tid];
    }

    // block 总和：有效区间内的最后一个 inclusive 值
    // blocks_per_row == 1 时 Phase 1 已输出最终结果，block_sums 可能为 null，跳过写入
    if (tid == 0 && blocks_per_row > 1) {
        size_t valid = axis_len - k0;
        T sum = (valid >= blockDim.x) ? sdata[blockDim.x - 1] : sdata[valid - 1];
        block_sums[row * blocks_per_row + block_in_row] = sum;
    }
}

// ── Phase 2: scan block_sums（单 block，原位 exclusive scan）──
// grid: 1, block: 256 threads（支撑 blocks_per_row ≤ 256，即 axis_len ≤ 65536）
// 输出 exclusive prefix：block_sums[row*bpr + i] = sum(block_sums[row*bpr + 0..i))，
// 即「第 i 个 block 之前所有 block 的累加」。第 0 个恒为 0。
//
// 算法：先做 inclusive scan，再各 thread 读取「前一个」位置（T0 补 0）。
// 因读 sdata[tid-1] 需在所有 thread 同步后做，用一个临时寄存器交换。
template <typename T>
__global__ void musapy_cumsum_scan_sums_kernel_v2(
    T* __restrict__ block_sums, size_t num_rows, size_t blocks_per_row
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    int tid = threadIdx.x;

    for (size_t row = 0; row < num_rows; row++) {
        T* row_sums = block_sums + row * blocks_per_row;

        // 载入（边界外补 0）
        T v = (tid < (int)blocks_per_row) ? row_sums[tid] : (T)0;
        sdata[tid] = v;
        __syncthreads();

        // inclusive scan over [0, blocks_per_row)
        for (int s = 1; s < (int)blocks_per_row; s <<= 1) {
            int idx = (tid * 2 * s) + (2 * s - 1);
            if (idx < (int)blocks_per_row) {
                sdata[idx] += sdata[idx - s];
            }
            __syncthreads();
        }
        for (int s = (int)blocks_per_row >> 1; s > 0; s >>= 1) {
            int idx = (tid * 2 * s) + (2 * s - 1);
            if (idx + s < (int)blocks_per_row) {
                sdata[idx + s] += sdata[idx];
            }
            __syncthreads();
        }

        // inclusive → exclusive：保存自身，写入前驱（T0 写 0）
        if (tid < (int)blocks_per_row) {
            T my_prev = (tid == 0) ? (T)0 : sdata[tid - 1];
            row_sums[tid] = my_prev;
        }
        __syncthreads();
    }
}

// ── Phase 3: add block prefix ────────────────────────────────
// grid: num_rows × blocks_per_row,  block: 256 threads
// 每 block 把「自身之前的累加前缀」加到本 tile 的所有输出元素。
template <typename T>
__global__ void musapy_cumsum_add_prefix_kernel_v2(
    T* __restrict__ c, const T* __restrict__ block_prefix,
    NdMetaReduce meta, size_t num_rows, size_t blocks_per_row
) {
    size_t row = blockIdx.x / blocks_per_row;
    size_t block_in_row = blockIdx.x % blocks_per_row;
    if (row >= num_rows) return;
    if (block_in_row == 0) return;  // 第 0 block 无前缀可加

    size_t base = cumsum_row_base(row, meta);
    ssize_t axis_stride = meta.in_strides[meta.axis];
    size_t axis_len = meta.axis_len;
    size_t k0 = block_in_row * blockDim.x;
    int tid = threadIdx.x;
    size_t k = k0 + tid;
    if (k >= axis_len) return;

    T prefix = block_prefix[row * blocks_per_row + block_in_row];
    T* dst = c + base + (size_t)((ssize_t)k * axis_stride);
    *dst += prefix;
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

/// cumsum v3 wrapper — work-efficient 三阶段 prefix sum。
///
/// 签名（v3）：
///   (a, c, tmp, ndim, in_shape, in_strides, axis, axis_len,
///    out_size, stream)
///
/// - `tmp`：host 预分配的 scratch buffer，大小 = num_rows × blocks_per_row × sizeof(T)
///   （num_rows = out_size / axis_len；blocks_per_row = (axis_len+255)/256）。
///   由 Rust 侧 op_builder 分配（stream-ordered）。
/// - 单 block 独占模式（blocks_per_row == 1）下，跳过 Phase 2/3 直接返回，
///   scratch buffer 可为 NULL（Rust 侧传 NULL 即可）。
#define CUMSUM_V3(T, SUFFIX)                                                   \
void musapy_cumsum_##SUFFIX##_v3(                                              \
    const T* __restrict__ a, T* __restrict__ c, T* __restrict__ tmp,           \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,               \
    int axis, size_t axis_len, size_t out_size, musaStream_t stream            \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim;                                                         \
    meta.axis = axis;                                                         \
    meta.axis_len = axis_len;                                                 \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t num_rows = (axis_len == 0) ? 0 : out_size / axis_len;             \
    size_t blocks_per_row = (axis_len + 255) / 256;                           \
    /* Phase 1: block-local scan，写出 c + block_sums */                     \
    size_t grid1 = num_rows * blocks_per_row;                                 \
    size_t smem = 256 * sizeof(T);                                            \
    musapy_cumsum_block_kernel_v2<T>                                          \
        <<<grid1, 256, smem, stream>>>(a, c, tmp, meta, num_rows, blocks_per_row); \
    /* blocks_per_row == 1 时 Phase 1 已写出最终结果，无需 Phase 2/3 */        \
    if (blocks_per_row > 1) {                                                 \
        /* Phase 2: scan block_sums (in-place exclusive) */                   \
        size_t smem2 = 256 * sizeof(T);                                       \
        musapy_cumsum_scan_sums_kernel_v2<T>                                  \
            <<<1, 256, smem2, stream>>>(tmp, num_rows, blocks_per_row);       \
        /* Phase 3: add block prefix */                                       \
        musapy_cumsum_add_prefix_kernel_v2<T>                                 \
            <<<grid1, 256, 0, stream>>>(c, tmp, meta, num_rows, blocks_per_row); \
    }                                                                         \
}

CUMSUM_V3(int64_t, i64)
CUMSUM_V3(float, f32)
CUMSUM_V3(double, f64)
#undef CUMSUM_V3

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
