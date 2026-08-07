// reduction.mu — 缩减算子（ADR-002-D3）
//
// 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回。
// ABI 版本：_v2（stride-aware N-dimensional）
//
// 三代 kernel 共存：
//   naive（one-thread-per-output）：保留，用于 mock、axis_len ≤ 16 与 arg*
//   small_axis（P2，每输出 32..256 线程组 + warp shuffle）：17..1024 轴长
//   parallel（block-cooperative）：axis_len > 1024，两阶段缩减；
//     partial 每线程 REDUCE_ITEMS=4 元素（P2）+ warp shuffle 块内归约
//
// 符号命名：musapy_<op>_<dtype>_v2（naive）
//           musapy_<op>_partial_<dtype>_v2（parallel phase 1）
//           musapy_<op>_final_<dtype>_v2（parallel phase 2）

#include "include/common.h"
#include <math.h>
#include <float.h>
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

// ── Partial kernel 向量化辅助（P2）───────────────────────────
//
// 每线程处理 REDUCE_ITEMS=4 个连续 axis 元素（原为 1 个），线程数降为
// 1/4。host 侧 tiles_per_output = ceil(axis_len / (256 * REDUCE_ITEMS))。
// 注意：float4 显式向量加载在本编译器上会与 warp shuffle 组合出病态
// 代码（实测 47× 变慢，2026-08 基准探针），故只用标量循环——连续线程
// 访问连续元素，合并效果等价，实测与 float4 路径同速。

#define REDUCE_ITEMS 4

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

// ── 小 axis 并行 kernel（P2）─────────────────────────────────
//
// naive one-thread-per-output 在 axis_len 中等（17..1024）且 out_size 小
// 时并行度极差（256×256 axis=0 只有 256 线程）。此 kernel 每个输出配
// G 个线程（G=32/64/128/256，host 按 axis_len 选择），组内 warp shuffle
// + 静态 smem 两级归约。
//
// 线程映射：tid = out_idx * G + lane。G 为编译期模板参数（2 的幂），
// 除法/取模编译为移位（mp_22 64 位 div/mod 为软件模拟，须避免）。
// G ≥ 32 且 block=256 → 组边界与 warp/block 对齐，shuffle 全 warp 参与。

/// max/min 归约单位元（各 dtype 的极值）。
template <typename T> struct ReduceLimits;
template <> struct ReduceLimits<float> {
    __device__ static float lo() { return -FLT_MAX; }
    __device__ static float hi() { return FLT_MAX; }
};
template <> struct ReduceLimits<double> {
    __device__ static double lo() { return -DBL_MAX; }
    __device__ static double hi() { return DBL_MAX; }
};
template <> struct ReduceLimits<int64_t> {
    __device__ static int64_t lo() { return LLONG_MIN; }
    __device__ static int64_t hi() { return LLONG_MAX; }
};

template <typename T> struct ReduceOpSum {
    __device__ static T identity() { return (T)0; }
    __device__ static T combine(T x, T y) { return x + y; }
};
template <typename T> struct ReduceOpProd {
    __device__ static T identity() { return (T)1; }
    __device__ static T combine(T x, T y) { return x * y; }
};
template <typename T> struct ReduceOpMax {
    __device__ static T identity() { return ReduceLimits<T>::lo(); }
    __device__ static T combine(T x, T y) { return y > x ? y : x; }
};
template <typename T> struct ReduceOpMin {
    __device__ static T identity() { return ReduceLimits<T>::hi(); }
    __device__ static T combine(T x, T y) { return y < x ? y : x; }
};

/// 小 axis 归约：每输出 G 线程。DIVIDE_AXIS=true 时结果除以 axis_len（mean）。
template <typename T, typename Op, int G, bool DIVIDE_AXIS>
__global__ void musapy_reduce_small_axis_kernel(
    const T* __restrict__ a, T* __restrict__ c,
    NdMetaReduce meta, size_t out_size
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_idx = tid / G;          // G 为 2 的幂 → 移位
    int lane = (int)(tid & (G - 1));
    bool valid = out_idx < out_size;

    // 越界线程也参与 shuffle（单位元贡献），避免 warp 内分歧死锁
    T acc = Op::identity();
    if (valid) {
        size_t base = reduce_offset(out_idx, meta, 0);
        ssize_t axis_stride = meta.in_strides[meta.axis];
        for (size_t k = (size_t)lane; k < meta.axis_len; k += G) {
            acc = Op::combine(acc, a[base + k * (size_t)axis_stride]);
        }
    }

    // 第一级：warp 内 shuffle 归约（5 级）
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc = Op::combine(acc, __shfl_down_sync(0xffffffff, acc, offset));
    }

    if (G == 32) {
        if (valid && (threadIdx.x & 31) == 0) {
            if (DIVIDE_AXIS) acc /= (T)meta.axis_len;
            c[out_idx] = acc;
        }
        return;
    }

    // 第二级：组内跨 warp（G/32 个 warp 的 lane0 结果经静态 smem 合并）
    __shared__ T warp_partials[8];  // block=256 → 8 warps
    if ((threadIdx.x & 31) == 0) {
        warp_partials[threadIdx.x >> 5] = acc;
    }
    __syncthreads();
    if (valid && lane == 0) {
        int warps_per_group = G >> 5;
        int group_in_block = (int)threadIdx.x / G;
        int first = group_in_block * warps_per_group;
        T total = warp_partials[first];
        for (int w = 1; w < warps_per_group; w++) {
            total = Op::combine(total, warp_partials[first + w]);
        }
        if (DIVIDE_AXIS) total /= (T)meta.axis_len;
        c[out_idx] = total;
    }
}

// ── Cumsum（work-efficient parallel prefix sum）──────────────
//
// 原 naive 实现是 O(N²) work（每个 thread 从头重算前缀），在长 axis 上
// 退化灾难性（1M 元素 1D 需 3920ms）。这里改为经典三阶段 inclusive scan：
//
//   Phase 1 (block-local):  每 block 用 Blelloch 树形 scan 算出本 block 内
//                            的 inclusive prefix，写回输出；同时记录 block 总和。
//   Phase 2 (scan sums):    对 block 总和做 inclusive scan，得到全局前缀。
//                            * blocks_per_row ≤ 256：每行一个 block 单级扫描；
//                            * blocks_per_row > 256（host 保证 ≤ 65536）：
//                              分层——tile_scan 对每 256 个 sum 做 tile 内
//                              inclusive scan 并记 tile 总和，scan_sums 扫描
//                              tile 总和（每行一个 block），tile_prefix 把
//                              tile 前缀传播回各 block sum。容量 256^3 元素/行。
//   Phase 3 (add prefix):   每 block 把「自身之前的累加前缀」加到本 block
//                            所有输出上。
//
// Phase 2 输出的 block_sums 为 inclusive prefix（截至并包含各 block 的总和），
// Phase 3 对第 i 个 block（i ≥ 1）读取 block_prefix[i-1]。
//
// 总 work O(N)（含常数次线性扫描），steps O(log B)（B = block 数）。
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

// ── Phase 2: scan sums（每行一个 block，原位 inclusive scan）──
// grid: num_rows, block: 256 threads, smem = 256 * sizeof(T)
// 对至多 256 个 sum（n ≤ 256）做 inclusive scan：block_sums[row*n + i] =
// sum(block_sums[row*n + 0..=i])。≥ n 的位置补 0 参与扫描但不写回。
//
// Blelloch 树固定按 256（2 的幂）槽位扫描——补 0 不影响前 n 个前缀，
// 同时避免 n 非 2 的幂时带 guard 的树归约产生错误结果。
//
// 两个调用场景：
// - blocks_per_row ≤ 256：直接扫描 block_sums；
// - 分层路径：扫描 tile_sums（tiles_per_row ≤ 256，由 host 保证）。
template <typename T>
__global__ void musapy_cumsum_scan_sums_kernel_v2(
    T* __restrict__ sums, size_t num_rows, size_t n
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    int tid = threadIdx.x;

    size_t row = blockIdx.x;
    if (row >= num_rows) return;
    T* row_sums = sums + row * n;

    // 载入（边界外补 0）
    sdata[tid] = ((size_t)tid < n) ? row_sums[tid] : (T)0;
    __syncthreads();

    // Blelloch inclusive scan（固定 256 槽位，2 的幂）
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

    if ((size_t)tid < n) {
        row_sums[tid] = sdata[tid];
    }
}

// ── Phase 2a（分层路径）: tile 内 inclusive scan + 写 tile 总和 ──
// grid: num_rows × tiles_per_row, block: 256 threads, smem = 256 * sizeof(T)
// 每 block 对一个 tile 内的至多 256 个 block sum 做 inclusive scan（原位
// 写回），并把 tile 总和写入 tile_sums。仅在 blocks_per_row > 256
// （axis_len > 65536）时使用。
template <typename T>
__global__ void musapy_cumsum_tile_scan_kernel(
    T* __restrict__ block_sums, T* __restrict__ tile_sums,
    size_t num_rows, size_t blocks_per_row, size_t tiles_per_row
) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    int tid = threadIdx.x;

    size_t row = blockIdx.x / tiles_per_row;
    size_t tile = blockIdx.x % tiles_per_row;
    if (row >= num_rows) return;

    T* row_sums = block_sums + row * blocks_per_row;
    size_t j0 = tile * blockDim.x;  // 本 tile 起始 block-sum 下标

    // 载入本 tile（边界外补 0）
    T v = (T)0;
    size_t j = j0 + tid;
    if (j < blocks_per_row) {
        v = row_sums[j];
    }
    sdata[tid] = v;
    __syncthreads();

    // Blelloch inclusive scan（固定 256 槽位，2 的幂）
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

    // 原位写回（边界外不写）
    if (j < blocks_per_row) {
        row_sums[j] = sdata[tid];
    }

    // tile 总和：有效区间内的最后一个 inclusive 值
    if (tid == 0) {
        size_t valid = blocks_per_row - j0;
        T sum = (valid >= blockDim.x) ? sdata[blockDim.x - 1] : sdata[valid - 1];
        tile_sums[row * tiles_per_row + tile] = sum;
    }
}

// ── Phase 2b（分层路径）: 传播 tile 前缀 ──────────────────────
// grid: grid_size_1d(num_rows × blocks_per_row), block: 256 threads, 无 smem
// 每线程为一个 block sum 加上「本 tile 之前所有 tile 的总和」，使
// block_sums 成为全局 inclusive prefix（截至并包含本 block 的总和）。
// 前置条件：tile_sums 已被 scan_sums 扫描为 inclusive prefix
// （tile_sums[t-1] 即 tile t 之前所有 tile 的总和）。
template <typename T>
__global__ void musapy_cumsum_tile_prefix_kernel(
    T* __restrict__ block_sums, const T* __restrict__ tile_sums,
    size_t num_rows, size_t blocks_per_row, size_t tiles_per_row
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_rows * blocks_per_row;
    if (idx >= total) return;

    size_t row = idx / blocks_per_row;
    size_t j = idx % blocks_per_row;
    size_t tile = j / 256;
    if (tile == 0) return;  // tile 0 无前缀可加

    block_sums[idx] += tile_sums[row * tiles_per_row + tile - 1];
}

// ── Phase 3: add block prefix ────────────────────────────────
// grid: num_rows × blocks_per_row,  block: 256 threads
// 每 block 把「自身之前的累加前缀」加到本 tile 的所有输出元素。
// block_prefix 为全局 inclusive prefix（截至并包含各 block 的总和），
// 故第 i 个 block（i ≥ 1）的前缀位于 block_prefix[i-1]。
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

    // inclusive 约定：本 block 之前所有 block 的总和位于 block_in_row - 1
    //（block_in_row == 0 已在上方提前返回）
    T prefix = block_prefix[row * blocks_per_row + block_in_row - 1];
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
// 注（P2 起）：tiles_per_output = ceil(axis_len/1024)（每线程 4 元素），
// 且该路径仅 axis_len > 1024 时进入 → tiles 恒 ≥ 2，两阶段总成对执行；
// host 侧无「跳过 Phase 2」的捷径。

// ── Phase 1: partial reduction kernels ────────────────────────

/// partial kernel 公共框架（P2 增强版）——宏内联：
/// 每线程 REDUCE_ITEMS=4 个元素（原为 1）+ warp shuffle 两级块内归约。
/// 注意：必须宏内联展开，不能用 __device__ 函数——mcc 对含 extern
/// __shared__ + __shfl 的 device 函数不内联，函数调用路径实测 75× 变慢
/// （2026-08 probe：inline 0.053ms vs fn 4.0ms）。
/// OP 为 ReduceOpSum/Prod/Max/Min<T>，提供 identity()/combine()。
/// 宏展开后 kernel 内变量：acc（线程部分和）、total（块归约结果，仅
/// threadIdx.x==0 有效）。
#define PARTIAL_BODY(T, OP)                                                    \
    size_t base = reduce_offset(blockIdx.x / tiles_per_output, meta, 0);       \
    ssize_t axis_stride = meta.in_strides[meta.axis];                          \
    size_t total_threads = tiles_per_output * blockDim.x;                      \
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;                   \
    size_t axis_len = meta.axis_len;                                           \
    T acc = OP::identity();                                                    \
    for (size_t k0 = global_tid * REDUCE_ITEMS; k0 < axis_len;                 \
         k0 += total_threads * REDUCE_ITEMS) {                                 \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < REDUCE_ITEMS; j++) {                               \
            size_t k = k0 + j;                                                 \
            if (k < axis_len) {                                                \
                acc = OP::combine(acc, a[base + k * (size_t)axis_stride]);     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    extern __shared__ char smem[];                                             \
    T* sdata = (T*)smem;                                                       \
    for (int offset = 16; offset > 0; offset >>= 1) {                          \
        acc = OP::combine(acc, __shfl_down_sync(0xffffffff, acc, offset));     \
    }                                                                          \
    if ((threadIdx.x & 31) == 0) {                                             \
        sdata[threadIdx.x >> 5] = acc;                                         \
    }                                                                          \
    __syncthreads();                                                           \
    T total = OP::identity();                                                  \
    if (threadIdx.x < 32) {                                                    \
        int nwarps = (int)(blockDim.x >> 5);                                   \
        if ((int)threadIdx.x < nwarps) total = sdata[threadIdx.x];             \
        for (int offset = nwarps / 2; offset > 0; offset >>= 1) {              \
            total = OP::combine(total, __shfl_down_sync(0xffffffff, total, offset)); \
        }                                                                      \
    }

template <typename T>
__global__ void musapy_sum_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    PARTIAL_BODY(T, ReduceOpSum<T>)
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = total;
    }
}

template <typename T>
__global__ void musapy_prod_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    PARTIAL_BODY(T, ReduceOpProd<T>)
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = total;
    }
}

template <typename T>
__global__ void musapy_max_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    PARTIAL_BODY(T, ReduceOpMax<T>)
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = total;
    }
}

template <typename T>
__global__ void musapy_min_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    PARTIAL_BODY(T, ReduceOpMin<T>)
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = total;
    }
}

// mean partial = sum partial（final 阶段除以 axis_len）
template <typename T>
__global__ void musapy_mean_partial_kernel_v2(
    const T* __restrict__ a, T* __restrict__ partials,
    NdMetaReduce meta, size_t out_size, size_t tiles_per_output
) {
    size_t out_idx = blockIdx.x / tiles_per_output;
    size_t tile_idx = blockIdx.x % tiles_per_output;
    if (out_idx >= out_size) return;

    PARTIAL_BODY(T, ReduceOpSum<T>)
    if (threadIdx.x == 0) {
        partials[out_idx * tiles_per_output + tile_idx] = total;
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

// ── Argmid（P2b，2026-08-08）：中间级 arg 归约 ────────────────
// argreduce_partial 的 idx 是「轴内 k」，中间级必须沿袭输入 (val, idx)
// 对的 idx 而非重新计算——故不能复用 argreduce_partial（其输入只有
// 值数组）。本 kernel 读 (partials_val, partials_idx) 对，写缩小后的
// (out_val, out_idx) 对；配合 host 侧多级 partial 流水（P2b），把
// final 阶段每输出单 block 串行扫 partials 的瓶颈（64M sum 65536
// partials → 200 GB/s）改为逐级 ÷1024。

#define ARGMID_KERNEL(OP, CMP)                                                 \
template <typename T>                                                          \
__global__ void musapy_##OP##_mid_kernel_v2(                                   \
    const T* __restrict__ partials_val, const int64_t* __restrict__ partials_idx, \
    T* __restrict__ out_val, int64_t* __restrict__ out_idx,                    \
    size_t out_size, size_t tiles_per_output, size_t axis_len                  \
) {                                                                            \
    extern __shared__ char smem[];                                             \
    T* sval = (T*)smem;                                                        \
    int64_t* sidx = (int64_t*)(sval + blockDim.x);                             \
                                                                               \
    size_t out_idx_ = blockIdx.x / tiles_per_output;                           \
    size_t tile_idx = blockIdx.x % tiles_per_output;                           \
    if (out_idx_ >= out_size) return;                                          \
                                                                               \
    const T* vsrc = partials_val + out_idx_ * axis_len;                        \
    const int64_t* isrc = partials_idx + out_idx_ * axis_len;                  \
    size_t total_threads = tiles_per_output * blockDim.x;                      \
    size_t global_tid = tile_idx * blockDim.x + threadIdx.x;                   \
                                                                               \
    T best_val = vsrc[0];                                                      \
    int64_t best_idx = isrc[0];                                                \
    for (size_t k = global_tid; k < axis_len; k += total_threads) {            \
        T val = vsrc[k];                                                       \
        if (val CMP best_val) {                                                \
            best_val = val;                                                    \
            best_idx = isrc[k];                                                \
        }                                                                      \
    }                                                                          \
    sval[threadIdx.x] = best_val;                                              \
    sidx[threadIdx.x] = best_idx;                                              \
    __syncthreads();                                                           \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                             \
        if (threadIdx.x < s) {                                                 \
            if (sval[threadIdx.x + s] CMP sval[threadIdx.x]) {                 \
                sval[threadIdx.x] = sval[threadIdx.x + s];                     \
                sidx[threadIdx.x] = sidx[threadIdx.x + s];                     \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (threadIdx.x == 0) {                                                    \
        out_val[out_idx_ * tiles_per_output + tile_idx] = sval[0];             \
        out_idx[out_idx_ * tiles_per_output + tile_idx] = sidx[0];             \
    }                                                                          \
}

ARGMID_KERNEL(argmax, >)
ARGMID_KERNEL(argmin, <)

#undef ARGMID_KERNEL

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

/// 小 axis 并行 wrapper（P2）：sum/prod/max/min。
/// group_size ∈ {32,64,128,256}，host 按 axis_len 选择；
/// grid = ceil(out_size * group_size / 256)，block=256，无动态 smem。
#define REDUCE_SMALL_AXIS_V2(OP, OPSTRUCT)                                     \
void musapy_##OP##_small_axis_i64_v2(                                          \
    const int64_t* __restrict__ a, int64_t* __restrict__ c,                   \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, int group_size,               \
    musaStream_t stream                                                       \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = (out_size * (size_t)group_size + 255) / 256;                \
    switch (group_size) {                                                     \
        case 32: musapy_reduce_small_axis_kernel<int64_t, OPSTRUCT<int64_t>, 32, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        case 64: musapy_reduce_small_axis_kernel<int64_t, OPSTRUCT<int64_t>, 64, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        case 128: musapy_reduce_small_axis_kernel<int64_t, OPSTRUCT<int64_t>, 128, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        default: musapy_reduce_small_axis_kernel<int64_t, OPSTRUCT<int64_t>, 256, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
    }                                                                         \
}                                                                             \
void musapy_##OP##_small_axis_f32_v2(                                          \
    const float* __restrict__ a, float* __restrict__ c,                       \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, int group_size,               \
    musaStream_t stream                                                       \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = (out_size * (size_t)group_size + 255) / 256;                \
    switch (group_size) {                                                     \
        case 32: musapy_reduce_small_axis_kernel<float, OPSTRUCT<float>, 32, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        case 64: musapy_reduce_small_axis_kernel<float, OPSTRUCT<float>, 64, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        case 128: musapy_reduce_small_axis_kernel<float, OPSTRUCT<float>, 128, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        default: musapy_reduce_small_axis_kernel<float, OPSTRUCT<float>, 256, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
    }                                                                         \
}                                                                             \
void musapy_##OP##_small_axis_f64_v2(                                          \
    const double* __restrict__ a, double* __restrict__ c,                     \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, int group_size,               \
    musaStream_t stream                                                       \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = (out_size * (size_t)group_size + 255) / 256;                \
    switch (group_size) {                                                     \
        case 32: musapy_reduce_small_axis_kernel<double, OPSTRUCT<double>, 32, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        case 64: musapy_reduce_small_axis_kernel<double, OPSTRUCT<double>, 64, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        case 128: musapy_reduce_small_axis_kernel<double, OPSTRUCT<double>, 128, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        default: musapy_reduce_small_axis_kernel<double, OPSTRUCT<double>, 256, false><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
    }                                                                         \
}

REDUCE_SMALL_AXIS_V2(sum, ReduceOpSum)
REDUCE_SMALL_AXIS_V2(prod, ReduceOpProd)
REDUCE_SMALL_AXIS_V2(max, ReduceOpMax)
REDUCE_SMALL_AXIS_V2(min, ReduceOpMin)
#undef REDUCE_SMALL_AXIS_V2

/// mean 小 axis wrapper：只有 f32/f64（DIVIDE_AXIS=true）
#define MEAN_SMALL_AXIS_V2(T, SUFFIX)                                          \
void musapy_mean_small_axis_##SUFFIX##_v2(                                     \
    const T* __restrict__ a, T* __restrict__ c,                               \
    int ndim, const size_t* in_shape, const ssize_t* in_strides,              \
    int axis, size_t axis_len, size_t out_size, int group_size,               \
    musaStream_t stream                                                       \
) {                                                                           \
    NdMetaReduce meta;                                                        \
    meta.ndim = ndim; meta.axis = axis; meta.axis_len = axis_len;            \
    for (int i = 0; i < ndim; i++) {                                          \
        meta.in_shape[i] = in_shape[i];                                       \
        meta.in_strides[i] = in_strides[i];                                   \
    }                                                                         \
    size_t grid = (out_size * (size_t)group_size + 255) / 256;                \
    switch (group_size) {                                                     \
        case 32: musapy_reduce_small_axis_kernel<T, ReduceOpSum<T>, 32, true><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        case 64: musapy_reduce_small_axis_kernel<T, ReduceOpSum<T>, 64, true><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        case 128: musapy_reduce_small_axis_kernel<T, ReduceOpSum<T>, 128, true><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
        default: musapy_reduce_small_axis_kernel<T, ReduceOpSum<T>, 256, true><<<grid, 256, 0, stream>>>(a, c, meta, out_size); break; \
    }                                                                         \
}

MEAN_SMALL_AXIS_V2(float, f32)
MEAN_SMALL_AXIS_V2(double, f64)
#undef MEAN_SMALL_AXIS_V2

/// cumsum v3 wrapper — work-efficient 分层 prefix sum。
///
/// 签名（v3）：
///   (a, c, tmp, ndim, in_shape, in_strides, axis, axis_len,
///    out_size, stream)
///
/// - `tmp`：host 预分配的 scratch buffer（由 Rust 侧 op_builder 分配，
///   stream-ordered），布局为 block_sums 区 + tile_sums 区：
///   * blocks_per_row ≤ 256：仅需 block_sums 区，
///     大小 = num_rows × blocks_per_row × sizeof(T)；
///   * blocks_per_row > 256（host 保证 ≤ 65536，即 axis_len ≤ 256^3）：
///     block_sums 区后紧跟 tile_sums 区（num_rows × tiles_per_row × sizeof(T)，
///     tiles_per_row = ceil(blocks_per_row/256)），wrapper 内以指针偏移切分。
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
        /* Phase 2: 扫描 block_sums 为 inclusive prefix（每行一个 block）。
         * - bpr ≤ 256：单级 scan 即可；
         * - bpr > 256：分层——tile_scan（tile 内 inclusive + tile 总和）→
         *   scan tile_sums（每行 inclusive）→ tile_prefix（传播 tile 前缀），
         *   得全局 inclusive prefix。*/                                      \
        if (blocks_per_row <= 256) {                                          \
            musapy_cumsum_scan_sums_kernel_v2<T>                              \
                <<<num_rows, 256, smem, stream>>>(tmp, num_rows, blocks_per_row); \
        } else {                                                              \
            size_t tiles_per_row = (blocks_per_row + 255) / 256;              \
            T* tile_sums = tmp + num_rows * blocks_per_row;                   \
            musapy_cumsum_tile_scan_kernel<T>                                 \
                <<<num_rows * tiles_per_row, 256, smem, stream>>>(            \
                    tmp, tile_sums, num_rows, blocks_per_row, tiles_per_row); \
            musapy_cumsum_scan_sums_kernel_v2<T>                              \
                <<<num_rows, 256, smem, stream>>>(                            \
                    tile_sums, num_rows, tiles_per_row);                      \
            musapy_cumsum_tile_prefix_kernel<T>                               \
                <<<grid_size_1d(grid1), 256, 0, stream>>>(                    \
                    tmp, tile_sums, num_rows, blocks_per_row, tiles_per_row); \
        }                                                                     \
        /* Phase 3: add block prefix（inclusive 约定：读本 block 前驱）*/     \
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
#undef REDUCE_PARTIAL_V2

/// mean partial 只有 f32/f64（compute dtype 规则；P6 拆出专用宏，
/// 删掉原 REDUCE_PARTIAL_V2(mean) 顺带生成的 i64 死符号）。
#define MEAN_PARTIAL_V2(T, SUFFIX)                                            \
void musapy_mean_partial_##SUFFIX##_v2(                                       \
    const T* __restrict__ a, T* __restrict__ partials,                        \
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
    size_t smem = 256 * sizeof(T);                                            \
    musapy_mean_partial_kernel_v2<T>                                          \
        <<<grid, 256, smem, stream>>>(a, partials, meta, out_size, tiles_per_output); \
}

MEAN_PARTIAL_V2(float, f32)
MEAN_PARTIAL_V2(double, f64)
#undef MEAN_PARTIAL_V2

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

// ── Argmid（P2b）：中间级 (val, idx) 对归约 wrapper ──
// 签名：(partials_val, partials_idx, out_val, out_idx, out_size,
//        tiles_per_output, axis_len, stream)
#define ARGMID_WRAPPER(T, SUFFIX)                                              \
void musapy_argmax_mid_##SUFFIX##_v2(                                          \
    const T* __restrict__ partials_val, const int64_t* __restrict__ partials_idx, \
    T* __restrict__ out_val, int64_t* __restrict__ out_idx,                    \
    size_t out_size, size_t tiles_per_output, size_t axis_len,                 \
    musaStream_t stream                                                         \
) {                                                                            \
    size_t grid = out_size * tiles_per_output;                                 \
    size_t smem = 256 * (sizeof(T) + sizeof(int64_t));                         \
    musapy_argmax_mid_kernel_v2<T>                                             \
        <<<grid, 256, smem, stream>>>(partials_val, partials_idx,              \
        out_val, out_idx, out_size, tiles_per_output, axis_len);               \
}                                                                              \
void musapy_argmin_mid_##SUFFIX##_v2(                                          \
    const T* __restrict__ partials_val, const int64_t* __restrict__ partials_idx, \
    T* __restrict__ out_val, int64_t* __restrict__ out_idx,                    \
    size_t out_size, size_t tiles_per_output, size_t axis_len,                 \
    musaStream_t stream                                                         \
) {                                                                            \
    size_t grid = out_size * tiles_per_output;                                 \
    size_t smem = 256 * (sizeof(T) + sizeof(int64_t));                         \
    musapy_argmin_mid_kernel_v2<T>                                             \
        <<<grid, 256, smem, stream>>>(partials_val, partials_idx,              \
        out_val, out_idx, out_size, tiles_per_output, axis_len);               \
}

ARGMID_WRAPPER(int64_t, i64)
ARGMID_WRAPPER(float, f32)
ARGMID_WRAPPER(double, f64)

#undef ARGMID_WRAPPER

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
