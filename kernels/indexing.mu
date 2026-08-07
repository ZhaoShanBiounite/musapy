// indexing.mu — 索引算子 GPU kernel（Phase 6, ADR 002-D4）
//
// gather/scatter：按 axis + indices 取/写元素（copy 语义，分配新 buffer）。
// copy：stride-aware identity（视图物化为连续布局）。
//
// 约定（与 elementwise.mu 一致）：
// - 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回
// - 输入指针由 ops 层按 layout.offset 预调整（common.h offset_nd 的
//   无符号回绕语义要求基指针指向逻辑首元素）
// - indices 固定 int64；越界由 device 侧错误标志报告（P1，见下）
//
// ABI 版本嵌入符号名：gather/scatter v2 起带 _v2 后缀（P1 签名变更）；
// copy 保持首版签名。

#include "include/common.h"

#define MUSAPY_MAX_NDIM 32

// ── 元数据结构 ──────────────────────────────────────────────

/// gather 参数：output shape（axis 维 = n_indices）+ input strides +
/// axis 长度（device 侧越界检查，P1）。
struct GatherMeta {
    int ndim;
    int axis;
    size_t out_shape[MUSAPY_MAX_NDIM];
    ssize_t in_strides[MUSAPY_MAX_NDIM];
    size_t in_axis_len;
};

/// scatter 参数：values shape（axis 维 = n_indices）+ values strides +
/// output 各维的连续 stride（row-major）+ axis 长度（device 侧越界检查，P1）。
/// output 为连续布局。
struct ScatterMeta {
    int ndim;
    int axis;
    size_t val_shape[MUSAPY_MAX_NDIM];
    ssize_t val_strides[MUSAPY_MAX_NDIM];
    size_t out_strides[MUSAPY_MAX_NDIM];
    size_t out_axis_len;
};

// ── P1：device 侧索引越界报告（方案二：GPU 错误标志）────────
//
// gather/scatter 不再依赖 host 端同步 D2H 校验（原 ~10ms/op 的性能瓶颈）：
// 每个线程检查自己用到的 index，越界时跳过本次读/写，用 atomicCAS 记录
// 首个越界条目的展平序号（pos）与越界索引值（val），并 atomicOr 置位 flag。
// host 在下一次 Stream::synchronize() 批量读回 flag 并带算子上下文报错
// （见 musapy-core stream.rs 的 index_checks 机制）。
//
// 错误槽布局（16B）：[flag: int][pos: int][val: long long]，
// 初始/复位状态 flag=0、pos=-1（atomicCAS 哨兵）。
// pos 用 int：entry > 2^31 时截断，仅用于错误定位，可接受。

__device__ static inline void musapy_report_index_oob(
    int* err_flag, int* err_pos, long long* err_val,
    size_t entry, long long bad_index
) {
    if (atomicCAS(err_pos, -1, (int)entry) == -1) {
        *err_val = bad_index;  // 仅 CAS 胜出者写 val，与 pos 配对
    }
    atomicOr(err_flag, 1);
}

/// copy 参数（stride-aware identity，与 NdMetaUnary 同构）。
struct CopyMeta {
    int ndim;
    size_t shape[MUSAPY_MAX_NDIM];
    ssize_t in_strides[MUSAPY_MAX_NDIM];
};

// ── Kernels ─────────────────────────────────────────────────

/// gather：out[idx] = input[offset]，其中 axis 维坐标取自 indices。
/// 越界索引：跳过该元素读/写并记录错误标志（P1，host 在 sync 时报错）。
///
/// 性能（P1 实测）：64 位整数 div/mod 在 mp_22 上是软件实现，逐元素 unravel
/// 会让 kernel 变成计算瓶颈（1M f32 gather：f32/f64 同速 ~0.47ms，2D 3 倍慢）。
/// 因此：ndim==1 直接映射免 unravel；总元素数 ≤ 2^32 时用 32 位 div/mod
/// （offset 仍 ssize_t 累加，strides 可超 32 位）；更大才走 64 位路径。
template <typename T>
__global__ void musapy_gather_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    const int64_t* __restrict__ indices, GatherMeta meta, size_t n_out,
    int* err_flag, int* err_pos, long long* err_val
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_out) return;

    // ndim==1 快路径：无 unravel
    if (meta.ndim == 1) {
        long long raw = (long long)indices[idx];
        if (raw < 0 || raw >= (long long)meta.in_axis_len) {
            musapy_report_index_oob(err_flag, err_pos, err_val, idx, raw);
            return;
        }
        output[idx] = input[(size_t)(raw * meta.in_strides[0])];
        return;
    }

    ssize_t off = 0;
    if (n_out <= 0xFFFFFFFFull) {
        unsigned int tmp = (unsigned int)idx;
        for (int i = meta.ndim - 1; i >= 0; i--) {
            unsigned int dim = (unsigned int)meta.out_shape[i];
            unsigned int coord = tmp % dim;
            tmp /= dim;
            size_t k;
            if (i == meta.axis) {
                long long raw = (long long)indices[coord];
                if (raw < 0 || raw >= (long long)meta.in_axis_len) {
                    musapy_report_index_oob(err_flag, err_pos, err_val, idx, raw);
                    return;
                }
                k = (size_t)raw;
            } else {
                k = coord;
            }
            off += (ssize_t)k * meta.in_strides[i];
        }
    } else {
        size_t tmp = idx;
        for (int i = meta.ndim - 1; i >= 0; i--) {
            size_t coord = tmp % meta.out_shape[i];
            tmp /= meta.out_shape[i];
            size_t k;
            if (i == meta.axis) {
                long long raw = (long long)indices[coord];
                if (raw < 0 || raw >= (long long)meta.in_axis_len) {
                    musapy_report_index_oob(err_flag, err_pos, err_val, idx, raw);
                    return;
                }
                k = (size_t)raw;
            } else {
                k = coord;
            }
            off += (ssize_t)k * meta.in_strides[i];
        }
    }
    output[idx] = input[(size_t)off];
}

/// scatter：output[out_offset] = values[idx]，axis 维坐标经 indices 映射。
/// 每个线程处理一个 values 元素；重复 indices 的写序未定义（与 PyTorch 一致）。
/// 越界索引：跳过该元素写并记录错误标志（P1，host 在 sync 时报错）。
/// 快路径策略与 gather 相同（ndim==1 免 unravel；≤2^32 用 32 位 div/mod）。
template <typename T>
__global__ void musapy_scatter_kernel(
    T* __restrict__ output, const T* __restrict__ values,
    const int64_t* __restrict__ indices, ScatterMeta meta, size_t n_values,
    int* err_flag, int* err_pos, long long* err_val
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;

    // ndim==1 快路径：无 unravel
    if (meta.ndim == 1) {
        long long raw = (long long)indices[idx];
        if (raw < 0 || raw >= (long long)meta.out_axis_len) {
            musapy_report_index_oob(err_flag, err_pos, err_val, idx, raw);
            return;
        }
        output[(size_t)(raw * meta.out_strides[0])] =
            values[(size_t)(idx * meta.val_strides[0])];
        return;
    }

    size_t out_off = 0;
    ssize_t val_off = 0;
    if (n_values <= 0xFFFFFFFFull) {
        unsigned int tmp = (unsigned int)idx;
        for (int i = meta.ndim - 1; i >= 0; i--) {
            unsigned int dim = (unsigned int)meta.val_shape[i];
            unsigned int coord = tmp % dim;
            tmp /= dim;
            val_off += (ssize_t)coord * meta.val_strides[i];
            size_t k;
            if (i == meta.axis) {
                long long raw = (long long)indices[coord];
                if (raw < 0 || raw >= (long long)meta.out_axis_len) {
                    musapy_report_index_oob(err_flag, err_pos, err_val, idx, raw);
                    return;
                }
                k = (size_t)raw;
            } else {
                k = coord;
            }
            out_off += k * meta.out_strides[i];
        }
    } else {
        size_t tmp = idx;
        for (int i = meta.ndim - 1; i >= 0; i--) {
            size_t coord = tmp % meta.val_shape[i];
            tmp /= meta.val_shape[i];
            val_off += (ssize_t)coord * meta.val_strides[i];
            size_t k;
            if (i == meta.axis) {
                long long raw = (long long)indices[coord];
                if (raw < 0 || raw >= (long long)meta.out_axis_len) {
                    musapy_report_index_oob(err_flag, err_pos, err_val, idx, raw);
                    return;
                }
                k = (size_t)raw;
            } else {
                k = coord;
            }
            out_off += k * meta.out_strides[i];
        }
    }
    output[out_off] = values[(size_t)val_off];
}

/// copy：out[idx] = input[offset_nd(idx)]（视图物化为连续布局）。
/// n ≤ 2^32 时用 32 位 div/mod unravel（mp_22 上 64 位整数除法为软件模拟，
/// 是 flip 等 strided 视图物化的主要瓶颈，P4）。
template <typename T>
__global__ void musapy_copy_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    CopyMeta meta, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    size_t off;
    if (n <= 0xFFFFFFFFull) {
        unsigned int tmp = (unsigned int)idx;
        off = 0;
        for (int i = meta.ndim - 1; i >= 0; i--) {
            unsigned int dim = (unsigned int)meta.shape[i];
            unsigned int coord = tmp % dim;
            tmp /= dim;
            off += (size_t)((ssize_t)coord * meta.in_strides[i]);
        }
    } else {
        size_t tmp = idx;
        off = 0;
        for (int i = meta.ndim - 1; i >= 0; i--) {
            size_t coord = tmp % meta.shape[i];
            tmp /= meta.shape[i];
            off += (size_t)((ssize_t)coord * meta.in_strides[i]);
        }
    }
    output[idx] = input[off];
}

/// 2D 转置物化专用 kernel（P4）：视图 strides == [1, rows]（= 连续数组
/// 的转置，V[r][c] = base[c*rows + r]）。
///
/// 经典 tiled 转置：32×32 tile，block 32×8=256 线程，每线程 4 元素。
/// 读侧沿源 r 连续（base 行内），写侧沿输出 c 连续——两侧均全合并；
/// smem [32][33] padding 防 bank conflict（读写相位各错开 1）。
template <typename T>
__global__ void musapy_copy_transpose2d_tiled_kernel(
    const T* __restrict__ src, T* __restrict__ dst,
    size_t rows, size_t cols
) {
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x;              // [0,32)：tile 内 r 坐标（输入 fast 维）
    int ty = threadIdx.y;              // [0,8)：tile 内 c 坐标块
    size_t r0 = blockIdx.x * TILE_DIM;
    size_t c0 = blockIdx.y * TILE_DIM;

    // 读：warp（tx 连续）沿 r 合并；tile[ty+i][tx] = src[c][r]
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        size_t r = r0 + tx;
        size_t c = c0 + ty + i;
        if (r < rows && c < cols) {
            tile[ty + i][tx] = src[c * rows + r];
        }
    }
    __syncthreads();
    // 写：warp（tx 连续）沿 c 合并；dst[r][c] = tile[c_in_tile][r_in_tile]
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        size_t r = r0 + ty + i;
        size_t c = c0 + tx;
        if (r < rows && c < cols) {
            dst[r * cols + c] = tile[tx][ty + i];
        }
    }
}

// ── extract_diag（P0：solve 奇异检测 LU 对角提取）───────────────
// 绕开 musaMemcpy2D 跨步 D2H（SDK 3.1.0 逐行传输 ~26µs/行，8KB 对角
// 实测 26.5ms；且该 API 小 pitch D2H 行为非确定性，见 sdk-3.1.0-
// limitations.md）。本 kernel 把列主序 LU 缓冲的对角 U(k,k)（偏移
// k·ldu 元素）提取为连续数组，host 侧一次连续 D2H（0.18ms）读回。
// 语义：U(k,k) == 0.0 精确零（LAPACK 判据，与旧 host 扫描一致）。

template <typename T>
__global__ void musapy_extract_diag_kernel(
    const T* __restrict__ lu, T* __restrict__ diag, size_t n, size_t ldu
) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) diag[k] = lu[k * ldu];
}

// ── extern "C" 稳定 ABI ────────────────────────────────────────

extern "C" {

// ── gather ──

#define GATHER_WRAPPER(T, SUFFIX)                                             \
void musapy_gather_##SUFFIX##_v2(                                            \
    const T* __restrict__ input, T* __restrict__ output,                     \
    const int64_t* __restrict__ indices,                                     \
    int ndim, int axis, const size_t* out_shape, const ssize_t* in_strides,  \
    size_t n_out, size_t axis_len,                                           \
    int* err_flag, int* err_pos, long long* err_val, musaStream_t stream     \
) {                                                                          \
    if (n_out == 0) return;                                                  \
    GatherMeta meta;                                                         \
    meta.ndim = ndim;                                                        \
    meta.axis = axis;                                                        \
    meta.in_axis_len = axis_len;                                             \
    for (int i = 0; i < ndim; i++) {                                        \
        meta.out_shape[i] = out_shape[i];                                    \
        meta.in_strides[i] = in_strides[i];                                  \
    }                                                                        \
    musapy_gather_kernel<T><<<grid_size_1d(n_out), 256, 0, stream>>>(        \
        input, output, indices, meta, n_out, err_flag, err_pos, err_val);    \
}

GATHER_WRAPPER(float, f32)
GATHER_WRAPPER(double, f64)
GATHER_WRAPPER(int32_t, i32)
GATHER_WRAPPER(int64_t, i64)

// ── scatter ──

#define SCATTER_WRAPPER(T, SUFFIX)                                            \
void musapy_scatter_##SUFFIX##_v2(                                           \
    T* __restrict__ output, const T* __restrict__ values,                    \
    const int64_t* __restrict__ indices,                                     \
    int ndim, int axis, const size_t* val_shape, const ssize_t* val_strides, \
    const size_t* out_strides, size_t n_values, size_t axis_len,             \
    int* err_flag, int* err_pos, long long* err_val, musaStream_t stream     \
) {                                                                          \
    if (n_values == 0) return;                                               \
    ScatterMeta meta;                                                        \
    meta.ndim = ndim;                                                        \
    meta.axis = axis;                                                        \
    meta.out_axis_len = axis_len;                                            \
    for (int i = 0; i < ndim; i++) {                                        \
        meta.val_shape[i] = val_shape[i];                                    \
        meta.val_strides[i] = val_strides[i];                                \
        meta.out_strides[i] = out_strides[i];                                \
    }                                                                        \
    musapy_scatter_kernel<T><<<grid_size_1d(n_values), 256, 0, stream>>>(    \
        output, values, indices, meta, n_values, err_flag, err_pos, err_val);\
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

// ── copy（2D 转置 tiled，P4）──
// 签名：(src, dst, rows, cols, stream)。src 为转置视图底层 buffer 指针
// （host 已调整 offset），rows/cols 为视图 shape；src[c*rows + r] 为
// 视图 (r, c) 元素（= 底层连续数组的转置）。

#define COPY_TRANSPOSE2D_WRAPPER(T, SUFFIX)                                   \
void musapy_copy_transpose2d_##SUFFIX(                                         \
    const T* __restrict__ src, T* __restrict__ dst,                           \
    size_t rows, size_t cols, musaStream_t stream                             \
) {                                                                            \
    if (rows == 0 || cols == 0) return;                                       \
    dim3 grid((unsigned)((rows + 31) / 32), (unsigned)((cols + 31) / 32));    \
    dim3 block(32, 8);                                                        \
    musapy_copy_transpose2d_tiled_kernel<T><<<grid, block, 0, stream>>>(      \
        src, dst, rows, cols);                                                \
}

COPY_TRANSPOSE2D_WRAPPER(float, f32)
COPY_TRANSPOSE2D_WRAPPER(double, f64)
COPY_TRANSPOSE2D_WRAPPER(int32_t, i32)
COPY_TRANSPOSE2D_WRAPPER(int64_t, i64)

// ── extract_diag（P0）──

#define EXTRACT_DIAG_WRAPPER(T, SUFFIX)                                        \
void musapy_extract_diag_##SUFFIX##_v1(                                         \
    const T* __restrict__ lu, T* __restrict__ diag,                            \
    size_t n, size_t ldu, musaStream_t stream                                  \
) {                                                                            \
    if (n == 0) return;                                                        \
    musapy_extract_diag_kernel<T><<<grid_size_1d(n), 256, 0, stream>>>(        \
        lu, diag, n, ldu);                                                     \
}

EXTRACT_DIAG_WRAPPER(float, f32)
EXTRACT_DIAG_WRAPPER(double, f64)

} // extern "C"
