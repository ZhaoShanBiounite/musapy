#pragma once

#include <musa_runtime.h>
#include <sys/types.h>

/// 计算 1D grid 的大小（block 256，向上取整）。
static inline size_t grid_size_1d(size_t n) {
    return (n + 255) / 256;
}

/// 检测 strides 是否为 C-contiguous（row-major，元素单位）。
/// 用于 wrapper 内部选择 flat kernel 还是 stride-aware kernel。
static inline int is_contiguous_strides(const size_t* shape, const ssize_t* strides, int ndim) {
    ssize_t expected = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (strides[i] != expected) return 0;
        expected *= (ssize_t)shape[i];
    }
    return 1;
}

/// 将输出线性索引（row-major contiguous）转换为指定 stride 操作数的元素偏移。
///
/// - `linear_idx`：输出（连续）的线性索引 [0, product(shape))
/// - `shape`：广播输出 shape，长度 ndim
/// - `strides`：操作数 strides（元素单位），长度 ndim；broadcast 维为 0
/// - `ndim`：维度数
///
/// 算法：从最低维开始，逐维取坐标并累加 coord * stride。
/// broadcast 维 stride=0 → 偏移不随该维变化（元素被复制）。
__device__ static inline size_t offset_nd(size_t linear_idx, const size_t* shape,
                                          const ssize_t* strides, int ndim) {
    size_t offset = 0;
    for (int i = ndim - 1; i >= 0; i--) {
        size_t coord = linear_idx % shape[i];
        linear_idx /= shape[i];
        offset += (size_t)((ssize_t)coord * strides[i]);
    }
    return offset;
}

/// 给定 output linear idx 和 axis 坐标 k，计算 reduction input 元素偏移。
///
/// - `out_idx`：输出（连续）的线性索引 [0, out_size)
/// - `in_shape`：输入 shape，长度 ndim
/// - `in_strides`：输入 strides（元素单位），长度 ndim
/// - `ndim`：输入维度数
/// - `axis`：被缩减的轴 [0, ndim)
/// - `k`：该轴上的坐标 [0, in_shape[axis])
///
/// 算法：将 out_idx 展开为 output 坐标（in_shape 去掉 axis 维），
/// 在 axis 位置插入 k，与 in_strides 点积得到 input offset。
__device__ static inline size_t reduce_input_offset(
    size_t out_idx, const size_t* in_shape, const ssize_t* in_strides,
    int ndim, int axis, size_t k
) {
    // 單遍計算：提取非-axis 維坐標並直接累加 offset。
    // 加法交換律允許在分解 out_idx 的同一循環中累加，無需中間存儲。
    size_t offset = k * (size_t)in_strides[axis];
    size_t tmp = out_idx;
    for (int i = ndim - 1; i >= 0; i--) {
        if (i == axis) continue;
        size_t coord = tmp % in_shape[i];
        tmp /= in_shape[i];
        offset += coord * (size_t)in_strides[i];
    }
    return offset;
}
