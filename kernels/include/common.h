#pragma once

#include <musa_runtime.h>
#include <sys/types.h>

/// 计算 1D grid 的大小（block 256，向上取整）。
static inline size_t grid_size_1d(size_t n) {
    return (n + 255) / 256;
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
