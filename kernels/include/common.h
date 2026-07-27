#pragma once

#include <musa_runtime.h>

/// 计算 1D grid 的大小（block 256，向上取整）。
static inline size_t grid_size_1d(size_t n) {
    return (n + 255) / 256;
}
