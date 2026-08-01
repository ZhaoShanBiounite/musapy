// init.mu — 创建算子内核（Phase 5）
//
// 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回。
// 输出始终 C-contiguous（创建算子无输入 strides）。
//
// 内核类型：
//   1. fill: 写入常量值（zeros/ones/full 共用）
//   2. arange: out[i] = start + i * step
//   3. linspace: out[i] = start + i * (stop - start) / (n - 1)
//   4. eye: out[row*m + col] = (col - row == k) ? 1 : 0

#include "include/common.h"

// ── Fill kernel ─────────────────────────────────────────────

template <typename T>
__global__ void musapy_fill_kernel(T* __restrict__ out, T value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

// ── Arange kernel ───────────────────────────────────────────

template <typename T>
__global__ void musapy_arange_kernel(T* __restrict__ out, T start, T step, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + (T)idx * step;
    }
}

// ── Linspace kernel ─────────────────────────────────────────

template <typename T>
__global__ void musapy_linspace_kernel(T* __restrict__ out, T start, T step, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + (T)idx * step;
    }
}

// ── Eye kernel ──────────────────────────────────────────────

template <typename T>
__global__ void musapy_eye_kernel(T* __restrict__ out, size_t n_rows, size_t m, int k, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        size_t row = idx / m;
        size_t col = idx % m;
        // col - row == k (handle negative k via signed comparison)
        out[idx] = ((int)col - (int)row == k) ? (T)1 : (T)0;
    }
}

// ══════════════════════════════════════════════════════════════
// extern "C" wrapper 函数
// ══════════════════════════════════════════════════════════════

extern "C" {

// ── Fill wrappers ───────────────────────────────────────────

#define FILL_WRAPPER(SUFFIX, CTYPE)                                           \
void musapy_fill_##SUFFIX(CTYPE* out, CTYPE value, size_t n, musaStream_t stream) { \
    if (n == 0) return;                                                       \
    musapy_fill_kernel<CTYPE><<<grid_size_1d(n), 256, 0, stream>>>(out, value, n); \
}

FILL_WRAPPER(f32, float)
FILL_WRAPPER(f64, double)
FILL_WRAPPER(i64, int64_t)
FILL_WRAPPER(i32, int32_t)
FILL_WRAPPER(i16, int16_t)
FILL_WRAPPER(i8,  int8_t)
FILL_WRAPPER(u64, uint64_t)
FILL_WRAPPER(u32, uint32_t)
FILL_WRAPPER(u16, uint16_t)
FILL_WRAPPER(u8,  uint8_t)

#undef FILL_WRAPPER

// ── Arange wrappers ─────────────────────────────────────────

#define ARANGE_WRAPPER(SUFFIX, CTYPE)                                         \
void musapy_arange_##SUFFIX(CTYPE* out, CTYPE start, CTYPE step, size_t n, musaStream_t stream) { \
    if (n == 0) return;                                                       \
    musapy_arange_kernel<CTYPE><<<grid_size_1d(n), 256, 0, stream>>>(out, start, step, n); \
}

ARANGE_WRAPPER(f32, float)
ARANGE_WRAPPER(f64, double)
ARANGE_WRAPPER(i64, int64_t)
ARANGE_WRAPPER(i32, int32_t)

#undef ARANGE_WRAPPER

// ── Linspace wrappers ───────────────────────────────────────
// step = (stop - start) / (n - 1) 在 host 端计算，kernel 与 arange 同构。

#define LINESPACE_WRAPPER(SUFFIX, CTYPE)                                      \
void musapy_linspace_##SUFFIX(CTYPE* out, CTYPE start, CTYPE stop, size_t n, musaStream_t stream) { \
    if (n == 0) return;                                                       \
    CTYPE step;                                                               \
    if (n == 1) {                                                             \
        step = (CTYPE)0;                                                      \
    } else {                                                                  \
        step = (stop - start) / (CTYPE)(n - 1);                              \
    }                                                                         \
    musapy_linspace_kernel<CTYPE><<<grid_size_1d(n), 256, 0, stream>>>(out, start, step, n); \
}

LINESPACE_WRAPPER(f32, float)
LINESPACE_WRAPPER(f64, double)

#undef LINESPACE_WRAPPER

// ── Eye wrappers ────────────────────────────────────────────

#define EYE_WRAPPER(SUFFIX, CTYPE)                                            \
void musapy_eye_##SUFFIX(CTYPE* out, size_t n, size_t m, int k, musaStream_t stream) { \
    size_t total = n * m;                                                     \
    if (total == 0) return;                                                   \
    musapy_eye_kernel<CTYPE><<<grid_size_1d(total), 256, 0, stream>>>(out, n, m, k, total); \
}

EYE_WRAPPER(f32, float)
EYE_WRAPPER(f64, double)
EYE_WRAPPER(i64, int64_t)
EYE_WRAPPER(i32, int32_t)

#undef EYE_WRAPPER

} // extern "C"
