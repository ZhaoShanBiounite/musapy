#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stddef.h>
#include <stdint.h>

#define MUSAPY_ABI_VERSION 1

// ── Elementwise add operation (device function overloads) ──

template <typename T>
__device__ T musapy_add_op(T a, T b) {
    return a + b;
}

__device__ __half musapy_add_op(__half a, __half b) {
    return __hadd(a, b);
}

__device__ __nv_bfloat16 musapy_add_op(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd_bf16(a, b);
}

__device__ float2 musapy_add_op(float2 a, float2 b) {
    float2 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

__device__ double2 musapy_add_op(double2 a, double2 b) {
    double2 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

// ── Generic kernel ──

template <typename T>
__global__ void musapy_add_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = musapy_add_op(a[idx], b[idx]);
    }
}

// ── Exported wrappers (ABI stable) ──

#define MUSAPY_ADD_WRAPPER(dtype, suffix)                                      \
    extern "C" void musapy_add_##suffix##_v1(                                  \
        const dtype* a, const dtype* b, dtype* c,                              \
        size_t n, musaStream_t stream                                          \
    ) {                                                                        \
        constexpr size_t block_size = 256;                                     \
        size_t grid = (n + block_size - 1) / block_size;                       \
        musapy_add_kernel<dtype><<<grid, block_size, 0, stream>>>(a, b, c, n); \
    }

MUSAPY_ADD_WRAPPER(int8_t, i8)
MUSAPY_ADD_WRAPPER(int16_t, i16)
MUSAPY_ADD_WRAPPER(int32_t, i32)
MUSAPY_ADD_WRAPPER(int64_t, i64)
MUSAPY_ADD_WRAPPER(uint8_t, u8)
MUSAPY_ADD_WRAPPER(uint16_t, u16)
MUSAPY_ADD_WRAPPER(uint32_t, u32)
MUSAPY_ADD_WRAPPER(uint64_t, u64)
MUSAPY_ADD_WRAPPER(__half, f16)
MUSAPY_ADD_WRAPPER(float, f32)
MUSAPY_ADD_WRAPPER(double, f64)
MUSAPY_ADD_WRAPPER(__nv_bfloat16, bf16)
MUSAPY_ADD_WRAPPER(float2, c64)
MUSAPY_ADD_WRAPPER(double2, c128)

#undef MUSAPY_ADD_WRAPPER
