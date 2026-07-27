// elementwise.mu — 逐元素算子（ADR L2-2）
//
// 纯并行计算 kernel，无内存分配、无 host 代码、无错误返回。
// 所有指针 __restrict__（由 ops 层 alias 检测保证）。
// ABI 版本嵌入符号名：musapy_<op>_<dtype>_v<abi>（ADR L2-1）

#include "include/common.h"

template <typename T>
__global__ void musapy_add_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" {

void musapy_add_f32_v1(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    size_t n,
    musaStream_t stream
) {
    musapy_add_kernel<float><<<grid_size_1d(n), 256, 0, stream>>>(a, b, c, n);
}

void musapy_add_f64_v1(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ c,
    size_t n,
    musaStream_t stream
) {
    musapy_add_kernel<double><<<grid_size_1d(n), 256, 0, stream>>>(a, b, c, n);
}

} // extern "C"
