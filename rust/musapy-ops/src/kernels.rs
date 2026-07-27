//! Kernel FFI 绑定（ADR L2-2）
//!
//! 真实模式：链接 build.rs 编译的 libmusapy_kernels.a 的 extern "C" 符号。
//! Mock 模式：提供 Rust stub（CPU 逐元素计算），让 CI/无 GPU 开发机也能跑。
//!
//! Kernel 接口契约（ADR L2-2）：
//! - 纯 C，无状态：`extern "C" void musapy_<op>_<dtype>_v<abi>(...)`
//! - 所有指针 `__restrict__`（由 ops 层 alias 检测保证）
//! - 无错误返回（kernels 返回 void；launch 错误由 ops 层检查 musaGetLastError）

use musapy_core::musa_ffi::musaStream_t;

// ── 真实模式：链接 C 编译的 kernel ──────────────────────────

#[cfg(not(musapy_mock_musa))]
unsafe extern "C" {
    /// `musapy_add_f32_v1(a, b, c, n, stream)` — float32 逐元素加法
    pub fn musapy_add_f32_v1(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        n: usize,
        stream: musaStream_t,
    );

    /// `musapy_add_f64_v1(a, b, c, n, stream)` — float64 逐元素加法
    pub fn musapy_add_f64_v1(
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        n: usize,
        stream: musaStream_t,
    );
}

// ── Mock 模式：CPU 逐元素加法 stub ─────────────────────────

#[cfg(musapy_mock_musa)]
pub unsafe fn musapy_add_f32_v1(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    n: usize,
    _stream: musaStream_t,
) {
    if a.is_null() || b.is_null() || c.is_null() || n == 0 {
        return;
    }
    let (sa, sb, sc) = (
        std::slice::from_raw_parts(a, n),
        std::slice::from_raw_parts(b, n),
        std::slice::from_raw_parts_mut(c, n),
    );
    for i in 0..n {
        sc[i] = sa[i] + sb[i];
    }
}

#[cfg(musapy_mock_musa)]
pub unsafe fn musapy_add_f64_v1(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    n: usize,
    _stream: musaStream_t,
) {
    if a.is_null() || b.is_null() || c.is_null() || n == 0 {
        return;
    }
    let (sa, sb, sc) = (
        std::slice::from_raw_parts(a, n),
        std::slice::from_raw_parts(b, n),
        std::slice::from_raw_parts_mut(c, n),
    );
    for i in 0..n {
        sc[i] = sa[i] + sb[i];
    }
}
