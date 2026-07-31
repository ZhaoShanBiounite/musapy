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
    // v1（保留）
    pub fn musapy_add_f32_v1(a: *const f32, b: *const f32, c: *mut f32, n: usize, stream: musaStream_t);
    pub fn musapy_add_f64_v1(a: *const f64, b: *const f64, c: *mut f64, n: usize, stream: musaStream_t);

    // v2 Binary
    pub fn musapy_add_f32_v2(a: *const f32, b: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_add_f64_v2(a: *const f64, b: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_sub_f32_v2(a: *const f32, b: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_sub_f64_v2(a: *const f64, b: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_mul_f32_v2(a: *const f32, b: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_mul_f64_v2(a: *const f64, b: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_div_f32_v2(a: *const f32, b: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_div_f64_v2(a: *const f64, b: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_pow_f32_v2(a: *const f32, b: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_pow_f64_v2(a: *const f64, b: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);

    // v2 Unary
    pub fn musapy_sin_f32_v2(a: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_sin_f64_v2(a: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cos_f32_v2(a: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cos_f64_v2(a: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_exp_f32_v2(a: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_exp_f64_v2(a: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_log_f32_v2(a: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_log_f64_v2(a: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_abs_f32_v2(a: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_abs_f64_v2(a: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_sign_f32_v2(a: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_sign_f64_v2(a: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_neg_f32_v2(a: *const f32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_neg_f64_v2(a: *const f64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);

    // v2 Clamp
    pub fn musapy_clamp_f32_v2(a: *const f32, c: *mut f32, lo: f32, hi: f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_clamp_f64_v2(a: *const f64, c: *mut f64, lo: f64, hi: f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);

    // v2 Cast → f32
    pub fn musapy_cast_i8_f32_v2(a: *const i8, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_i16_f32_v2(a: *const i16, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_i32_f32_v2(a: *const i32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_i64_f32_v2(a: *const i64, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u8_f32_v2(a: *const u8, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u16_f32_v2(a: *const u16, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u32_f32_v2(a: *const u32, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u64_f32_v2(a: *const u64, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_f64_f32_v2(a: *const f64, c: *mut f32, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);

    // v2 Cast → f64
    pub fn musapy_cast_i8_f64_v2(a: *const i8, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_i16_f64_v2(a: *const i16, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_i32_f64_v2(a: *const i32, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_i64_f64_v2(a: *const i64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u8_f64_v2(a: *const u8, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u16_f64_v2(a: *const u16, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u32_f64_v2(a: *const u32, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u64_f64_v2(a: *const u64, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_f32_f64_v2(a: *const f32, c: *mut f64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
}

// ── Mock 模式：CPU stub ─────────────────────────────────────

#[cfg(musapy_mock_musa)]
fn mock_offset_nd(linear_idx: usize, shape: &[usize], strides: &[isize]) -> usize {
    let mut offset = 0usize;
    let mut idx = linear_idx;
    let ndim = shape.len();
    for i in (0..ndim).rev() {
        let coord = idx % shape[i];
        idx /= shape[i];
        offset = (offset as isize + coord as isize * strides[i]) as usize;
    }
    offset
}

// Mock binary v2 宏
#[cfg(musapy_mock_musa)]
macro_rules! mock_binary_v2 {
    ($name:ident, $t:ty, $op:expr) => {
        pub unsafe fn $name(
            a: *const $t, b: *const $t, c: *mut $t,
            ndim: i32, shape: *const usize,
            a_strides: *const isize, b_strides: *const isize,
            _stream: musaStream_t,
        ) {
            if a.is_null() || b.is_null() || c.is_null() || ndim < 0 { return; }
            let ndim = ndim as usize;
            let shape_s = std::slice::from_raw_parts(shape, ndim);
            let as_s = std::slice::from_raw_parts(a_strides, ndim);
            let bs_s = std::slice::from_raw_parts(b_strides, ndim);
            let n: usize = shape_s.iter().product();
            let op: fn($t, $t) -> $t = $op;
            for idx in 0..n {
                let ao = mock_offset_nd(idx, shape_s, as_s);
                let bo = mock_offset_nd(idx, shape_s, bs_s);
                *c.add(idx) = op(*a.add(ao), *b.add(bo));
            }
        }
    };
}

// Mock unary v2 宏
#[cfg(musapy_mock_musa)]
macro_rules! mock_unary_v2 {
    ($name:ident, $t:ty, $op:expr) => {
        pub unsafe fn $name(
            a: *const $t, c: *mut $t,
            ndim: i32, shape: *const usize, a_strides: *const isize,
            _stream: musaStream_t,
        ) {
            if a.is_null() || c.is_null() || ndim < 0 { return; }
            let ndim = ndim as usize;
            let shape_s = std::slice::from_raw_parts(shape, ndim);
            let as_s = std::slice::from_raw_parts(a_strides, ndim);
            let n: usize = shape_s.iter().product();
            let op: fn($t) -> $t = $op;
            for idx in 0..n {
                let ao = mock_offset_nd(idx, shape_s, as_s);
                *c.add(idx) = op(*a.add(ao));
            }
        }
    };
}

// Mock cast v2 宏
#[cfg(musapy_mock_musa)]
macro_rules! mock_cast_v2 {
    ($name:ident, $src:ty, $dst:ty) => {
        pub unsafe fn $name(
            a: *const $src, c: *mut $dst,
            ndim: i32, shape: *const usize, a_strides: *const isize,
            _stream: musaStream_t,
        ) {
            if a.is_null() || c.is_null() || ndim < 0 { return; }
            let ndim = ndim as usize;
            let shape_s = std::slice::from_raw_parts(shape, ndim);
            let as_s = std::slice::from_raw_parts(a_strides, ndim);
            let n: usize = shape_s.iter().product();
            for idx in 0..n {
                let ao = mock_offset_nd(idx, shape_s, as_s);
                *c.add(idx) = *a.add(ao) as $dst;
            }
        }
    };
}

#[cfg(musapy_mock_musa)]
mod mock {
    use super::*;

    // v1（保留）
    pub unsafe fn musapy_add_f32_v1(a: *const f32, b: *const f32, c: *mut f32, n: usize, _s: musaStream_t) {
        if a.is_null() || b.is_null() || c.is_null() || n == 0 { return; }
        for i in 0..n { *c.add(i) = *a.add(i) + *b.add(i); }
    }
    pub unsafe fn musapy_add_f64_v1(a: *const f64, b: *const f64, c: *mut f64, n: usize, _s: musaStream_t) {
        if a.is_null() || b.is_null() || c.is_null() || n == 0 { return; }
        for i in 0..n { *c.add(i) = *a.add(i) + *b.add(i); }
    }

    // v2 Binary
    mock_binary_v2!(musapy_add_f32_v2, f32, |a, b| a + b);
    mock_binary_v2!(musapy_add_f64_v2, f64, |a, b| a + b);
    mock_binary_v2!(musapy_sub_f32_v2, f32, |a, b| a - b);
    mock_binary_v2!(musapy_sub_f64_v2, f64, |a, b| a - b);
    mock_binary_v2!(musapy_mul_f32_v2, f32, |a, b| a * b);
    mock_binary_v2!(musapy_mul_f64_v2, f64, |a, b| a * b);
    mock_binary_v2!(musapy_div_f32_v2, f32, |a, b| a / b);
    mock_binary_v2!(musapy_div_f64_v2, f64, |a, b| a / b);
    mock_binary_v2!(musapy_pow_f32_v2, f32, |a, b| a.powf(b));
    mock_binary_v2!(musapy_pow_f64_v2, f64, |a, b| a.powf(b));

    // v2 Unary
    mock_unary_v2!(musapy_sin_f32_v2, f32, |a| a.sin());
    mock_unary_v2!(musapy_sin_f64_v2, f64, |a| a.sin());
    mock_unary_v2!(musapy_cos_f32_v2, f32, |a| a.cos());
    mock_unary_v2!(musapy_cos_f64_v2, f64, |a| a.cos());
    mock_unary_v2!(musapy_exp_f32_v2, f32, |a| a.exp());
    mock_unary_v2!(musapy_exp_f64_v2, f64, |a| a.exp());
    mock_unary_v2!(musapy_log_f32_v2, f32, |a| a.ln());
    mock_unary_v2!(musapy_log_f64_v2, f64, |a| a.ln());
    mock_unary_v2!(musapy_abs_f32_v2, f32, |a| a.abs());
    mock_unary_v2!(musapy_abs_f64_v2, f64, |a| a.abs());
    mock_unary_v2!(musapy_sign_f32_v2, f32, |a| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 });
    mock_unary_v2!(musapy_sign_f64_v2, f64, |a| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 });
    mock_unary_v2!(musapy_neg_f32_v2, f32, |a| -a);
    mock_unary_v2!(musapy_neg_f64_v2, f64, |a| -a);

    // v2 Clamp
    pub unsafe fn musapy_clamp_f32_v2(a: *const f32, c: *mut f32, lo: f32, hi: f32, ndim: i32, shape: *const usize, a_strides: *const isize, _s: musaStream_t) {
        if a.is_null() || c.is_null() || ndim < 0 { return; }
        let ndim = ndim as usize;
        let shape_s = std::slice::from_raw_parts(shape, ndim);
        let as_s = std::slice::from_raw_parts(a_strides, ndim);
        let n: usize = shape_s.iter().product();
        for idx in 0..n {
            let ao = mock_offset_nd(idx, shape_s, as_s);
            let v = *a.add(ao);
            *c.add(idx) = v.max(lo).min(hi);
        }
    }
    pub unsafe fn musapy_clamp_f64_v2(a: *const f64, c: *mut f64, lo: f64, hi: f64, ndim: i32, shape: *const usize, a_strides: *const isize, _s: musaStream_t) {
        if a.is_null() || c.is_null() || ndim < 0 { return; }
        let ndim = ndim as usize;
        let shape_s = std::slice::from_raw_parts(shape, ndim);
        let as_s = std::slice::from_raw_parts(a_strides, ndim);
        let n: usize = shape_s.iter().product();
        for idx in 0..n {
            let ao = mock_offset_nd(idx, shape_s, as_s);
            let v = *a.add(ao);
            *c.add(idx) = v.max(lo).min(hi);
        }
    }

    // v2 Cast → f32
    mock_cast_v2!(musapy_cast_i8_f32_v2, i8, f32);
    mock_cast_v2!(musapy_cast_i16_f32_v2, i16, f32);
    mock_cast_v2!(musapy_cast_i32_f32_v2, i32, f32);
    mock_cast_v2!(musapy_cast_i64_f32_v2, i64, f32);
    mock_cast_v2!(musapy_cast_u8_f32_v2, u8, f32);
    mock_cast_v2!(musapy_cast_u16_f32_v2, u16, f32);
    mock_cast_v2!(musapy_cast_u32_f32_v2, u32, f32);
    mock_cast_v2!(musapy_cast_u64_f32_v2, u64, f32);
    mock_cast_v2!(musapy_cast_f64_f32_v2, f64, f32);

    // v2 Cast → f64
    mock_cast_v2!(musapy_cast_i8_f64_v2, i8, f64);
    mock_cast_v2!(musapy_cast_i16_f64_v2, i16, f64);
    mock_cast_v2!(musapy_cast_i32_f64_v2, i32, f64);
    mock_cast_v2!(musapy_cast_i64_f64_v2, i64, f64);
    mock_cast_v2!(musapy_cast_u8_f64_v2, u8, f64);
    mock_cast_v2!(musapy_cast_u16_f64_v2, u16, f64);
    mock_cast_v2!(musapy_cast_u32_f64_v2, u32, f64);
    mock_cast_v2!(musapy_cast_u64_f64_v2, u64, f64);
    mock_cast_v2!(musapy_cast_f32_f64_v2, f32, f64);
}

// Mock 模式 re-export
#[cfg(musapy_mock_musa)]
pub use mock::*;
