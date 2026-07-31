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

    // v2 Cast → i64（Phase 4 reduction 整数累加用）
    pub fn musapy_cast_i8_i64_v2(a: *const i8, c: *mut i64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_i16_i64_v2(a: *const i16, c: *mut i64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_i32_i64_v2(a: *const i32, c: *mut i64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u8_i64_v2(a: *const u8, c: *mut i64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u16_i64_v2(a: *const u16, c: *mut i64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u32_i64_v2(a: *const u32, c: *mut i64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);
    pub fn musapy_cast_u64_i64_v2(a: *const u64, c: *mut i64, ndim: i32, shape: *const usize, a_strides: *const isize, stream: musaStream_t);

    // v2 Comparison（Phase 3：输入 T，输出 u8/bool）
    pub fn musapy_eq_f32_v2(a: *const f32, b: *const f32, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_eq_f64_v2(a: *const f64, b: *const f64, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_ne_f32_v2(a: *const f32, b: *const f32, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_ne_f64_v2(a: *const f64, b: *const f64, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_lt_f32_v2(a: *const f32, b: *const f32, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_lt_f64_v2(a: *const f64, b: *const f64, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_gt_f32_v2(a: *const f32, b: *const f32, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_gt_f64_v2(a: *const f64, b: *const f64, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_le_f32_v2(a: *const f32, b: *const f32, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_le_f64_v2(a: *const f64, b: *const f64, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_ge_f32_v2(a: *const f32, b: *const f32, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);
    pub fn musapy_ge_f64_v2(a: *const f64, b: *const f64, c: *mut u8, ndim: i32, shape: *const usize, a_strides: *const isize, b_strides: *const isize, stream: musaStream_t);

    // v2 Reduction（Phase 4：沿轴缩减）
    pub fn musapy_sum_i64_v2(a: *const i64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_sum_f32_v2(a: *const f32, c: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_sum_f64_v2(a: *const f64, c: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_prod_i64_v2(a: *const i64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_prod_f32_v2(a: *const f32, c: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_prod_f64_v2(a: *const f64, c: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_max_i64_v2(a: *const i64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_max_f32_v2(a: *const f32, c: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_max_f64_v2(a: *const f64, c: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_min_i64_v2(a: *const i64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_min_f32_v2(a: *const f32, c: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_min_f64_v2(a: *const f64, c: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_mean_f32_v2(a: *const f32, c: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_mean_f64_v2(a: *const f64, c: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmax_i64_v2(a: *const i64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmax_f32_v2(a: *const f32, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmax_f64_v2(a: *const f64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmin_i64_v2(a: *const i64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmin_f32_v2(a: *const f32, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmin_f64_v2(a: *const f64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_cumsum_i64_v2(a: *const i64, c: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, out_size: usize, stream: musaStream_t);
    pub fn musapy_cumsum_f32_v2(a: *const f32, c: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, out_size: usize, stream: musaStream_t);
    pub fn musapy_cumsum_f64_v2(a: *const f64, c: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, out_size: usize, stream: musaStream_t);
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

    // v2 Cast → i64（Phase 4）
    mock_cast_v2!(musapy_cast_i8_i64_v2, i8, i64);
    mock_cast_v2!(musapy_cast_i16_i64_v2, i16, i64);
    mock_cast_v2!(musapy_cast_i32_i64_v2, i32, i64);
    mock_cast_v2!(musapy_cast_u8_i64_v2, u8, i64);
    mock_cast_v2!(musapy_cast_u16_i64_v2, u16, i64);
    mock_cast_v2!(musapy_cast_u32_i64_v2, u32, i64);
    mock_cast_v2!(musapy_cast_u64_i64_v2, u64, i64);

    // v2 Comparison mock（Phase 3）
    macro_rules! mock_compare_v2 {
        ($name:ident, $t:ty, $cmp:expr) => {
            pub unsafe fn $name(
                a: *const $t, b: *const $t, c: *mut u8,
                ndim: i32, shape: *const usize,
                a_strides: *const isize, b_strides: *const isize,
                _stream: musaStream_t,
            ) {
                if a.is_null() || b.is_null() || c.is_null() || ndim < 0 { return; }
                let ndim = ndim as usize;
                let shape_s = std::slice::from_raw_parts(shape, ndim);
                let a_str = std::slice::from_raw_parts(a_strides, ndim);
                let b_str = std::slice::from_raw_parts(b_strides, ndim);
                let n: usize = shape_s.iter().product();
                let cmp: fn($t, $t) -> bool = $cmp;
                for idx in 0..n {
                    let a_off = mock_offset_nd(idx, shape_s, a_str);
                    let b_off = mock_offset_nd(idx, shape_s, b_str);
                    *c.add(idx) = if cmp(*a.add(a_off), *b.add(b_off)) { 1 } else { 0 };
                }
            }
        };
    }

    mock_compare_v2!(musapy_eq_f32_v2, f32, |a, b| a == b);
    mock_compare_v2!(musapy_eq_f64_v2, f64, |a, b| a == b);
    mock_compare_v2!(musapy_ne_f32_v2, f32, |a, b| a != b);
    mock_compare_v2!(musapy_ne_f64_v2, f64, |a, b| a != b);
    mock_compare_v2!(musapy_lt_f32_v2, f32, |a, b| a < b);
    mock_compare_v2!(musapy_lt_f64_v2, f64, |a, b| a < b);
    mock_compare_v2!(musapy_gt_f32_v2, f32, |a, b| a > b);
    mock_compare_v2!(musapy_gt_f64_v2, f64, |a, b| a > b);
    mock_compare_v2!(musapy_le_f32_v2, f32, |a, b| a <= b);
    mock_compare_v2!(musapy_le_f64_v2, f64, |a, b| a <= b);
    mock_compare_v2!(musapy_ge_f32_v2, f32, |a, b| a >= b);
    mock_compare_v2!(musapy_ge_f64_v2, f64, |a, b| a >= b);

    // v2 Reduction mock（Phase 4）
    // 辅助：计算 reduce_input_offset（与 common.h 逻辑一致）
    fn mock_reduce_offset(out_idx: usize, in_shape: &[usize], in_strides: &[isize], axis: usize, k: usize) -> usize {
        let ndim = in_shape.len();
        // 展开 out_idx 到非 axis 维坐标
        let mut coords = [0usize; 32];
        let mut ci = 0;
        let mut tmp = out_idx;
        for i in (0..ndim).rev() {
            if i == axis { continue; }
            coords[ci] = tmp % in_shape[i];
            tmp /= in_shape[i];
            ci += 1;
        }
        // 计算 offset
        let mut offset = 0isize;
        ci = 0;
        for i in (0..ndim).rev() {
            let coord = if i == axis { k } else { let c = coords[ci]; ci += 1; c };
            offset += coord as isize * in_strides[i];
        }
        offset as usize
    }

    // 标准 reduction mock 宏（sum/prod/max/min）
    macro_rules! mock_reduce_v2 {
        ($name:ident, $t:ty, $identity:expr, $accum:expr) => {
            pub unsafe fn $name(
                a: *const $t, c: *mut $t,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, axis_len: usize, out_size: usize,
                _stream: musaStream_t,
            ) {
                if a.is_null() || c.is_null() || ndim <= 0 || out_size == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                let accum: fn($t, $t) -> $t = $accum;
                for idx in 0..out_size {
                    let base = mock_reduce_offset(idx, shape_s, strides_s, axis_u, 0);
                    let axis_stride = strides_s[axis_u];
                    let mut acc: $t = $identity;
                    for k in 0..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        acc = accum(acc, *a.add(off));
                    }
                    *c.add(idx) = acc;
                }
            }
        };
    }

    mock_reduce_v2!(musapy_sum_i64_v2, i64, 0, |acc, v| acc + v);
    mock_reduce_v2!(musapy_sum_f32_v2, f32, 0.0, |acc, v| acc + v);
    mock_reduce_v2!(musapy_sum_f64_v2, f64, 0.0, |acc, v| acc + v);
    mock_reduce_v2!(musapy_prod_i64_v2, i64, 1, |acc, v| acc * v);
    mock_reduce_v2!(musapy_prod_f32_v2, f32, 1.0, |acc, v| acc * v);
    mock_reduce_v2!(musapy_prod_f64_v2, f64, 1.0, |acc, v| acc * v);

    // max/min mock（用第一个元素初始化）
    macro_rules! mock_minmax_v2 {
        ($name:ident, $t:ty, $is_better:expr) => {
            pub unsafe fn $name(
                a: *const $t, c: *mut $t,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, axis_len: usize, out_size: usize,
                _stream: musaStream_t,
            ) {
                if a.is_null() || c.is_null() || ndim <= 0 || out_size == 0 || axis_len == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                let is_better: fn($t, $t) -> bool = $is_better;
                for idx in 0..out_size {
                    let base = mock_reduce_offset(idx, shape_s, strides_s, axis_u, 0);
                    let axis_stride = strides_s[axis_u];
                    let mut acc = *a.add(base);
                    for k in 1..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        let val = *a.add(off);
                        if is_better(val, acc) { acc = val; }
                    }
                    *c.add(idx) = acc;
                }
            }
        };
    }

    mock_minmax_v2!(musapy_max_i64_v2, i64, |v, acc| v > acc);
    mock_minmax_v2!(musapy_max_f32_v2, f32, |v, acc| v > acc);
    mock_minmax_v2!(musapy_max_f64_v2, f64, |v, acc| v > acc);
    mock_minmax_v2!(musapy_min_i64_v2, i64, |v, acc| v < acc);
    mock_minmax_v2!(musapy_min_f32_v2, f32, |v, acc| v < acc);
    mock_minmax_v2!(musapy_min_f64_v2, f64, |v, acc| v < acc);

    // mean mock（只有 f32/f64）
    macro_rules! mock_mean_v2 {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(
                a: *const $t, c: *mut $t,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, axis_len: usize, out_size: usize,
                _stream: musaStream_t,
            ) {
                if a.is_null() || c.is_null() || ndim <= 0 || out_size == 0 || axis_len == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                for idx in 0..out_size {
                    let base = mock_reduce_offset(idx, shape_s, strides_s, axis_u, 0);
                    let axis_stride = strides_s[axis_u];
                    let mut acc: $t = 0.0;
                    for k in 0..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        acc += *a.add(off);
                    }
                    *c.add(idx) = acc / axis_len as $t;
                }
            }
        };
    }

    mock_mean_v2!(musapy_mean_f32_v2, f32);
    mock_mean_v2!(musapy_mean_f64_v2, f64);

    // argmax/argmin mock（输入 T，输出 i64）
    macro_rules! mock_argreduce_v2 {
        ($name:ident, $t:ty, $is_better:expr) => {
            pub unsafe fn $name(
                a: *const $t, c: *mut i64,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, axis_len: usize, out_size: usize,
                _stream: musaStream_t,
            ) {
                if a.is_null() || c.is_null() || ndim <= 0 || out_size == 0 || axis_len == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                let is_better: fn($t, $t) -> bool = $is_better;
                for idx in 0..out_size {
                    let base = mock_reduce_offset(idx, shape_s, strides_s, axis_u, 0);
                    let axis_stride = strides_s[axis_u];
                    let mut best_val = *a.add(base);
                    let mut best_idx: i64 = 0;
                    for k in 1..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        let val = *a.add(off);
                        if is_better(val, best_val) {
                            best_val = val;
                            best_idx = k as i64;
                        }
                    }
                    *c.add(idx) = best_idx;
                }
            }
        };
    }

    mock_argreduce_v2!(musapy_argmax_i64_v2, i64, |v, best| v > best);
    mock_argreduce_v2!(musapy_argmax_f32_v2, f32, |v, best| v > best);
    mock_argreduce_v2!(musapy_argmax_f64_v2, f64, |v, best| v > best);
    mock_argreduce_v2!(musapy_argmin_i64_v2, i64, |v, best| v < best);
    mock_argreduce_v2!(musapy_argmin_f32_v2, f32, |v, best| v < best);
    mock_argreduce_v2!(musapy_argmin_f64_v2, f64, |v, best| v < best);

    // cumsum mock（输出同 shape，prefix sum）
    macro_rules! mock_cumsum_v2 {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(
                a: *const $t, c: *mut $t,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, out_size: usize,
                _stream: musaStream_t,
            ) {
                if a.is_null() || c.is_null() || ndim <= 0 || out_size == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                for idx in 0..out_size {
                    // 展开 idx 得到 axis 坐标
                    let mut tmp = idx;
                    let mut axis_coord = 0usize;
                    for i in (0..ndim_u).rev() {
                        let coord = tmp % shape_s[i];
                        tmp /= shape_s[i];
                        if i == axis_u { axis_coord = coord; }
                    }
                    // 计算 axis=0 时的 base offset
                    let mut base = 0isize;
                    tmp = idx;
                    for i in (0..ndim_u).rev() {
                        let coord = tmp % shape_s[i];
                        tmp /= shape_s[i];
                        if i != axis_u {
                            base += coord as isize * strides_s[i];
                        }
                    }
                    let axis_stride = strides_s[axis_u];
                    let mut acc: $t = 0.0 as $t;
                    for k in 0..=axis_coord {
                        let off = (base + k as isize * axis_stride) as usize;
                        acc += *a.add(off);
                    }
                    *c.add(idx) = acc;
                }
            }
        };
    }

    mock_cumsum_v2!(musapy_cumsum_i64_v2, i64);
    mock_cumsum_v2!(musapy_cumsum_f32_v2, f32);
    mock_cumsum_v2!(musapy_cumsum_f64_v2, f64);
}

// Mock 模式 re-export
#[cfg(musapy_mock_musa)]
pub use mock::*;
