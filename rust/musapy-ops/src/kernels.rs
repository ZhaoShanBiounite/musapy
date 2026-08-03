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

    // v3 Cumsum — work-efficient 三阶段 prefix sum（含 scratch buffer）
    // 签名：(a, c, tmp, ndim, in_shape, in_strides, axis, axis_len, out_size, stream)
    pub fn musapy_cumsum_i64_v3(a: *const i64, c: *mut i64, tmp: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_cumsum_f32_v3(a: *const f32, c: *mut f32, tmp: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_cumsum_f64_v3(a: *const f64, c: *mut f64, tmp: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, stream: musaStream_t);

    // ── Init/Creation kernels（Phase 5）──
    // 输出始终 C-contiguous，无 stride 参数。

    // Fill（zeros/ones/full 共用）
    pub fn musapy_fill_f32(out: *mut f32, value: f32, n: usize, stream: musaStream_t);
    pub fn musapy_fill_f64(out: *mut f64, value: f64, n: usize, stream: musaStream_t);
    pub fn musapy_fill_i64(out: *mut i64, value: i64, n: usize, stream: musaStream_t);
    pub fn musapy_fill_i32(out: *mut i32, value: i32, n: usize, stream: musaStream_t);
    pub fn musapy_fill_i16(out: *mut i16, value: i16, n: usize, stream: musaStream_t);
    pub fn musapy_fill_i8(out: *mut i8, value: i8, n: usize, stream: musaStream_t);
    pub fn musapy_fill_u64(out: *mut u64, value: u64, n: usize, stream: musaStream_t);
    pub fn musapy_fill_u32(out: *mut u32, value: u32, n: usize, stream: musaStream_t);
    pub fn musapy_fill_u16(out: *mut u16, value: u16, n: usize, stream: musaStream_t);
    pub fn musapy_fill_u8(out: *mut u8, value: u8, n: usize, stream: musaStream_t);

    // Arange
    pub fn musapy_arange_f32(out: *mut f32, start: f32, step: f32, n: usize, stream: musaStream_t);
    pub fn musapy_arange_f64(out: *mut f64, start: f64, step: f64, n: usize, stream: musaStream_t);
    pub fn musapy_arange_i64(out: *mut i64, start: i64, step: i64, n: usize, stream: musaStream_t);
    pub fn musapy_arange_i32(out: *mut i32, start: i32, step: i32, n: usize, stream: musaStream_t);

    // Linspace（仅浮点）
    pub fn musapy_linspace_f32(out: *mut f32, start: f32, stop: f32, n: usize, stream: musaStream_t);
    pub fn musapy_linspace_f64(out: *mut f64, start: f64, stop: f64, n: usize, stream: musaStream_t);

    // Eye
    pub fn musapy_eye_f32(out: *mut f32, n: usize, m: usize, k: i32, stream: musaStream_t);
    pub fn musapy_eye_f64(out: *mut f64, n: usize, m: usize, k: i32, stream: musaStream_t);
    pub fn musapy_eye_i64(out: *mut i64, n: usize, m: usize, k: i32, stream: musaStream_t);
    pub fn musapy_eye_i32(out: *mut i32, n: usize, m: usize, k: i32, stream: musaStream_t);

    // ── Parallel reduction: partial（Phase 1）──
    // 签名：(a, partials, ndim, in_shape, in_strides, axis, axis_len, out_size, tiles_per_output, stream)
    pub fn musapy_sum_partial_i64_v2(a: *const i64, partials: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_sum_partial_f32_v2(a: *const f32, partials: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_sum_partial_f64_v2(a: *const f64, partials: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_prod_partial_i64_v2(a: *const i64, partials: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_prod_partial_f32_v2(a: *const f32, partials: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_prod_partial_f64_v2(a: *const f64, partials: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_max_partial_i64_v2(a: *const i64, partials: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_max_partial_f32_v2(a: *const f32, partials: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_max_partial_f64_v2(a: *const f64, partials: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_min_partial_i64_v2(a: *const i64, partials: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_min_partial_f32_v2(a: *const f32, partials: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_min_partial_f64_v2(a: *const f64, partials: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_mean_partial_i64_v2(a: *const i64, partials: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_mean_partial_f32_v2(a: *const f32, partials: *mut f32, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_mean_partial_f64_v2(a: *const f64, partials: *mut f64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);

    // ── Parallel reduction: final（Phase 2）──
    // 签名：(partials, c, num_partials, out_size, stream)
    pub fn musapy_sum_final_i64_v2(partials: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_sum_final_f32_v2(partials: *const f32, c: *mut f32, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_sum_final_f64_v2(partials: *const f64, c: *mut f64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_prod_final_i64_v2(partials: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_prod_final_f32_v2(partials: *const f32, c: *mut f32, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_prod_final_f64_v2(partials: *const f64, c: *mut f64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_max_final_i64_v2(partials: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_max_final_f32_v2(partials: *const f32, c: *mut f32, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_max_final_f64_v2(partials: *const f64, c: *mut f64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_min_final_i64_v2(partials: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_min_final_f32_v2(partials: *const f32, c: *mut f32, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_min_final_f64_v2(partials: *const f64, c: *mut f64, num_partials: usize, out_size: usize, stream: musaStream_t);

    // mean final 额外需要 axis_len
    pub fn musapy_mean_final_f32_v2(partials: *const f32, c: *mut f32, num_partials: usize, out_size: usize, axis_len: usize, stream: musaStream_t);
    pub fn musapy_mean_final_f64_v2(partials: *const f64, c: *mut f64, num_partials: usize, out_size: usize, axis_len: usize, stream: musaStream_t);

    // ── Argmax/Argmin parallel: partial ──
    // 签名：(a, partials_val, partials_idx, ndim, in_shape, in_strides, axis, axis_len, out_size, tiles_per_output, stream)
    pub fn musapy_argmax_partial_i64_v2(a: *const i64, partials_val: *mut i64, partials_idx: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_argmax_partial_f32_v2(a: *const f32, partials_val: *mut f32, partials_idx: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_argmax_partial_f64_v2(a: *const f64, partials_val: *mut f64, partials_idx: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_argmin_partial_i64_v2(a: *const i64, partials_val: *mut i64, partials_idx: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_argmin_partial_f32_v2(a: *const f32, partials_val: *mut f32, partials_idx: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);
    pub fn musapy_argmin_partial_f64_v2(a: *const f64, partials_val: *mut f64, partials_idx: *mut i64, ndim: i32, in_shape: *const usize, in_strides: *const isize, axis: i32, axis_len: usize, out_size: usize, tiles_per_output: usize, stream: musaStream_t);

    // ── Argmax/Argmin parallel: final ──
    // 签名：(partials_val, partials_idx, c, num_partials, out_size, stream)
    pub fn musapy_argmax_final_i64_v2(partials_val: *const i64, partials_idx: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmax_final_f32_v2(partials_val: *const f32, partials_idx: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmax_final_f64_v2(partials_val: *const f64, partials_idx: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmin_final_i64_v2(partials_val: *const i64, partials_idx: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmin_final_f32_v2(partials_val: *const f32, partials_idx: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);
    pub fn musapy_argmin_final_f64_v2(partials_val: *const f64, partials_idx: *const i64, c: *mut i64, num_partials: usize, out_size: usize, stream: musaStream_t);

    // ── Phase 6 indexing: gather / scatter / copy ──
    // gather: out[idx] = input[axis 维坐标经 indices 映射后的偏移]
    pub fn musapy_gather_f32(input: *const f32, output: *mut f32, indices: *const i64, ndim: i32, axis: i32, out_shape: *const usize, in_strides: *const isize, n_out: usize, stream: musaStream_t);
    pub fn musapy_gather_f64(input: *const f64, output: *mut f64, indices: *const i64, ndim: i32, axis: i32, out_shape: *const usize, in_strides: *const isize, n_out: usize, stream: musaStream_t);
    pub fn musapy_gather_i32(input: *const i32, output: *mut i32, indices: *const i64, ndim: i32, axis: i32, out_shape: *const usize, in_strides: *const isize, n_out: usize, stream: musaStream_t);
    pub fn musapy_gather_i64(input: *const i64, output: *mut i64, indices: *const i64, ndim: i32, axis: i32, out_shape: *const usize, in_strides: *const isize, n_out: usize, stream: musaStream_t);

    // scatter: output[连续偏移] = values[idx]（output 为连续布局）
    pub fn musapy_scatter_f32(output: *mut f32, values: *const f32, indices: *const i64, ndim: i32, axis: i32, val_shape: *const usize, val_strides: *const isize, out_strides: *const usize, n_values: usize, stream: musaStream_t);
    pub fn musapy_scatter_f64(output: *mut f64, values: *const f64, indices: *const i64, ndim: i32, axis: i32, val_shape: *const usize, val_strides: *const isize, out_strides: *const usize, n_values: usize, stream: musaStream_t);
    pub fn musapy_scatter_i32(output: *mut i32, values: *const i32, indices: *const i64, ndim: i32, axis: i32, val_shape: *const usize, val_strides: *const isize, out_strides: *const usize, n_values: usize, stream: musaStream_t);
    pub fn musapy_scatter_i64(output: *mut i64, values: *const i64, indices: *const i64, ndim: i32, axis: i32, val_shape: *const usize, val_strides: *const isize, out_strides: *const usize, n_values: usize, stream: musaStream_t);

    // copy：stride-aware identity（视图物化为连续布局）
    pub fn musapy_copy_f32(input: *const f32, output: *mut f32, ndim: i32, shape: *const usize, in_strides: *const isize, stream: musaStream_t);
    pub fn musapy_copy_f64(input: *const f64, output: *mut f64, ndim: i32, shape: *const usize, in_strides: *const isize, stream: musaStream_t);
    pub fn musapy_copy_i32(input: *const i32, output: *mut i32, ndim: i32, shape: *const usize, in_strides: *const isize, stream: musaStream_t);
    pub fn musapy_copy_i64(input: *const i64, output: *mut i64, ndim: i32, shape: *const usize, in_strides: *const isize, stream: musaStream_t);
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

    // v3 cumsum mock — 与 v2 同算法（逐行 inclusive prefix sum），忽略 scratch buffer。
    // mock 模式只求正确性，性能不要求；复用 v2 的实现即可。
    macro_rules! mock_cumsum_v3 {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(
                a: *const $t, c: *mut $t, _tmp: *mut $t,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, axis_len: usize, out_size: usize,
                _stream: musaStream_t,
            ) {
                if a.is_null() || c.is_null() || ndim <= 0 || out_size == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                for idx in 0..out_size {
                    let mut tmp = idx;
                    let mut axis_coord = 0usize;
                    for i in (0..ndim_u).rev() {
                        let coord = tmp % shape_s[i];
                        tmp /= shape_s[i];
                        if i == axis_u { axis_coord = coord; }
                    }
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
                let _ = axis_len; // 与 v3 签名保持一致（未使用）
            }
        };
    }

    mock_cumsum_v3!(musapy_cumsum_i64_v3, i64);
    mock_cumsum_v3!(musapy_cumsum_f32_v3, f32);
    mock_cumsum_v3!(musapy_cumsum_f64_v3, f64);

    // ── Parallel reduction mock（Phase B）──

    // Partial mock 宏（sum/prod/max/min）：将 axis 分成 tiles_per_output 段，每段独立缩减
    macro_rules! mock_reduce_partial_v2 {
        ($name:ident, $t:ty, $identity:expr, $accum:expr) => {
            pub unsafe fn $name(
                a: *const $t, partials: *mut $t,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, axis_len: usize, out_size: usize,
                tiles_per_output: usize, _stream: musaStream_t,
            ) {
                if a.is_null() || partials.is_null() || ndim <= 0 || out_size == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                let accum: fn($t, $t) -> $t = $accum;
                let chunk = (axis_len + tiles_per_output - 1) / tiles_per_output;
                for out_idx in 0..out_size {
                    let base = mock_reduce_offset(out_idx, shape_s, strides_s, axis_u, 0);
                    let axis_stride = strides_s[axis_u];
                    for tile in 0..tiles_per_output {
                        let start = tile * chunk;
                        let end = (start + chunk).min(axis_len);
                        let mut acc: $t = $identity;
                        for k in start..end {
                            let off = (base as isize + k as isize * axis_stride) as usize;
                            acc = accum(acc, *a.add(off));
                        }
                        *partials.add(out_idx * tiles_per_output + tile) = acc;
                    }
                }
            }
        };
    }

    mock_reduce_partial_v2!(musapy_sum_partial_i64_v2, i64, 0, |acc, v| acc + v);
    mock_reduce_partial_v2!(musapy_sum_partial_f32_v2, f32, 0.0, |acc, v| acc + v);
    mock_reduce_partial_v2!(musapy_sum_partial_f64_v2, f64, 0.0, |acc, v| acc + v);
    mock_reduce_partial_v2!(musapy_prod_partial_i64_v2, i64, 1, |acc, v| acc * v);
    mock_reduce_partial_v2!(musapy_prod_partial_f32_v2, f32, 1.0, |acc, v| acc * v);
    mock_reduce_partial_v2!(musapy_prod_partial_f64_v2, f64, 1.0, |acc, v| acc * v);

    // max/min partial mock（用第一个元素初始化）
    macro_rules! mock_minmax_partial_v2 {
        ($name:ident, $t:ty, $is_better:expr) => {
            pub unsafe fn $name(
                a: *const $t, partials: *mut $t,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, axis_len: usize, out_size: usize,
                tiles_per_output: usize, _stream: musaStream_t,
            ) {
                if a.is_null() || partials.is_null() || ndim <= 0 || out_size == 0 || axis_len == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                let is_better: fn($t, $t) -> bool = $is_better;
                let chunk = (axis_len + tiles_per_output - 1) / tiles_per_output;
                for out_idx in 0..out_size {
                    let base = mock_reduce_offset(out_idx, shape_s, strides_s, axis_u, 0);
                    let axis_stride = strides_s[axis_u];
                    for tile in 0..tiles_per_output {
                        let start = tile * chunk;
                        let end = (start + chunk).min(axis_len);
                        if start >= axis_len {
                            // 空 tile：写入 identity（不影响 final）
                            *partials.add(out_idx * tiles_per_output + tile) = *a.add(base);
                            continue;
                        }
                        let first_off = (base as isize + start as isize * axis_stride) as usize;
                        let mut acc = *a.add(first_off);
                        for k in (start + 1)..end {
                            let off = (base as isize + k as isize * axis_stride) as usize;
                            let val = *a.add(off);
                            if is_better(val, acc) { acc = val; }
                        }
                        *partials.add(out_idx * tiles_per_output + tile) = acc;
                    }
                }
            }
        };
    }

    mock_minmax_partial_v2!(musapy_max_partial_i64_v2, i64, |v, acc| v > acc);
    mock_minmax_partial_v2!(musapy_max_partial_f32_v2, f32, |v, acc| v > acc);
    mock_minmax_partial_v2!(musapy_max_partial_f64_v2, f64, |v, acc| v > acc);
    mock_minmax_partial_v2!(musapy_min_partial_i64_v2, i64, |v, acc| v < acc);
    mock_minmax_partial_v2!(musapy_min_partial_f32_v2, f32, |v, acc| v < acc);
    mock_minmax_partial_v2!(musapy_min_partial_f64_v2, f64, |v, acc| v < acc);

    // mean partial mock（只做 sum，final 再除）
    mock_reduce_partial_v2!(musapy_mean_partial_i64_v2, i64, 0, |acc, v| acc + v);
    mock_reduce_partial_v2!(musapy_mean_partial_f32_v2, f32, 0.0, |acc, v| acc + v);
    mock_reduce_partial_v2!(musapy_mean_partial_f64_v2, f64, 0.0, |acc, v| acc + v);

    // Final mock 宏（sum/prod/max/min）：缩减 partials → 最终输出
    macro_rules! mock_reduce_final_v2 {
        ($name:ident, $t:ty, $identity:expr, $accum:expr) => {
            pub unsafe fn $name(
                partials: *const $t, c: *mut $t,
                num_partials: usize, out_size: usize, _stream: musaStream_t,
            ) {
                if partials.is_null() || c.is_null() || out_size == 0 { return; }
                let accum: fn($t, $t) -> $t = $accum;
                for out_idx in 0..out_size {
                    let mut acc: $t = $identity;
                    for i in 0..num_partials {
                        acc = accum(acc, *partials.add(out_idx * num_partials + i));
                    }
                    *c.add(out_idx) = acc;
                }
            }
        };
    }

    mock_reduce_final_v2!(musapy_sum_final_i64_v2, i64, 0, |acc, v| acc + v);
    mock_reduce_final_v2!(musapy_sum_final_f32_v2, f32, 0.0, |acc, v| acc + v);
    mock_reduce_final_v2!(musapy_sum_final_f64_v2, f64, 0.0, |acc, v| acc + v);
    mock_reduce_final_v2!(musapy_prod_final_i64_v2, i64, 1, |acc, v| acc * v);
    mock_reduce_final_v2!(musapy_prod_final_f32_v2, f32, 1.0, |acc, v| acc * v);
    mock_reduce_final_v2!(musapy_prod_final_f64_v2, f64, 1.0, |acc, v| acc * v);

    // max/min final mock
    macro_rules! mock_minmax_final_v2 {
        ($name:ident, $t:ty, $is_better:expr) => {
            pub unsafe fn $name(
                partials: *const $t, c: *mut $t,
                num_partials: usize, out_size: usize, _stream: musaStream_t,
            ) {
                if partials.is_null() || c.is_null() || out_size == 0 || num_partials == 0 { return; }
                let is_better: fn($t, $t) -> bool = $is_better;
                for out_idx in 0..out_size {
                    let mut acc = *partials.add(out_idx * num_partials);
                    for i in 1..num_partials {
                        let val = *partials.add(out_idx * num_partials + i);
                        if is_better(val, acc) { acc = val; }
                    }
                    *c.add(out_idx) = acc;
                }
            }
        };
    }

    mock_minmax_final_v2!(musapy_max_final_i64_v2, i64, |v, acc| v > acc);
    mock_minmax_final_v2!(musapy_max_final_f32_v2, f32, |v, acc| v > acc);
    mock_minmax_final_v2!(musapy_max_final_f64_v2, f64, |v, acc| v > acc);
    mock_minmax_final_v2!(musapy_min_final_i64_v2, i64, |v, acc| v < acc);
    mock_minmax_final_v2!(musapy_min_final_f32_v2, f32, |v, acc| v < acc);
    mock_minmax_final_v2!(musapy_min_final_f64_v2, f64, |v, acc| v < acc);

    // mean final mock（sum partials 再除 axis_len）
    macro_rules! mock_mean_final_v2 {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(
                partials: *const $t, c: *mut $t,
                num_partials: usize, out_size: usize, axis_len: usize, _stream: musaStream_t,
            ) {
                if partials.is_null() || c.is_null() || out_size == 0 || axis_len == 0 { return; }
                for out_idx in 0..out_size {
                    let mut acc: $t = 0.0;
                    for i in 0..num_partials {
                        acc += *partials.add(out_idx * num_partials + i);
                    }
                    *c.add(out_idx) = acc / axis_len as $t;
                }
            }
        };
    }

    mock_mean_final_v2!(musapy_mean_final_f32_v2, f32);
    mock_mean_final_v2!(musapy_mean_final_f64_v2, f64);

    // Argmax/Argmin partial mock
    macro_rules! mock_argreduce_partial_v2 {
        ($name:ident, $t:ty, $is_better:expr) => {
            pub unsafe fn $name(
                a: *const $t, partials_val: *mut $t, partials_idx: *mut i64,
                ndim: i32, in_shape: *const usize, in_strides: *const isize,
                axis: i32, axis_len: usize, out_size: usize,
                tiles_per_output: usize, _stream: musaStream_t,
            ) {
                if a.is_null() || partials_val.is_null() || partials_idx.is_null() || ndim <= 0 || out_size == 0 || axis_len == 0 { return; }
                let ndim_u = ndim as usize;
                let shape_s = std::slice::from_raw_parts(in_shape, ndim_u);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim_u);
                let axis_u = axis as usize;
                let is_better: fn($t, $t) -> bool = $is_better;
                let chunk = (axis_len + tiles_per_output - 1) / tiles_per_output;
                for out_idx in 0..out_size {
                    let base = mock_reduce_offset(out_idx, shape_s, strides_s, axis_u, 0);
                    let axis_stride = strides_s[axis_u];
                    for tile in 0..tiles_per_output {
                        let start = tile * chunk;
                        let end = (start + chunk).min(axis_len);
                        if start >= axis_len {
                            // 空 tile
                            let off = (base as isize) as usize;
                            *partials_val.add(out_idx * tiles_per_output + tile) = *a.add(off);
                            *partials_idx.add(out_idx * tiles_per_output + tile) = 0;
                            continue;
                        }
                        let first_off = (base as isize + start as isize * axis_stride) as usize;
                        let mut best_val = *a.add(first_off);
                        let mut best_idx = start as i64;
                        for k in (start + 1)..end {
                            let off = (base as isize + k as isize * axis_stride) as usize;
                            let val = *a.add(off);
                            if is_better(val, best_val) {
                                best_val = val;
                                best_idx = k as i64;
                            }
                        }
                        *partials_val.add(out_idx * tiles_per_output + tile) = best_val;
                        *partials_idx.add(out_idx * tiles_per_output + tile) = best_idx;
                    }
                }
            }
        };
    }

    mock_argreduce_partial_v2!(musapy_argmax_partial_i64_v2, i64, |v, best| v > best);
    mock_argreduce_partial_v2!(musapy_argmax_partial_f32_v2, f32, |v, best| v > best);
    mock_argreduce_partial_v2!(musapy_argmax_partial_f64_v2, f64, |v, best| v > best);
    mock_argreduce_partial_v2!(musapy_argmin_partial_i64_v2, i64, |v, best| v < best);
    mock_argreduce_partial_v2!(musapy_argmin_partial_f32_v2, f32, |v, best| v < best);
    mock_argreduce_partial_v2!(musapy_argmin_partial_f64_v2, f64, |v, best| v < best);

    // Argmax/Argmin final mock
    macro_rules! mock_argreduce_final_v2 {
        ($name:ident, $t:ty, $is_better:expr) => {
            pub unsafe fn $name(
                partials_val: *const $t, partials_idx: *const i64, c: *mut i64,
                num_partials: usize, out_size: usize, _stream: musaStream_t,
            ) {
                if partials_val.is_null() || partials_idx.is_null() || c.is_null() || out_size == 0 || num_partials == 0 { return; }
                let is_better: fn($t, $t) -> bool = $is_better;
                for out_idx in 0..out_size {
                    let base = out_idx * num_partials;
                    let mut best_val = *partials_val.add(base);
                    let mut best_idx = *partials_idx.add(base);
                    for i in 1..num_partials {
                        let val = *partials_val.add(base + i);
                        if is_better(val, best_val) {
                            best_val = val;
                            best_idx = *partials_idx.add(base + i);
                        }
                    }
                    *c.add(out_idx) = best_idx;
                }
            }
        };
    }

    mock_argreduce_final_v2!(musapy_argmax_final_i64_v2, i64, |v, best| v > best);
    mock_argreduce_final_v2!(musapy_argmax_final_f32_v2, f32, |v, best| v > best);
    mock_argreduce_final_v2!(musapy_argmax_final_f64_v2, f64, |v, best| v > best);
    mock_argreduce_final_v2!(musapy_argmin_final_i64_v2, i64, |v, best| v < best);
    mock_argreduce_final_v2!(musapy_argmin_final_f32_v2, f32, |v, best| v < best);
    mock_argreduce_final_v2!(musapy_argmin_final_f64_v2, f64, |v, best| v < best);

    // ── Init/Creation kernel mock（Phase 5）──

    macro_rules! mock_fill {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(out: *mut $t, value: $t, n: usize, _stream: musaStream_t) {
                if out.is_null() || n == 0 { return; }
                for i in 0..n { *out.add(i) = value; }
            }
        };
    }

    mock_fill!(musapy_fill_f32, f32);
    mock_fill!(musapy_fill_f64, f64);
    mock_fill!(musapy_fill_i64, i64);
    mock_fill!(musapy_fill_i32, i32);
    mock_fill!(musapy_fill_i16, i16);
    mock_fill!(musapy_fill_i8, i8);
    mock_fill!(musapy_fill_u64, u64);
    mock_fill!(musapy_fill_u32, u32);
    mock_fill!(musapy_fill_u16, u16);
    mock_fill!(musapy_fill_u8, u8);

    macro_rules! mock_arange {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(out: *mut $t, start: $t, step: $t, n: usize, _stream: musaStream_t) {
                if out.is_null() || n == 0 { return; }
                for i in 0..n { *out.add(i) = start + (i as $t) * step; }
            }
        };
    }

    mock_arange!(musapy_arange_f32, f32);
    mock_arange!(musapy_arange_f64, f64);
    mock_arange!(musapy_arange_i64, i64);
    mock_arange!(musapy_arange_i32, i32);

    macro_rules! mock_linspace {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(out: *mut $t, start: $t, stop: $t, n: usize, _stream: musaStream_t) {
                if out.is_null() || n == 0 { return; }
                if n == 1 {
                    *out.add(0) = start;
                    return;
                }
                let step = (stop - start) / ((n - 1) as $t);
                for i in 0..n { *out.add(i) = start + (i as $t) * step; }
            }
        };
    }

    mock_linspace!(musapy_linspace_f32, f32);
    mock_linspace!(musapy_linspace_f64, f64);

    macro_rules! mock_eye {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(out: *mut $t, n: usize, m: usize, k: i32, _stream: musaStream_t) {
                if out.is_null() || n == 0 || m == 0 { return; }
                let total = n * m;
                for idx in 0..total {
                    let row = idx / m;
                    let col = idx % m;
                    *out.add(idx) = if (col as i32 - row as i32) == k { 1 as $t } else { 0 as $t };
                }
            }
        };
    }

    mock_eye!(musapy_eye_f32, f32);
    mock_eye!(musapy_eye_f64, f64);
    mock_eye!(musapy_eye_i64, i64);
    mock_eye!(musapy_eye_i32, i32);

    // ── Phase 6 indexing: gather / scatter / copy ──

    macro_rules! mock_gather {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(
                input: *const $t, output: *mut $t, indices: *const i64,
                ndim: i32, axis: i32, out_shape: *const usize, in_strides: *const isize,
                n_out: usize, _stream: musaStream_t,
            ) {
                if input.is_null() || output.is_null() || indices.is_null() || ndim <= 0 { return; }
                let ndim = ndim as usize;
                let axis = axis as usize;
                let shape_s = std::slice::from_raw_parts(out_shape, ndim);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim);
                for idx in 0..n_out {
                    let mut tmp = idx;
                    let mut off = 0isize;
                    for i in (0..ndim).rev() {
                        let coord = tmp % shape_s[i];
                        tmp /= shape_s[i];
                        let k = if i == axis { *indices.add(coord) as usize } else { coord };
                        off += k as isize * strides_s[i];
                    }
                    *output.add(idx) = *input.add(off as usize);
                }
            }
        };
    }

    macro_rules! mock_scatter {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(
                output: *mut $t, values: *const $t, indices: *const i64,
                ndim: i32, axis: i32, val_shape: *const usize, val_strides: *const isize,
                out_strides: *const usize, n_values: usize, _stream: musaStream_t,
            ) {
                if output.is_null() || values.is_null() || indices.is_null() || ndim <= 0 { return; }
                let ndim = ndim as usize;
                let axis = axis as usize;
                let val_shape_s = std::slice::from_raw_parts(val_shape, ndim);
                let val_strides_s = std::slice::from_raw_parts(val_strides, ndim);
                let out_strides_s = std::slice::from_raw_parts(out_strides, ndim);
                for idx in 0..n_values {
                    let mut tmp = idx;
                    let mut out_off = 0usize;
                    let mut val_off = 0isize;
                    for i in (0..ndim).rev() {
                        let coord = tmp % val_shape_s[i];
                        tmp /= val_shape_s[i];
                        val_off += coord as isize * val_strides_s[i];
                        let k = if i == axis { *indices.add(coord) as usize } else { coord };
                        out_off += k * out_strides_s[i];
                    }
                    *output.add(out_off) = *values.add(val_off as usize);
                }
            }
        };
    }

    macro_rules! mock_copy {
        ($name:ident, $t:ty) => {
            pub unsafe fn $name(
                input: *const $t, output: *mut $t,
                ndim: i32, shape: *const usize, in_strides: *const isize,
                _stream: musaStream_t,
            ) {
                if input.is_null() || output.is_null() || ndim < 0 { return; }
                let ndim = ndim as usize;
                let shape_s = std::slice::from_raw_parts(shape, ndim);
                let strides_s = std::slice::from_raw_parts(in_strides, ndim);
                let n: usize = shape_s.iter().product();
                for idx in 0..n {
                    let off = mock_offset_nd(idx, shape_s, strides_s);
                    *output.add(idx) = *input.add(off);
                }
            }
        };
    }

    mock_gather!(musapy_gather_f32, f32);
    mock_gather!(musapy_gather_f64, f64);
    mock_gather!(musapy_gather_i32, i32);
    mock_gather!(musapy_gather_i64, i64);
    mock_scatter!(musapy_scatter_f32, f32);
    mock_scatter!(musapy_scatter_f64, f64);
    mock_scatter!(musapy_scatter_i32, i32);
    mock_scatter!(musapy_scatter_i64, i64);
    mock_copy!(musapy_copy_f32, f32);
    mock_copy!(musapy_copy_f64, f64);
    mock_copy!(musapy_copy_i32, i32);
    mock_copy!(musapy_copy_i64, i64);
}

// Mock 模式 re-export
#[cfg(musapy_mock_musa)]
pub use mock::*;
