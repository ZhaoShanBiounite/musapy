//! Elementwise 算子公开 API（v0.2-alpha, Phase 2）
//!
//! 所有算子经 OpBuilder 3-phase 骨架执行（ADR L2-4, L2-5）：
//!   参数解析（capture-safe）→ kernel launch（可重放）→ 事件/OpContext 后处理。
//!
//! Binary（add/sub/mul/div/pow）：
//!   device 匹配 → NumPy 广播（ADR-002-D2）→ 类型提升（ADR L1-5, L1-14）→
//!   计算白名单（float32/float64）→ 输入按需内部 cast → stride-aware _v2 kernel。
//!
//! Unary（sin/cos/exp/log/abs/sign）与 clamp：
//!   单输入，输出 shape = 输入 shape，dtype 白名单 float32/float64，stride-aware。
//!
//! astype：显式 dtype 转换（目标白名单 float32/float64；同 dtype = 深拷贝）。

use crate::op_builder::{self, BinaryKernel, UnaryKernel};
use musapy_core::{Array, Dtype, Result};

// ── Binary ops（类型提升 + 广播）──────────────────────────────

/// `ms.add(a, b, out=None)` — 逐元素加法（stride-aware, broadcast, 类型提升）。
///
/// 支持 NumPy 广播规则（ADR-002-D2）：
/// - `(3,1) + (4,)` → `(3,4)`
/// - 0-dim + 任意 shape → 任意 shape
///
/// 类型提升（ADR L1-5, L1-14）：结果 dtype 由 `promote(a, b, all_gpu)` 决定，
/// 必须落在 float32/float64 计算白名单；输入 dtype 不一致时内部自动 cast。
///
/// - 无 `out=`：分配新 Buffer，在 a 的 stream（或 stream context）上执行
/// - 有 `out=`：写入 out 的 Buffer，在 out 的 stream 上执行（ADR L1-8）
///
/// 别名检测（ADR L2-5）：out 不能同时是输入。
pub fn add(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::binary_elementwise(a, b, out, BinaryKernel::Add)
}

/// `ms.sub(a, b, out=None)` — 逐元素减法（stride-aware, broadcast, 类型提升）。
pub fn sub(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::binary_elementwise(a, b, out, BinaryKernel::Sub)
}

/// `ms.mul(a, b, out=None)` — 逐元素乘法（stride-aware, broadcast, 类型提升）。
pub fn mul(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::binary_elementwise(a, b, out, BinaryKernel::Mul)
}

/// `ms.div(a, b, out=None)` — 逐元素除法（stride-aware, broadcast, 类型提升）。
///
/// 浮点语义：除零得 inf/NaN（与 IEEE 754 一致，不抛异常）。
pub fn div(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::binary_elementwise(a, b, out, BinaryKernel::Div)
}

/// `ms.pow(a, b, out=None)` — 逐元素幂运算（stride-aware, broadcast, 类型提升）。
///
/// `a ** b`，浮点 powf 语义（负底数非整数指数得 NaN）。
pub fn pow(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::binary_elementwise(a, b, out, BinaryKernel::Pow)
}

// ── Unary ops ─────────────────────────────────────────────────

/// `ms.sin(a, out=None)` — 逐元素正弦（stride-aware，f32/f64）。
pub fn sin(a: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::unary_elementwise(a, out, UnaryKernel::Sin)
}

/// `ms.cos(a, out=None)` — 逐元素余弦（stride-aware，f32/f64）。
pub fn cos(a: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::unary_elementwise(a, out, UnaryKernel::Cos)
}

/// `ms.exp(a, out=None)` — 逐元素自然指数 e^x（stride-aware，f32/f64）。
pub fn exp(a: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::unary_elementwise(a, out, UnaryKernel::Exp)
}

/// `ms.log(a, out=None)` — 逐元素自然对数 ln(x)（stride-aware，f32/f64）。
pub fn log(a: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::unary_elementwise(a, out, UnaryKernel::Log)
}

/// `ms.abs(a, out=None)` — 逐元素绝对值（stride-aware，f32/f64）。
pub fn abs(a: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::unary_elementwise(a, out, UnaryKernel::Abs)
}

/// `ms.sign(a, out=None)` — 逐元素符号函数（stride-aware，f32/f64）。
///
/// `x > 0 → 1.0`，`x < 0 → -1.0`，`x == 0 → 0.0`（NaN 透传）。
pub fn sign(a: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::unary_elementwise(a, out, UnaryKernel::Sign)
}

/// `ms.neg(a)` / `-a` — 逐元素取反（stride-aware，f32/f64）。
pub fn neg(a: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::unary_elementwise(a, out, UnaryKernel::Neg)
}

// ── Clamp ─────────────────────────────────────────────────────

/// `ms.clamp(a, lo, hi, out=None)` — 逐元素截断到 [lo, hi]（stride-aware，f32/f64）。
///
/// `out[i] = min(max(a[i], lo), hi)`。lo/hi 以 f64 传入，按输入 dtype 转换。
pub fn clamp(a: &Array, lo: f64, hi: f64, out: Option<&Array>) -> Result<Array> {
    op_builder::clamp_elementwise(a, lo, hi, out)
}

// ── Cast（公开 astype）────────────────────────────────────────

/// `a.astype(dtype, out=None)` — 显式 dtype 转换。
///
/// 目标 dtype 白名单：float32/float64（Phase 2）；源 dtype 支持
/// int8..int64 / uint8..uint64 / float32 / float64。
/// 同 dtype 调用返回深拷贝（要求连续布局）。
pub fn astype(a: &Array, dtype: Dtype, out: Option<&Array>) -> Result<Array> {
    op_builder::astype_op(a, dtype, out)
}

// ============================================================
// 单元测试（CPU fallback 路径）
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use musapy_core::{
        Buffer, BufferRef, Device, DeviceResolution, DtypeResolution, Layout, ResolutionSource,
        Stream,
    };
    use std::sync::Arc;

    // --- 测试辅助 ---

    fn cpu_array_with_layout(bytes: &[u8], layout: Layout, dtype: Dtype) -> Array {
        let device = Device::Cpu;
        let stream = Arc::new(Stream::new(device.clone(), 0).unwrap());
        let buffer = Buffer::alloc(bytes.len().max(1), device.clone(), &stream).unwrap();
        let buf_arc = Arc::new(buffer);
        let data_ref = BufferRef::new(buf_arc);
        if let Some(ptr) = data_ref.buffer().ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len());
            }
        }
        Array::new(
            data_ref,
            layout,
            dtype,
            stream,
            DeviceResolution::new(device, ResolutionSource::Arg),
            DtypeResolution::new(dtype, ResolutionSource::Arg),
        )
    }

    fn f32_array(vals: &[f32], shape: Vec<usize>) -> Array {
        let bytes =
            unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
        cpu_array_with_layout(bytes, Layout::from_shape(shape), Dtype::Float32)
    }

    fn f64_array(vals: &[f64], shape: Vec<usize>) -> Array {
        let bytes =
            unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 8) };
        cpu_array_with_layout(bytes, Layout::from_shape(shape), Dtype::Float64)
    }

    fn i64_array(vals: &[i64], shape: Vec<usize>) -> Array {
        let bytes =
            unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 8) };
        cpu_array_with_layout(bytes, Layout::from_shape(shape), Dtype::Int64)
    }

    fn read_f32(a: &Array) -> Vec<f32> {
        let n = a.size();
        let mut out = vec![0f32; n];
        if let Some(ptr) = a.data().buffer().ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    n * 4,
                );
            }
        }
        out
    }

    fn read_f64(a: &Array) -> Vec<f64> {
        let n = a.size();
        let mut out = vec![0f64; n];
        if let Some(ptr) = a.data().buffer().ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    n * 8,
                );
            }
        }
        out
    }

    // --- binary：基本运算（CPU fallback）---

    #[test]
    fn binary_add_f32_cpu() {
        let a = f32_array(&[1.0, 2.0, 3.0], vec![3]);
        let b = f32_array(&[4.0, 5.0, 6.0], vec![3]);
        let r = add(&a, &b, None).unwrap();
        assert_eq!(r.dtype(), Dtype::Float32);
        assert_eq!(read_f32(&r), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn binary_sub_mul_div_pow_f32_cpu() {
        let a = f32_array(&[6.0, 8.0], vec![2]);
        let b = f32_array(&[2.0, 4.0], vec![2]);
        assert_eq!(read_f32(&sub(&a, &b, None).unwrap()), vec![4.0, 4.0]);
        assert_eq!(read_f32(&mul(&a, &b, None).unwrap()), vec![12.0, 32.0]);
        assert_eq!(read_f32(&div(&a, &b, None).unwrap()), vec![3.0, 2.0]);

        let x = f32_array(&[2.0, 3.0], vec![2]);
        let e = f32_array(&[2.0, 3.0], vec![2]);
        assert_eq!(read_f32(&pow(&x, &e, None).unwrap()), vec![4.0, 27.0]);
    }

    // --- binary：广播（ADR-002-D2）---

    #[test]
    fn binary_broadcast_3x1_plus_4() {
        let a = f32_array(&[1.0, 2.0, 3.0], vec![3, 1]);
        let b = f32_array(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        let r = add(&a, &b, None).unwrap();
        assert_eq!(r.shape(), &vec![3, 4]);
        assert_eq!(
            read_f32(&r),
            vec![
                11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0
            ]
        );
    }

    #[test]
    fn binary_broadcast_incompatible_errors() {
        let a = f32_array(&[1.0, 2.0, 3.0], vec![3]);
        let b = f32_array(&[1.0, 2.0], vec![2]);
        assert!(add(&a, &b, None).is_err());
    }

    // --- binary：类型提升（ADR L1-5, L1-14）---

    #[test]
    fn binary_promotion_f32_plus_f64_cpu() {
        // CPU（all_gpu=false）走 JAX 表：f32 + f64 → f64
        let a = f32_array(&[1.0, 2.0], vec![2]);
        let b = f64_array(&[0.5, 0.25], vec![2]);
        let r = add(&a, &b, None).unwrap();
        assert_eq!(r.dtype(), Dtype::Float64);
        assert_eq!(read_f64(&r), vec![1.5, 2.25]);
    }

    #[test]
    fn binary_promotion_i32_plus_f32_cpu() {
        // i32 + f32 → f32（JAX：exact + float，float 至少 f32）
        let bytes = unsafe { std::slice::from_raw_parts([1i32, 2, 3].as_ptr() as *const u8, 12) };
        let a = cpu_array_with_layout(bytes, Layout::from_shape(vec![3]), Dtype::Int32);
        let b = f32_array(&[0.5, 0.5, 0.5], vec![3]);
        let r = add(&a, &b, None).unwrap();
        assert_eq!(r.dtype(), Dtype::Float32);
        assert_eq!(read_f32(&r), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn binary_promotion_i64_plus_f32_cpu() {
        // i64 + f32 → f32（JAX：整数不因位宽升级浮点，v0.2 计划 §1.3；
        // 2026-08 修正自 f64）
        let a = i64_array(&[1, 2], vec![2]);
        let b = f32_array(&[0.5, 0.5], vec![2]);
        let r = mul(&a, &b, None).unwrap();
        assert_eq!(r.dtype(), Dtype::Float32);
        assert_eq!(read_f32(&r), vec![0.5, 1.0]);
    }

    #[test]
    fn binary_int_only_rejected_by_whitelist() {
        // i32 + i32 → i32，不在计算白名单（f32/f64）→ 报错
        let bytes = unsafe { std::slice::from_raw_parts([1i32, 2].as_ptr() as *const u8, 8) };
        let a = cpu_array_with_layout(bytes, Layout::from_shape(vec![2]), Dtype::Int32);
        let b = cpu_array_with_layout(bytes, Layout::from_shape(vec![2]), Dtype::Int32);
        assert!(add(&a, &b, None).is_err());
    }

    // --- binary：out= 与别名检测（ADR L2-5）---

    #[test]
    fn binary_out_param_writes_into_out() {
        let a = f32_array(&[1.0, 2.0], vec![2]);
        let b = f32_array(&[3.0, 4.0], vec![2]);
        let out = f32_array(&[0.0, 0.0], vec![2]);
        let r = add(&a, &b, Some(&out)).unwrap();
        assert_eq!(read_f32(&out), vec![4.0, 6.0]);
        assert!(r.data() == out.data()); // 同一 Buffer
    }

    #[test]
    fn binary_out_shape_mismatch_errors() {
        let a = f32_array(&[1.0, 2.0], vec![2]);
        let b = f32_array(&[3.0, 4.0], vec![2]);
        let out = f32_array(&[0.0, 0.0, 0.0], vec![3]);
        assert!(add(&a, &b, Some(&out)).is_err());
    }

    #[test]
    fn binary_alias_detected() {
        let a = f32_array(&[1.0, 2.0], vec![2]);
        let b = f32_array(&[3.0, 4.0], vec![2]);
        assert!(add(&a, &b, Some(&a)).is_err());
    }

    #[test]
    fn binary_device_mismatch_errors() {
        let a = f32_array(&[1.0], vec![1]);
        // Musa 占位数组（placeholder，不分配真实内存）：device 校验先于任何内存访问
        let device = Device::Musa(0);
        let stream = Arc::new(Stream::new(device.clone(), 0).unwrap());
        let buf = Arc::new(Buffer::placeholder(4, device.clone()));
        let b = Array::new(
            BufferRef::new(buf),
            Layout::from_shape(vec![1]),
            Dtype::Float32,
            stream,
            DeviceResolution::new(device, ResolutionSource::Arg),
            DtypeResolution::new(Dtype::Float32, ResolutionSource::Arg),
        );
        assert!(add(&a, &b, None).is_err());
    }

    // --- unary ---

    #[test]
    fn unary_sin_cos_exp_log_cpu() {
        let a = f32_array(&[0.0, std::f32::consts::FRAC_PI_2], vec![2]);
        let s = read_f32(&sin(&a, None).unwrap());
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[1] - 1.0).abs() < 1e-6);

        let c = read_f32(&cos(&a, None).unwrap());
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!(c[1].abs() < 1e-6);

        let e = read_f32(&exp(&f32_array(&[0.0, 1.0], vec![2]), None).unwrap());
        assert!((e[0] - 1.0).abs() < 1e-6);
        assert!((e[1] - std::f32::consts::E).abs() < 1e-5);

        let l = read_f32(&log(&f32_array(&[1.0, std::f32::consts::E], vec![2]), None).unwrap());
        assert!((l[0] - 0.0).abs() < 1e-6);
        assert!((l[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unary_abs_sign_cpu() {
        let a = f32_array(&[-2.5, 0.0, 3.5], vec![3]);
        assert_eq!(read_f32(&abs(&a, None).unwrap()), vec![2.5, 0.0, 3.5]);
        assert_eq!(read_f32(&sign(&a, None).unwrap()), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn unary_f64_cpu() {
        let a = f64_array(&[-1.0, 4.0], vec![2]);
        assert_eq!(read_f64(&abs(&a, None).unwrap()), vec![1.0, 4.0]);
    }

    #[test]
    fn unary_strided_view() {
        // 非连续视图：底层 [1, 99, -2, 99, 3, 99]，shape [3]，strides [2]
        let layout = Layout::from_shape_and_strides(vec![3], vec![2]).unwrap();
        let vals = [1.0f32, 99.0, -2.0, 99.0, 3.0, 99.0];
        let bytes =
            unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
        let a = cpu_array_with_layout(bytes, layout, Dtype::Float32);
        let r = abs(&a, None).unwrap();
        assert_eq!(r.shape(), &vec![3]);
        assert_eq!(read_f32(&r), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn unary_int_rejected_by_whitelist() {
        let a = i64_array(&[1, 2], vec![2]);
        assert!(abs(&a, None).is_err());
    }

    #[test]
    fn unary_out_param() {
        let a = f32_array(&[-1.0, 2.0], vec![2]);
        let out = f32_array(&[0.0, 0.0], vec![2]);
        let r = abs(&a, Some(&out)).unwrap();
        assert_eq!(read_f32(&out), vec![1.0, 2.0]);
        assert!(r.data() == out.data());
    }

    // --- clamp ---

    #[test]
    fn clamp_f32_cpu() {
        let a = f32_array(&[-5.0, 0.5, 9.0], vec![3]);
        let r = clamp(&a, 0.0, 1.0, None).unwrap();
        assert_eq!(read_f32(&r), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn clamp_f64_cpu() {
        let a = f64_array(&[-5.0, 0.5, 9.0], vec![3]);
        let r = clamp(&a, 0.0, 1.0, None).unwrap();
        assert_eq!(read_f64(&r), vec![0.0, 0.5, 1.0]);
    }

    // --- astype / cast ---

    #[test]
    fn astype_i64_to_f32_cpu() {
        let a = i64_array(&[1, -2, 3], vec![3]);
        let r = astype(&a, Dtype::Float32, None).unwrap();
        assert_eq!(r.dtype(), Dtype::Float32);
        assert_eq!(read_f32(&r), vec![1.0, -2.0, 3.0]);
    }

    #[test]
    fn astype_f32_to_f64_cpu() {
        let a = f32_array(&[1.5, -2.5], vec![2]);
        let r = astype(&a, Dtype::Float64, None).unwrap();
        assert_eq!(r.dtype(), Dtype::Float64);
        assert_eq!(read_f64(&r), vec![1.5, -2.5]);
    }

    #[test]
    fn astype_same_dtype_deep_copy_cpu() {
        let a = f32_array(&[7.0, 8.0], vec![2]);
        let r = astype(&a, Dtype::Float32, None).unwrap();
        assert_eq!(read_f32(&r), vec![7.0, 8.0]);
        assert!(r.data() != a.data()); // 深拷贝，新 Buffer
    }

    #[test]
    fn astype_unsupported_target_errors() {
        let a = f32_array(&[1.0], vec![1]);
        assert!(astype(&a, Dtype::Int32, None).is_err());
    }
}
