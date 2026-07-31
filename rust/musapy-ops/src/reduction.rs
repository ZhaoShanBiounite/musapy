//! Reduction 公开 API（Phase 4, ADR-002-D3）
//!
//! 8 个缩减算子：sum / prod / max / min / mean / argmax / argmin / cumsum。
//! 支持 axis=None（全局缩减）/ axis=int（单轴），keepdims 保留维度。
//!
//! 内部委托到 `op_builder::reduction_axis` / `op_builder::cumsum_op`。

use crate::op_builder::{self, ReduceKernel};
use musapy_core::error::ShapeError;
use musapy_core::{Array, Result};

/// Axis 归一化：None → 全局；Some(i) → 处理负数 + 越界检查。
pub(crate) fn resolve_axis(axis: Option<isize>, ndim: usize) -> Result<Option<usize>> {
    match axis {
        None => Ok(None),
        Some(ax) => {
            let normalized = if ax < 0 {
                ax + ndim as isize
            } else {
                ax
            };
            if normalized < 0 || normalized >= ndim as isize {
                return Err(ShapeError::Mismatch(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                ))
                .into());
            }
            Ok(Some(normalized as usize))
        }
    }
}

/// `ms.sum(a, axis=None, keepdims=False, out=None)` — 沿轴求和。
///
/// 整数输入以 int64 累加，输出 int64；浮点保持原 dtype。
pub fn sum(a: &Array, axis: Option<isize>, keepdims: bool, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::reduction_axis(a, ax, keepdims, out, ReduceKernel::Sum)
}

/// `ms.prod(a, axis=None, keepdims=False, out=None)` — 沿轴求积。
///
/// 整数输入以 int64 累加，输出 int64；浮点保持原 dtype。
pub fn prod(a: &Array, axis: Option<isize>, keepdims: bool, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::reduction_axis(a, ax, keepdims, out, ReduceKernel::Prod)
}

/// `ms.max(a, axis=None, keepdims=False, out=None)` — 沿轴最大值。
///
/// 整数输入 cast 到 int64 后比较，输出 int64；浮点保持原 dtype。
pub fn max(a: &Array, axis: Option<isize>, keepdims: bool, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::reduction_axis(a, ax, keepdims, out, ReduceKernel::Max)
}

/// `ms.min(a, axis=None, keepdims=False, out=None)` — 沿轴最小值。
///
/// 整数输入 cast 到 int64 后比较，输出 int64；浮点保持原 dtype。
pub fn min(a: &Array, axis: Option<isize>, keepdims: bool, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::reduction_axis(a, ax, keepdims, out, ReduceKernel::Min)
}

/// `ms.mean(a, axis=None, keepdims=False, out=None)` — 沿轴均值。
///
/// 整数输入以 float64 累加，输出 float64；f32 保持 f32，f64 保持 f64。
pub fn mean(a: &Array, axis: Option<isize>, keepdims: bool, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::reduction_axis(a, ax, keepdims, out, ReduceKernel::Mean)
}

/// `ms.argmax(a, axis=None, out=None)` — 沿轴最大值的索引。
///
/// 输出恒为 int64。axis=None 时返回展平后的全局索引。
pub fn argmax(a: &Array, axis: Option<isize>, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::reduction_axis(a, ax, false, out, ReduceKernel::Argmax)
}

/// `ms.argmin(a, axis=None, out=None)` — 沿轴最小值的索引。
///
/// 输出恒为 int64。axis=None 时返回展平后的全局索引。
pub fn argmin(a: &Array, axis: Option<isize>, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::reduction_axis(a, ax, false, out, ReduceKernel::Argmin)
}

/// `ms.cumsum(a, axis=None, out=None)` — 沿轴累积求和（prefix sum）。
///
/// axis=None 时展平为 1D 后 cumsum；axis=int 时保持原 shape。
/// 整数输入以 int64 累加，输出 int64；浮点保持原 dtype。
pub fn cumsum(a: &Array, axis: Option<isize>, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::cumsum_op(a, ax, out)
}
