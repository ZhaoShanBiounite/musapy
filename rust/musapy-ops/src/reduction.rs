//! Reduction 公开 API（Phase 4, ADR-002-D3；Phase 7 扩展 axis=tuple）
//!
//! 8 个缩减算子：sum / prod / max / min / mean / argmax / argmin / cumsum。
//! 支持 axis=None（全局缩减）/ axis=int（单轴）/ axis=tuple（多轴，Phase 7 P7.1）。
//!
//! 内部委托到 `op_builder::reduction_axis` / `op_builder::cumsum_op`。
//!
//! **多轴语义（P7.1，2026-08-08）**：
//!   - sum/prod/max/min/mean：逐轴迭代（正确性优先），排序后从低到高逐轴归约，
//!     全部轮 keepdims=true 保维（轴索引稳定），最后统一 squeeze 被归约轴
//!     （用户 keepdims=false 时）。
//!   - argmax/argmin：多轴用 transpose+合并轴方案（逐轴迭代会丢坐标），
//!     指定轴移到末尾、contiguous 物化、reshape 合并为单轴后走单轴 argreduce，
//!     索引即「展平指定轴」的扁平索引（NumPy 2.0 语义）；arg* 同步补 keepdims。

use crate::op_builder::{self, ReduceKernel};
use musapy_core::error::ShapeError;
use musapy_core::{Array, Result};

/// Axis 归一化：None → 全局；Some(i) → 处理负数 + 越界检查。
pub(crate) fn resolve_axis(axis: Option<isize>, ndim: usize) -> Result<Option<usize>> {
    match axis {
        None => Ok(None),
        Some(ax) => {
            let normalized = if ax < 0 { ax + ndim as isize } else { ax };
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

/// 多轴归一化：负数归一化、去重（重复报错，NumPy 语义）、升序排序。
///
/// 多轴归约结果与轴序无关（NumPy：重复轴报错），统一升序保证逐轴迭代稳定。
pub(crate) fn resolve_axes(axis: Option<&[isize]>, ndim: usize) -> Result<Vec<usize>> {
    let Some(axs) = axis else {
        return Ok(Vec::new());
    };
    let mut out: Vec<usize> = Vec::with_capacity(axs.len());
    for &ax in axs {
        let normalized = if ax < 0 { ax + ndim as isize } else { ax };
        if normalized < 0 || normalized >= ndim as isize {
            return Err(ShapeError::Mismatch(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax, ndim
            ))
            .into());
        }
        out.push(normalized as usize);
    }
    out.sort_unstable();
    for w in out.windows(2) {
        if w[0] == w[1] {
            return Err(ShapeError::Mismatch(format!(
                "duplicate value in 'axis' argument: {}",
                w[0]
            ))
            .into());
        }
    }
    Ok(out)
}

/// `ms.sum(a, axis=None, keepdims=False, out=None)` — 沿轴求和。
///
/// 整数输入以 int64 累加，输出 int64；浮点保持原 dtype。
pub fn sum(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
) -> Result<Array> {
    reduce_multi(a, axis, keepdims, out, ReduceKernel::Sum)
}

/// `ms.prod(a, axis=None, keepdims=False, out=None)` — 沿轴求积。
///
/// 整数输入以 int64 累加，输出 int64；浮点保持原 dtype。
pub fn prod(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
) -> Result<Array> {
    reduce_multi(a, axis, keepdims, out, ReduceKernel::Prod)
}

/// `ms.max(a, axis=None, keepdims=False, out=None)` — 沿轴最大值。
///
/// 整数输入 cast 到 int64 后比较，输出 int64；浮点保持原 dtype。
pub fn max(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
) -> Result<Array> {
    reduce_multi(a, axis, keepdims, out, ReduceKernel::Max)
}

/// `ms.min(a, axis=None, keepdims=False, out=None)` — 沿轴最小值。
///
/// 整数输入 cast 到 int64 后比较，输出 int64；浮点保持原 dtype。
pub fn min(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
) -> Result<Array> {
    reduce_multi(a, axis, keepdims, out, ReduceKernel::Min)
}

/// `ms.mean(a, axis=None, keepdims=False, out=None)` — 沿轴均值。
///
/// 整数输入以 float64 累加，输出 float64；f32 保持 f32，f64 保持 f64。
pub fn mean(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
) -> Result<Array> {
    reduce_multi(a, axis, keepdims, out, ReduceKernel::Mean)
}

/// 通用多轴归约（sum/prod/max/min/mean）：axis=None/单轴/多轴。
///
/// 多轴 → 逐轴迭代（升序），全部轮 keepdims=true 保维（轴索引稳定），
/// 最后若用户 keepdims=false 统一 squeeze 被归约轴。out= 多轴路径暂不支持
/// （中间轮 shape 与最终不同，out 语义复杂；单轴/全局路径正常支持）。
fn reduce_multi(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
    kernel: ReduceKernel,
) -> Result<Array> {
    let axes = resolve_axes(axis, a.ndim())?;
    match axes.len() {
        0 => op_builder::reduction_axis(a, None, keepdims, out, kernel),
        1 => op_builder::reduction_axis(a, Some(axes[0]), keepdims, out, kernel),
        _ => {
            if out.is_some() {
                return Err(ShapeError::Mismatch(
                    "reduction with multiple axes does not support out= yet".into(),
                )
                .into());
            }
            // 逐轴迭代：全部 keepdims=true 保维（axis 索引全程稳定）
            let mut cur: Option<Array> = None;
            for &ax in &axes {
                let input = cur.as_ref().unwrap_or(a);
                cur = Some(op_builder::reduction_axis(
                    input,
                    Some(ax),
                    true,
                    None,
                    kernel,
                )?);
            }
            let cur = cur.expect("non-empty axes");
            if keepdims {
                Ok(cur)
            } else {
                // squeeze 被归约轴（从低到高，每移除一维后续轴索引 -1）
                let mut result = cur;
                for (i, &ax) in axes.iter().enumerate() {
                    result = crate::indexing::index_select(&result, ax - i, 0)?;
                }
                Ok(result)
            }
        }
    }
}

/// `ms.argmax(a, axis=None, keepdims=False, out=None)` — 沿轴最大值的索引。
///
/// 输出恒为 int64。axis=None 时返回展平后的全局索引；多轴返回展平指定轴的扁平索引。
pub fn argmax(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
) -> Result<Array> {
    arg_reduce_multi(a, axis, keepdims, out, ReduceKernel::Argmax)
}

/// `ms.argmin(a, axis=None, keepdims=False, out=None)` — 沿轴最小值的索引。
///
/// 输出恒为 int64。axis=None 时返回展平后的全局索引；多轴返回展平指定轴的扁平索引。
pub fn argmin(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
) -> Result<Array> {
    arg_reduce_multi(a, axis, keepdims, out, ReduceKernel::Argmin)
}

/// argmax/argmin 多轴：transpose+合并轴方案。
///
/// 指定轴移到末尾（升序，保持相对顺序）、contiguous 物化、reshape 合并为
/// 单大轴（axis_len = ∏指定轴），走单轴 argreduce——输出索引即展平指定轴的
/// 扁平索引（NumPy 2.0 语义）。
fn arg_reduce_multi(
    a: &Array,
    axis: Option<&[isize]>,
    keepdims: bool,
    out: Option<&Array>,
    kernel: ReduceKernel,
) -> Result<Array> {
    let axes = resolve_axes(axis, a.ndim())?;
    match axes.len() {
        0 => op_builder::reduction_axis(a, None, keepdims, out, kernel),
        1 => op_builder::reduction_axis(a, Some(axes[0]), keepdims, out, kernel),
        _ => {
            // transpose：非轴维在前（相对顺序），指定轴移到末尾（升序）
            let ndim = a.ndim();
            let mut perm: Vec<usize> = (0..ndim).filter(|i| !axes.contains(i)).collect();
            perm.extend_from_slice(&axes);
            let transposed = crate::indexing::transpose(a, Some(&perm))?;
            // contiguous 物化 + 合并末尾多轴为单轴
            let contig = crate::indexing::contiguous(&transposed)?;
            let merged = crate::indexing::reshape_merge_last(&contig, axes.len())?;
            // 合并轴 = 末尾单轴
            let merged_axis = merged.ndim() - 1;
            let result =
                op_builder::reduction_axis(&merged, Some(merged_axis), keepdims, out, kernel)?;
            if keepdims {
                // keepdims：把结果从 [非轴..., 1] 拆回 [非轴..., 1×len]，
                // 再转置回原始轴序（perm 的逆）——被归约轴处恢复为 1
                let split = crate::indexing::reshape_split_last(&result, axes.len())?;
                // 逆 perm：perm[i] 表示输出轴 i 对应原轴 perm[i]；
                // 逆变换为按原轴序排列的 transpose
                let mut inv_perm = vec![0usize; ndim];
                for (i, &p) in perm.iter().enumerate() {
                    inv_perm[p] = i;
                }
                crate::indexing::transpose(&split, Some(&inv_perm))
            } else {
                Ok(result)
            }
        }
    }
}

/// `ms.cumsum(a, axis=None, out=None)` — 沿轴累积求和（prefix sum）。
///
/// axis=None 时展平为 1D 后 cumsum；axis=int 时保持原 shape。
/// 整数输入以 int64 累加，输出 int64；浮点保持原 dtype。
pub fn cumsum(a: &Array, axis: Option<isize>, out: Option<&Array>) -> Result<Array> {
    let ax = resolve_axis(axis, a.ndim())?;
    op_builder::cumsum_op(a, ax, out)
}
