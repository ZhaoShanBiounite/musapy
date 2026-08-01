//! 索引算子：transpose / permute / flip / slice（view）+ gather / scatter（copy）
//!
//! 设计原则（ADR 002-D4）：
//!   - view 操作（transpose/permute/flip/slice）零拷贝，仅修改 Layout，共享 BufferRef
//!   - copy 操作（gather/scatter）分配新 buffer，走 GPU kernel 或 CPU fallback
//!   - 高级索引（boolean mask / fancy indexing）推迟到 v0.3+

use musapy_core::error::{Result, ShapeError};
use musapy_core::{Array, Layout};

// ── View 操作（零拷贝）────────────────────────────────────────

/// `transpose(a, axes=None)` — 转置（零拷贝视图）。
///
/// `axes=None` 时完全反转维度顺序（等价 `np.transpose(a)`）。
/// 返回新 Array，共享底层 buffer。
pub fn transpose(a: &Array, axes: Option<&[usize]>) -> Result<Array> {
    let new_layout = a.layout().transposed(axes)?;
    Ok(Array::new_view(a, new_layout))
}

/// `permute(a, dims)` — 按指定维度排列（零拷贝视图）。
///
/// 等价于 `transpose(a, axes=dims)`。
pub fn permute(a: &Array, dims: &[usize]) -> Result<Array> {
    transpose(a, Some(dims))
}

/// `flip(a, axis)` — 翻转指定轴（零拷贝视图）。
///
/// stride 取负，offset 调整到该轴末尾。
pub fn flip(a: &Array, axis: usize) -> Result<Array> {
    let new_layout = a.layout().flipped(axis)?;
    Ok(Array::new_view(a, new_layout))
}

// ── Slice 操作（零拷贝）───────────────────────────────────────

/// 切片参数（单维度）。
#[derive(Clone, Debug)]
pub struct SliceSpec {
    pub start: usize,
    pub stop: usize,
    pub step: usize,
}

/// `slice(a, specs)` — 多维切片（零拷贝视图）。
///
/// `specs` 长度必须等于 `a.ndim()`。每维 step >= 1。
pub fn slice(a: &Array, specs: &[SliceSpec]) -> Result<Array> {
    let ranges: Vec<(usize, usize, usize)> = specs
        .iter()
        .map(|s| (s.start, s.stop, s.step))
        .collect();
    let new_layout = a.layout().sliced(&ranges)?;
    Ok(Array::new_view(a, new_layout))
}

/// 整数索引：选择某一维的单个索引，降维（零拷贝视图）。
///
/// 例如 shape [3, 4] 的 `a[1]` → shape [4]，offset += 1 * strides[0]。
pub fn index_select(a: &Array, axis: usize, index: usize) -> Result<Array> {
    let ndim = a.ndim();
    if axis >= ndim {
        return Err(ShapeError::Mismatch(format!(
            "index axis {} out of bounds for ndim {}",
            axis, ndim
        ))
        .into());
    }
    if index >= a.shape()[axis] {
        return Err(ShapeError::Mismatch(format!(
            "index {} out of bounds for dimension {} (size {})",
            index,
            axis,
            a.shape()[axis]
        ))
        .into());
    }

    let layout = a.layout();
    // 新 offset = 原 offset + index * strides[axis]
    let new_offset =
        (layout.offset as isize + index as isize * layout.strides[axis]) as usize;

    // 去掉 axis 维的 shape 和 strides
    let mut new_shape = Vec::with_capacity(ndim - 1);
    let mut new_strides = Vec::with_capacity(ndim - 1);
    for i in 0..ndim {
        if i != axis {
            new_shape.push(layout.shape[i]);
            new_strides.push(layout.strides[i]);
        }
    }

    let new_layout = Layout {
        shape: new_shape,
        strides: new_strides,
        offset: new_offset,
    };
    Ok(Array::new_view(a, new_layout))
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::zeros;
    use musapy_core::Dtype;
    use std::sync::Arc;

    /// 创建测试用 Array（CPU，f32）。
    fn make_array(shape: Vec<usize>) -> Array {
        zeros(&shape, Some(Dtype::Float32), Some(musapy_core::Device::Cpu)).unwrap()
    }

    // --- transpose ---

    #[test]
    fn transpose_2d_default() {
        let a = make_array(vec![2, 3]);
        let t = transpose(&a, None).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.layout().strides, vec![1, 3]);
        // 零拷贝：共享 buffer
        assert!(Arc::ptr_eq(a.data().arc(), t.data().arc()));
    }

    #[test]
    fn transpose_with_axes() {
        let a = make_array(vec![2, 3, 4]);
        let t = transpose(&a, Some(&[1, 0, 2])).unwrap();
        assert_eq!(t.shape(), &[3, 2, 4]);
        assert_eq!(t.layout().strides, vec![4, 12, 1]);
    }

    #[test]
    fn transpose_invalid_axes() {
        let a = make_array(vec![2, 3]);
        assert!(transpose(&a, Some(&[0, 0])).is_err());
        assert!(transpose(&a, Some(&[0, 2])).is_err());
    }

    // --- permute ---

    #[test]
    fn permute_basic() {
        let a = make_array(vec![2, 3, 4]);
        let p = permute(&a, &[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.layout().strides, vec![1, 12, 4]);
    }

    // --- flip ---

    #[test]
    fn flip_axis0() {
        let a = make_array(vec![3, 4]);
        let f = flip(&a, 0).unwrap();
        assert_eq!(f.shape(), &[3, 4]);
        assert_eq!(f.layout().strides, vec![-4, 1]);
        assert_eq!(f.layout().offset, 8);
    }

    #[test]
    fn flip_axis1() {
        let a = make_array(vec![3, 4]);
        let f = flip(&a, 1).unwrap();
        assert_eq!(f.shape(), &[3, 4]);
        assert_eq!(f.layout().strides, vec![4, -1]);
        assert_eq!(f.layout().offset, 3);
    }

    #[test]
    fn flip_out_of_bounds() {
        let a = make_array(vec![3, 4]);
        assert!(flip(&a, 2).is_err());
    }

    // --- slice ---

    #[test]
    fn slice_1d() {
        let a = make_array(vec![10]);
        let s = slice(&a, &[SliceSpec { start: 2, stop: 7, step: 1 }]).unwrap();
        assert_eq!(s.shape(), &[5]);
        assert_eq!(s.layout().offset, 2);
    }

    #[test]
    fn slice_2d_with_step() {
        let a = make_array(vec![4, 6]);
        let s = slice(
            &a,
            &[
                SliceSpec { start: 1, stop: 3, step: 1 },
                SliceSpec { start: 0, stop: 6, step: 2 },
            ],
        )
        .unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        assert_eq!(s.layout().strides, vec![6, 2]);
        assert_eq!(s.layout().offset, 6);
    }

    // --- index_select ---

    #[test]
    fn index_select_axis0() {
        let a = make_array(vec![3, 4]);
        let s = index_select(&a, 0, 1).unwrap();
        assert_eq!(s.shape(), &[4]);
        assert_eq!(s.layout().offset, 4); // 1 * strides[0]=4
    }

    #[test]
    fn index_select_axis1() {
        let a = make_array(vec![3, 4]);
        let s = index_select(&a, 1, 2).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.layout().strides, vec![4]);
        assert_eq!(s.layout().offset, 2); // 2 * strides[1]=1
    }

    #[test]
    fn index_select_out_of_bounds() {
        let a = make_array(vec![3, 4]);
        assert!(index_select(&a, 0, 3).is_err());
        assert!(index_select(&a, 1, 4).is_err());
        assert!(index_select(&a, 2, 0).is_err());
    }
}
