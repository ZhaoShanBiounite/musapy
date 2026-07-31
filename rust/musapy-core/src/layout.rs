//! 张量布局：shape / strides / offset（ADR L1-11）
//!
//! 职责：
//!   1. Layout 结构：描述数据在内存中的排列方式
//!   2. Shape 类型别名
//!   3. 连续布局计算、索引偏移计算
//!
//! 设计说明（ADR L1-11）：
//!   - 0-dim Array（shape=[]）无特殊路径，MUSA runtime 自动优化
//!   - strides 单位是"元素数"而非"字节数"（与 NumPy/CuPy 一致）
//!   - offset 单位也是"元素数"

use crate::error::{Result, ShapeError};
use std::fmt;

/// 形状类型别名。
pub type Shape = Vec<usize>;

/// 张量内存布局。
///
/// - `shape`：各维度大小
/// - `strides`：各维度的步长（元素数，非字节数）
/// - `offset`：起始偏移（元素数）
///
/// 连续布局（C order, row-major）的 strides 由 shape 计算：
///   shape [2, 3, 4] → strides [12, 4, 1]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub shape: Shape,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Layout {
    /// 从 shape 创建连续布局（C order, row-major）。
    ///
    /// shape [] → strides []（0-dim）
    /// shape [5] → strides [1]
    /// shape [2, 3] → strides [3, 1]
    /// shape [2, 3, 4] → strides [12, 4, 1]
    pub fn from_shape(shape: Shape) -> Self {
        let strides = compute_contiguous_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// 从 shape 和 strides 创建布局（offset=0）。
    pub fn from_shape_and_strides(shape: Shape, strides: Vec<usize>) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(ShapeError::Mismatch(format!(
                "shape rank {} != strides rank {}",
                shape.len(),
                strides.len()
            ))
            .into());
        }
        Ok(Self {
            shape,
            strides,
            offset: 0,
        })
    }

    /// 维度数（rank）。
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// 元素总数（shape 各维度乘积）。
    ///
    /// 0-dim（shape=[]）返回 1。
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// 是否为连续布局（C order, row-major, offset=0）。
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let expected = compute_contiguous_strides(&self.shape);
        self.strides == expected
    }

    /// 广播到目标 shape（ADR-002-D2, P1.5）。
    ///
    /// 返回新 Layout：target shape + 广播 strides（广播维 stride=0），offset 保持不变。
    /// 遵循 NumPy 广播规则：右对齐，每维相等或其一为 1。
    ///
    /// # 示例
    ///
    /// ```ignore
    /// Layout::from_shape(vec![3, 1]).broadcast_to(&[3, 4])
    ///   → Layout { shape: [3, 4], strides: [1, 0], offset: 0 }
    /// ```
    pub fn broadcast_to(&self, target: &[usize]) -> Result<Layout> {
        let target_ndim = target.len();
        if target_ndim < self.ndim() {
            return Err(ShapeError::Mismatch(format!(
                "broadcast_to: target rank {} < input rank {}",
                target_ndim,
                self.ndim()
            ))
            .into());
        }

        let offset = target_ndim - self.ndim();
        let mut new_strides = vec![0usize; target_ndim];

        for (i, (&dim, &stride)) in self.shape.iter().zip(self.strides.iter()).enumerate() {
            let out_i = offset + i;
            if dim == target[out_i] {
                new_strides[out_i] = stride;
            } else if dim == 1 {
                new_strides[out_i] = 0; // 广播维
            } else {
                return Err(ShapeError::Mismatch(format!(
                    "broadcast_to: cannot broadcast dim {} (size {}) to target size {}",
                    out_i, dim, target[out_i]
                ))
                .into());
            }
        }

        Ok(Layout {
            shape: target.to_vec(),
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// 计算给定多维索引的线性偏移（元素数）。
    ///
    /// 返回 `offset + sum(indices[i] * strides[i])`。
    /// 索引维度数不匹配或越界时返回 `ShapeError`。
    pub fn linear_offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(ShapeError::Mismatch(format!(
                "index rank {} != layout rank {}",
                indices.len(),
                self.shape.len()
            ))
            .into());
        }
        let mut off = self.offset;
        for i in 0..indices.len() {
            if indices[i] >= self.shape[i] {
                return Err(ShapeError::Mismatch(format!(
                    "index {} out of bounds for dimension {} (size {})",
                    indices[i], i, self.shape[i]
                ))
                .into());
            }
            off += indices[i] * self.strides[i];
        }
        Ok(off)
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Layout(shape={:?}, strides={:?}, offset={})",
            self.shape, self.strides, self.offset
        )
    }
}

/// 计算 C order（row-major）连续布局的 strides。
///
/// shape [2, 3, 4] → strides [12, 4, 1]
/// shape [] → strides []
fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return Vec::new();
    }
    let mut strides = vec![0usize; ndim];
    // 最后一个维度 stride = 1
    strides[ndim - 1] = 1;
    // 从后往前累乘
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- from_shape + strides ---

    #[test]
    fn from_shape_0dim() {
        let l = Layout::from_shape(vec![]);
        assert_eq!(l.shape, vec![]);
        assert_eq!(l.strides, vec![]);
        assert_eq!(l.offset, 0);
        assert_eq!(l.ndim(), 0);
        assert_eq!(l.size(), 1);
    }

    #[test]
    fn from_shape_1dim() {
        let l = Layout::from_shape(vec![5]);
        assert_eq!(l.shape, vec![5]);
        assert_eq!(l.strides, vec![1]);
    }

    #[test]
    fn from_shape_2dim() {
        let l = Layout::from_shape(vec![2, 3]);
        assert_eq!(l.strides, vec![3, 1]);
    }

    #[test]
    fn from_shape_3dim() {
        let l = Layout::from_shape(vec![2, 3, 4]);
        assert_eq!(l.strides, vec![12, 4, 1]);
    }

    // --- ndim / size ---

    #[test]
    fn ndim_and_size() {
        assert_eq!(Layout::from_shape(vec![2, 3, 4]).ndim(), 3);
        assert_eq!(Layout::from_shape(vec![2, 3, 4]).size(), 24);
        assert_eq!(Layout::from_shape(vec![100]).size(), 100);
        assert_eq!(Layout::from_shape(vec![]).size(), 1);
    }

    // --- is_contiguous ---

    #[test]
    fn is_contiguous_default() {
        assert!(Layout::from_shape(vec![]).is_contiguous());
        assert!(Layout::from_shape(vec![5]).is_contiguous());
        assert!(Layout::from_shape(vec![2, 3, 4]).is_contiguous());
    }

    #[test]
    fn is_contiguous_with_custom_strides() {
        // 非连续布局（转置）
        let l = Layout::from_shape_and_strides(vec![3, 4], vec![1, 3]).unwrap();
        assert!(!l.is_contiguous());
    }

    #[test]
    fn is_contiguous_with_offset() {
        let mut l = Layout::from_shape(vec![3, 4]);
        l.offset = 10;
        assert!(!l.is_contiguous());
    }

    // --- from_shape_and_strides ---

    #[test]
    fn from_shape_and_strides_rank_mismatch() {
        let result = Layout::from_shape_and_strides(vec![2, 3], vec![1, 2, 3]);
        assert!(result.is_err());
    }

    // --- linear_offset ---

    #[test]
    fn linear_offset_1dim() {
        let l = Layout::from_shape(vec![10]);
        assert_eq!(l.linear_offset(&[5]).unwrap(), 5);
        assert_eq!(l.linear_offset(&[0]).unwrap(), 0);
    }

    #[test]
    fn linear_offset_3dim() {
        // shape [2, 3, 4], strides [12, 4, 1]
        let l = Layout::from_shape(vec![2, 3, 4]);
        assert_eq!(l.linear_offset(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(l.linear_offset(&[1, 2, 3]).unwrap(), 12 + 8 + 3);
        assert_eq!(l.linear_offset(&[0, 1, 1]).unwrap(), 4 + 1);
    }

    #[test]
    fn linear_offset_with_custom_strides() {
        // 转置布局：shape [3, 4], strides [1, 3]
        let l = Layout::from_shape_and_strides(vec![3, 4], vec![1, 3]).unwrap();
        // [1, 2] → 1*1 + 2*3 = 7
        assert_eq!(l.linear_offset(&[1, 2]).unwrap(), 7);
    }

    #[test]
    fn linear_offset_with_offset() {
        let mut l = Layout::from_shape(vec![5]);
        l.offset = 100;
        assert_eq!(l.linear_offset(&[3]).unwrap(), 103);
    }

    #[test]
    fn linear_offset_rank_mismatch() {
        let l = Layout::from_shape(vec![2, 3]);
        assert!(l.linear_offset(&[0]).is_err()); // rank 1 != 2
        assert!(l.linear_offset(&[0, 0, 0]).is_err()); // rank 3 != 2
    }

    #[test]
    fn linear_offset_out_of_bounds() {
        let l = Layout::from_shape(vec![3, 4]);
        assert!(l.linear_offset(&[3, 0]).is_err()); // dim 0 越界
        assert!(l.linear_offset(&[0, 4]).is_err()); // dim 1 越界
    }

    // --- broadcast_to ---

    #[test]
    fn broadcast_to_same_shape() {
        let l = Layout::from_shape(vec![2, 3]);
        let b = l.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(b.strides, vec![3, 1]);
    }

    #[test]
    fn broadcast_to_expand_dim() {
        let l = Layout::from_shape(vec![3, 1]);
        let b = l.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(b.shape, vec![3, 4]);
        assert_eq!(b.strides, vec![1, 0]);
    }

    #[test]
    fn broadcast_to_add_leading_dim() {
        let l = Layout::from_shape(vec![4]);
        let b = l.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(b.shape, vec![3, 4]);
        assert_eq!(b.strides, vec![0, 1]);
    }

    #[test]
    fn broadcast_to_0dim() {
        let l = Layout::from_shape(vec![]);
        let b = l.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(b.shape, vec![3, 4]);
        assert_eq!(b.strides, vec![0, 0]);
    }

    #[test]
    fn broadcast_to_incompatible() {
        let l = Layout::from_shape(vec![2, 3]);
        assert!(l.broadcast_to(&[2, 4]).is_err());
    }

    #[test]
    fn broadcast_to_lower_rank() {
        let l = Layout::from_shape(vec![2, 3]);
        assert!(l.broadcast_to(&[3]).is_err());
    }

    // --- Display ---

    #[test]
    fn display_format() {
        let l = Layout::from_shape(vec![2, 3]);
        assert_eq!(
            l.to_string(),
            "Layout(shape=[2, 3], strides=[3, 1], offset=0)"
        );
    }
}
