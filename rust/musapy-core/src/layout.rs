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
/// - `strides`：各维度的步长（元素数，非字节数；可为负，如 flip 视图）
/// - `offset`：起始偏移（元素数，始终非负）
///
/// 连续布局（C order, row-major）的 strides 由 shape 计算：
///   shape [2, 3, 4] → strides [12, 4, 1]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub shape: Shape,
    pub strides: Vec<isize>,
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
    pub fn from_shape_and_strides(shape: Shape, strides: Vec<isize>) -> Result<Self> {
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
        self.offset == 0 && self.has_contiguous_strides()
    }

    /// strides 是否为 C 连续（忽略 offset）。
    ///
    /// 用于区分"带 offset 的连续切片"（指针调整即可）与
    /// "真正非连续视图"（transpose/flip 等，需物化）。
    pub fn has_contiguous_strides(&self) -> bool {
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
        let mut new_strides = vec![0isize; target_ndim];

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
        let mut off = self.offset as isize;
        for i in 0..indices.len() {
            if indices[i] >= self.shape[i] {
                return Err(ShapeError::Mismatch(format!(
                    "index {} out of bounds for dimension {} (size {})",
                    indices[i], i, self.shape[i]
                ))
                .into());
            }
            off += indices[i] as isize * self.strides[i];
        }
        Ok(off as usize)
    }

    // ── 视图变换（P6 indexing）──────────────────────────────────

    /// 转置：按 axes 重排维度（零拷贝视图）。
    ///
    /// `axes=None` 时完全反转维度顺序（等价 `np.transpose(a)`）。
    /// 返回新 Layout，共享底层 buffer。
    ///
    /// # 示例
    ///
    /// shape [2, 3], strides [3, 1] → transposed(None) → shape [3, 2], strides [1, 3]
    pub fn transposed(&self, axes: Option<&[usize]>) -> Result<Layout> {
        let ndim = self.ndim();
        if ndim == 0 {
            return Ok(self.clone());
        }

        let perm: Vec<usize> = match axes {
            None => (0..ndim).rev().collect(),
            Some(ax) => {
                if ax.len() != ndim {
                    return Err(ShapeError::Mismatch(format!(
                        "axes length {} != ndim {}",
                        ax.len(),
                        ndim
                    ))
                    .into());
                }
                // 验证 axes 是 0..ndim 的排列
                let mut seen = vec![false; ndim];
                for &a in ax {
                    if a >= ndim || seen[a] {
                        return Err(ShapeError::Mismatch(format!(
                            "invalid or duplicate axis {} for ndim {}",
                            a, ndim
                        ))
                        .into());
                    }
                    seen[a] = true;
                }
                ax.to_vec()
            }
        };

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<isize> = perm.iter().map(|&i| self.strides[i]).collect();

        Ok(Layout {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// 翻转指定轴（零拷贝视图）。
    ///
    /// stride 取负，offset 调整到该轴末尾元素位置。
    ///
    /// # 示例
    ///
    /// shape [3, 4], strides [4, 1], offset 0
    ///   → flipped(0) → shape [3, 4], strides [-4, 1], offset 8
    pub fn flipped(&self, axis: usize) -> Result<Layout> {
        if axis >= self.ndim() {
            return Err(ShapeError::Mismatch(format!(
                "flip axis {} out of bounds for ndim {}",
                axis,
                self.ndim()
            ))
            .into());
        }

        let mut new_strides = self.strides.clone();
        let mut new_offset = self.offset as isize;

        // 调整 offset 到该轴最后一个元素
        new_offset += (self.shape[axis] as isize - 1) * self.strides[axis];
        // stride 取负
        new_strides[axis] = -self.strides[axis];

        Ok(Layout {
            shape: self.shape.clone(),
            strides: new_strides,
            offset: new_offset as usize,
        })
    }

    /// 切片：按各维度的 (start, stop, step) 创建视图（零拷贝）。
    ///
    /// `ranges` 长度必须等于 ndim。每维 step >= 1。
    ///
    /// 新 shape[i] = ceil((stop - start) / step)
    /// 新 strides[i] = old_strides[i] * step
    /// 新 offset = old_offset + sum(start[i] * old_strides[i])
    pub fn sliced(&self, ranges: &[(usize, usize, usize)]) -> Result<Layout> {
        let ndim = self.ndim();
        if ranges.len() != ndim {
            return Err(ShapeError::Mismatch(format!(
                "slice ranges length {} != ndim {}",
                ranges.len(),
                ndim
            ))
            .into());
        }

        let mut new_shape = Vec::with_capacity(ndim);
        let mut new_strides = Vec::with_capacity(ndim);
        let mut new_offset = self.offset as isize;

        for i in 0..ndim {
            let (start, stop, step) = ranges[i];
            if step == 0 {
                return Err(ShapeError::Mismatch("slice step cannot be zero".into()).into());
            }
            if start > stop || stop > self.shape[i] {
                return Err(ShapeError::Mismatch(format!(
                    "slice [{}, {}, {}] out of bounds for dimension {} (size {})",
                    start, stop, step, i, self.shape[i]
                ))
                .into());
            }

            let dim_size = if stop > start {
                (stop - start + step - 1) / step // ceil division
            } else {
                0
            };

            new_shape.push(dim_size);
            new_strides.push(self.strides[i] * step as isize);
            new_offset += start as isize * self.strides[i];
        }

        Ok(Layout {
            shape: new_shape,
            strides: new_strides,
            offset: new_offset as usize,
        })
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
fn compute_contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    if ndim == 0 {
        return Vec::new();
    }
    let mut strides = vec![0isize; ndim];
    // 最后一个维度 stride = 1
    strides[ndim - 1] = 1;
    // 从后往前累乘
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
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

    // --- transposed ---

    #[test]
    fn transposed_2d_default() {
        // shape [2, 3], strides [3, 1] → shape [3, 2], strides [1, 3]
        let l = Layout::from_shape(vec![2, 3]);
        let t = l.transposed(None).unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.strides, vec![1, 3]);
        assert_eq!(t.offset, 0);
    }

    #[test]
    fn transposed_3d_default() {
        // shape [2, 3, 4] → reversed → shape [4, 3, 2], strides [1, 4, 12]
        let l = Layout::from_shape(vec![2, 3, 4]);
        let t = l.transposed(None).unwrap();
        assert_eq!(t.shape, vec![4, 3, 2]);
        assert_eq!(t.strides, vec![1, 4, 12]);
    }

    #[test]
    fn transposed_with_axes() {
        // shape [2, 3, 4], axes [1, 0, 2] → shape [3, 2, 4], strides [4, 12, 1]
        let l = Layout::from_shape(vec![2, 3, 4]);
        let t = l.transposed(Some(&[1, 0, 2])).unwrap();
        assert_eq!(t.shape, vec![3, 2, 4]);
        assert_eq!(t.strides, vec![4, 12, 1]);
    }

    #[test]
    fn transposed_identity() {
        let l = Layout::from_shape(vec![2, 3]);
        let t = l.transposed(Some(&[0, 1])).unwrap();
        assert_eq!(t, l);
    }

    #[test]
    fn transposed_0dim() {
        let l = Layout::from_shape(vec![]);
        let t = l.transposed(None).unwrap();
        assert_eq!(t, l);
    }

    #[test]
    fn transposed_invalid_axes() {
        let l = Layout::from_shape(vec![2, 3]);
        assert!(l.transposed(Some(&[0, 0])).is_err()); // duplicate
        assert!(l.transposed(Some(&[0, 2])).is_err()); // out of bounds
        assert!(l.transposed(Some(&[0])).is_err()); // wrong length
    }

    #[test]
    fn transposed_preserves_offset() {
        let l = Layout::from_shape(vec![2, 3]);
        let mut l2 = l.clone();
        l2.offset = 5;
        let t = l2.transposed(None).unwrap();
        assert_eq!(t.offset, 5);
    }

    // --- flipped ---

    #[test]
    fn flipped_axis0() {
        // shape [3, 4], strides [4, 1] → strides [-4, 1], offset = (3-1)*4 = 8
        let l = Layout::from_shape(vec![3, 4]);
        let f = l.flipped(0).unwrap();
        assert_eq!(f.shape, vec![3, 4]);
        assert_eq!(f.strides, vec![-4, 1]);
        assert_eq!(f.offset, 8);
    }

    #[test]
    fn flipped_axis1() {
        // shape [3, 4], strides [4, 1] → strides [4, -1], offset = (4-1)*1 = 3
        let l = Layout::from_shape(vec![3, 4]);
        let f = l.flipped(1).unwrap();
        assert_eq!(f.shape, vec![3, 4]);
        assert_eq!(f.strides, vec![4, -1]);
        assert_eq!(f.offset, 3);
    }

    #[test]
    fn flipped_double_is_identity() {
        let l = Layout::from_shape(vec![3, 4]);
        let f = l.flipped(0).unwrap().flipped(0).unwrap();
        assert_eq!(f, l);
    }

    #[test]
    fn flipped_out_of_bounds() {
        let l = Layout::from_shape(vec![3, 4]);
        assert!(l.flipped(2).is_err());
    }

    #[test]
    fn flipped_linear_offset_correct() {
        // shape [3], strides [1] → flipped → strides [-1], offset 2
        // element [0] should map to original index 2
        let l = Layout::from_shape(vec![3]);
        let f = l.flipped(0).unwrap();
        assert_eq!(f.linear_offset(&[0]).unwrap(), 2);
        assert_eq!(f.linear_offset(&[1]).unwrap(), 1);
        assert_eq!(f.linear_offset(&[2]).unwrap(), 0);
    }

    // --- sliced ---

    #[test]
    fn sliced_basic() {
        // shape [10], strides [1], slice [2, 7, 1] → shape [5], strides [1], offset 2
        let l = Layout::from_shape(vec![10]);
        let s = l.sliced(&[(2, 7, 1)]).unwrap();
        assert_eq!(s.shape, vec![5]);
        assert_eq!(s.strides, vec![1]);
        assert_eq!(s.offset, 2);
    }

    #[test]
    fn sliced_with_step() {
        // shape [10], strides [1], slice [0, 10, 2] → shape [5], strides [2], offset 0
        let l = Layout::from_shape(vec![10]);
        let s = l.sliced(&[(0, 10, 2)]).unwrap();
        assert_eq!(s.shape, vec![5]);
        assert_eq!(s.strides, vec![2]);
        assert_eq!(s.offset, 0);
    }

    #[test]
    fn sliced_2d() {
        // shape [4, 6], strides [6, 1]
        // slice dim0 [1, 3, 1], dim1 [0, 6, 2]
        // → shape [2, 3], strides [6, 2], offset = 1*6 + 0*1 = 6
        let l = Layout::from_shape(vec![4, 6]);
        let s = l.sliced(&[(1, 3, 1), (0, 6, 2)]).unwrap();
        assert_eq!(s.shape, vec![2, 3]);
        assert_eq!(s.strides, vec![6, 2]);
        assert_eq!(s.offset, 6);
    }

    #[test]
    fn sliced_linear_offset_correct() {
        // shape [5], slice [1, 4, 1] → shape [3], offset 1
        // element [0] → original index 1, [1] → 2, [2] → 3
        let l = Layout::from_shape(vec![5]);
        let s = l.sliced(&[(1, 4, 1)]).unwrap();
        assert_eq!(s.linear_offset(&[0]).unwrap(), 1);
        assert_eq!(s.linear_offset(&[1]).unwrap(), 2);
        assert_eq!(s.linear_offset(&[2]).unwrap(), 3);
    }

    #[test]
    fn sliced_empty_range() {
        let l = Layout::from_shape(vec![10]);
        let s = l.sliced(&[(5, 5, 1)]).unwrap();
        assert_eq!(s.shape, vec![0]);
        assert_eq!(s.size(), 0);
    }

    #[test]
    fn sliced_step_zero_error() {
        let l = Layout::from_shape(vec![10]);
        assert!(l.sliced(&[(0, 5, 0)]).is_err());
    }

    #[test]
    fn sliced_out_of_bounds() {
        let l = Layout::from_shape(vec![10]);
        assert!(l.sliced(&[(0, 11, 1)]).is_err()); // stop > shape
        assert!(l.sliced(&[(7, 3, 1)]).is_err()); // start > stop
    }
}
