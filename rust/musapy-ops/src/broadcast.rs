//! 广播（Broadcast）— NumPy 规则（ADR-002-D2）
//!
//! 广播不是独立机制，而是 elementwise kernel 的 per-operand stride 能力：
//! 广播维度的 stride 设为 0，kernel 索引不沿该维推进 → 元素被复制。
//!
//! 规则（NumPy）：从最右维对齐；每维必须相等，或其一为 1（或操作数为 0-dim）。

use musapy_core::error::{Result, ShapeError};
use musapy_core::{Layout, Shape};

/// 计算多个输入 shape 的广播输出 shape（NumPy 规则）。
///
/// 从最右维对齐，每维必须相等或其一为 1。
/// 不兼容时返回 `ShapeError::Mismatch`。
///
/// # 示例
///
/// ```ignore
/// broadcast_shape(&[&[3, 1], &[4]])     → Ok([3, 4])
/// broadcast_shape(&[&[2, 1, 3], &[4, 1]]) → Ok([2, 4, 3])
/// broadcast_shape(&[&[], &[5]])          → Ok([5])  // 0-dim broadcast
/// broadcast_shape(&[&[2, 3], &[4]])      → Err(...)  // 不兼容
/// ```
pub fn broadcast_shape(shapes: &[&[usize]]) -> Result<Shape> {
    if shapes.is_empty() {
        return Ok(vec![]);
    }

    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result = vec![1usize; max_ndim];

    for shape in shapes {
        // 右对齐：shape 的第 i 维对应 result 的第 (offset + i) 维
        let offset = max_ndim - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let out_i = offset + i;
            if result[out_i] == 1 {
                result[out_i] = dim;
            } else if dim != 1 && dim != result[out_i] {
                return Err(ShapeError::Mismatch(format!(
                    "broadcast: incompatible shapes {:?} (dim {} = {} vs {})",
                    shapes
                        .iter()
                        .map(|s| format!("{:?}", s))
                        .collect::<Vec<_>>()
                        .join(", "),
                    out_i,
                    result[out_i],
                    dim
                ))
                .into());
            }
        }
    }

    Ok(result)
}

/// 推导操作数参与广播输出的 strides（元素单位）。
///
/// 返回长度为 `target.len()` 的 strides 向量：
/// - 非广播维：保留操作数原始 stride
/// - 广播维（操作数该维为 1 或操作数无该维）：stride = 0
///
/// 0-dim 操作数（shape=[]）→ 全 0 strides（ADR L1-11）。
///
/// # 前置条件
///
/// 调用者应已通过 `broadcast_shape` 验证兼容性。
/// 若 `layout.ndim() > target.len()`，多出的高位维被忽略（不应发生）。
pub fn broadcast_strides(layout: &Layout, target: &[usize]) -> Vec<isize> {
    let target_ndim = target.len();
    let mut strides = vec![0isize; target_ndim];

    // 右对齐：layout 的第 i 维对应 target 的第 (offset + i) 维
    let offset = target_ndim.saturating_sub(layout.ndim());

    for (i, (&dim, &stride)) in layout.shape.iter().zip(layout.strides.iter()).enumerate() {
        let out_i = offset + i;
        if out_i >= target_ndim {
            break;
        }
        if dim == target[out_i] {
            // 非广播维：保留原始 stride
            strides[out_i] = stride;
        }
        // dim == 1 且 target[out_i] > 1 → 广播维，stride 保持 0
        // dim == target[out_i] == 1 → stride 也设为 0（无实际影响）
    }

    strides
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- broadcast_shape ---

    #[test]
    fn shape_same() {
        assert_eq!(broadcast_shape(&[&[2, 3], &[2, 3]]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn shape_3x1_plus_4() {
        assert_eq!(broadcast_shape(&[&[3, 1], &[4]]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn shape_1x4_plus_3x1() {
        assert_eq!(broadcast_shape(&[&[1, 4], &[3, 1]]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn shape_0dim_plus_any() {
        assert_eq!(broadcast_shape(&[&[], &[5]]).unwrap(), vec![5]);
        assert_eq!(broadcast_shape(&[&[], &[2, 3]]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[&[], &[]]).unwrap(), vec![]);
    }

    #[test]
    fn shape_high_dim() {
        // (2,1,3) + (4,1) → (2,4,3)
        assert_eq!(
            broadcast_shape(&[&[2, 1, 3], &[4, 1]]).unwrap(),
            vec![2, 4, 3]
        );
    }

    #[test]
    fn shape_scalar_plus_scalar() {
        assert_eq!(broadcast_shape(&[&[1], &[1]]).unwrap(), vec![1]);
    }

    #[test]
    fn shape_incompatible() {
        assert!(broadcast_shape(&[&[2, 3], &[4]]).is_err());
        assert!(broadcast_shape(&[&[3], &[4]]).is_err());
        assert!(broadcast_shape(&[&[2, 3], &[2, 4]]).is_err());
    }

    #[test]
    fn shape_empty_input() {
        assert_eq!(broadcast_shape(&[]).unwrap(), vec![]);
    }

    #[test]
    fn shape_three_operands() {
        assert_eq!(
            broadcast_shape(&[&[3, 1], &[1, 4], &[3, 4]]).unwrap(),
            vec![3, 4]
        );
    }

    // --- broadcast_strides ---

    #[test]
    fn strides_same_shape_contiguous() {
        let layout = Layout::from_shape(vec![2, 3]);
        // strides = [3, 1], target = [2, 3]
        assert_eq!(broadcast_strides(&layout, &[2, 3]), vec![3, 1]);
    }

    #[test]
    fn strides_broadcast_dim() {
        let layout = Layout::from_shape(vec![3, 1]);
        // layout strides = [1, 1], target = [3, 4]
        // dim 1: layout=1, target=4 → broadcast → stride 0
        assert_eq!(broadcast_strides(&layout, &[3, 4]), vec![1, 0]);
    }

    #[test]
    fn strides_1d_to_2d() {
        let layout = Layout::from_shape(vec![4]);
        // layout strides = [1], target = [3, 4]
        // offset = 1, dim 0 of layout → target dim 1
        assert_eq!(broadcast_strides(&layout, &[3, 4]), vec![0, 1]);
    }

    #[test]
    fn strides_0dim() {
        let layout = Layout::from_shape(vec![]);
        // 0-dim → all strides 0
        assert_eq!(broadcast_strides(&layout, &[3, 4]), vec![0, 0]);
        assert_eq!(broadcast_strides(&layout, &[5]), vec![0]);
        assert_eq!(broadcast_strides(&layout, &[]), vec![]);
    }

    #[test]
    fn strides_high_dim() {
        // layout (4,1) strides [1,1], target (2,4,3)
        let layout = Layout::from_shape(vec![4, 1]);
        // offset = 1
        // layout dim 0 (size 4) → target dim 1 (size 4): stride 1
        // layout dim 1 (size 1) → target dim 2 (size 3): broadcast → 0
        assert_eq!(broadcast_strides(&layout, &[2, 4, 3]), vec![0, 1, 0]);
    }

    #[test]
    fn strides_non_contiguous_input() {
        // 模拟转置视图：shape [3, 4], strides [1, 3]
        let layout = Layout::from_shape_and_strides(vec![3, 4], vec![1, 3]).unwrap();
        // target [3, 4] → 保留原始 strides
        assert_eq!(broadcast_strides(&layout, &[3, 4]), vec![1, 3]);
    }

    #[test]
    fn strides_non_contiguous_with_broadcast() {
        // 模拟：shape [1, 4], strides [0, 1]（已经是广播形式），target [3, 4]
        let layout = Layout::from_shape_and_strides(vec![1, 4], vec![0, 1]).unwrap();
        assert_eq!(broadcast_strides(&layout, &[3, 4]), vec![0, 1]);
    }
}
