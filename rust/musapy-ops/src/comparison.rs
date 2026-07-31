//! Comparison ops（Phase 3, ADR-002 Phase 3）
//!
//! eq / ne / lt / gt / le / ge — 输出 bool 数组。

use crate::op_builder::{self, CompareKernel};
use musapy_core::{Array, Result};

/// `ms.eq(a, b, out=None)` — 逐元素等于比较（广播 + 类型提升 → bool）。
pub fn eq(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::comparison_elementwise(a, b, out, CompareKernel::Eq)
}

/// `ms.ne(a, b, out=None)` — 逐元素不等比较。
pub fn ne(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::comparison_elementwise(a, b, out, CompareKernel::Ne)
}

/// `ms.lt(a, b, out=None)` — 逐元素小于比较。
pub fn lt(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::comparison_elementwise(a, b, out, CompareKernel::Lt)
}

/// `ms.gt(a, b, out=None)` — 逐元素大于比较。
pub fn gt(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::comparison_elementwise(a, b, out, CompareKernel::Gt)
}

/// `ms.le(a, b, out=None)` — 逐元素小于等于比较。
pub fn le(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::comparison_elementwise(a, b, out, CompareKernel::Le)
}

/// `ms.ge(a, b, out=None)` — 逐元素大于等于比较。
pub fn ge(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    op_builder::comparison_elementwise(a, b, out, CompareKernel::Ge)
}
