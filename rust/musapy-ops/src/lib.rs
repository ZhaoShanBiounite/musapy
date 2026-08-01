//! musapy-ops: 算子层（ADR L2-4, L2-5）
//!
//! v0.1: OpBuilder + elementwise（add only）
//! v0.2: broadcast + stride-aware ABI（Phase 1）;
//!       elementwise 全家桶 + 类型提升（Phase 2）;
//!       comparison 套件（Phase 3）;
//!       reduction 套件（Phase 4）;
//!       creation 套件（Phase 5）;
//!       indexing 套件（Phase 6）

pub mod broadcast;
pub mod comparison;
pub mod creation;
pub mod elementwise;
pub mod indexing;
pub mod kernels;
pub mod op_builder;
pub mod reduction;

// 公开 API 在 elementwise 模块；根级再导出保持 `musapy_ops::add` 兼容。
pub use elementwise::*;
pub use comparison::*;
pub use reduction::*;
pub use creation::{zeros, ones, full, eye, arange, linspace, zeros_like, ones_like};
pub use indexing::{transpose, permute, flip, slice, index_select, SliceSpec};
