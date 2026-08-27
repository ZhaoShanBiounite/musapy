//! musapy-ops: 算子层（ADR L2-4, L2-5）
//!
//! v0.1: OpBuilder + elementwise（add only）
//! v0.2: broadcast + stride-aware ABI（Phase 1）;
//!       elementwise 全家桶 + 类型提升（Phase 2）;
//!       comparison 套件（Phase 3）;
//!       reduction 套件（Phase 4）;
//!       creation 套件（Phase 5）;
//!       indexing 套件（Phase 6）
//! v0.3: linalg A——matmul/dot/solve（Phase 2，ADR-003 003-D3/D6，GPU-only）

pub mod broadcast;
pub mod comparison;
pub mod creation;
pub mod elementwise;
pub mod fft;
pub mod indexing;
pub mod kernels;
pub mod linalg;
pub mod op_builder;
pub mod random;
pub mod reduction;
pub mod sparse;

// 公开 API 在 elementwise 模块；根级再导出保持 `musapy_ops::add` 兼容。
pub use comparison::*;
pub use creation::{arange, eye, full, linspace, ones, ones_like, zeros, zeros_like};
pub use elementwise::*;
pub use fft::{FftNorm, fft, ifft, rfft};
pub use indexing::{
    SliceSpec, adv_index, boolean_mask, contiguous, flip, gather, index_select, permute, scatter,
    slice, transpose,
};
pub use linalg::{dot, lu, matmul, qr, solve, svd};
pub use random::{bernoulli, normal, rand, randn, uniform};
pub use reduction::*;
pub use sparse::{CsrMatrix, csr_from_arrays, csr_from_host, spmm, spmv, toarray};
