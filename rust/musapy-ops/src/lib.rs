//! musapy-ops: 算子层（ADR L2-4, L2-5）
//!
//! Phase 6: OpBuilder + elementwise（add only）
//! Phase 3+: 加 reduction/init/linalg/random/fft/sparse/indexing/broadcast/comparison

pub mod kernels;
pub mod op_builder;

pub use op_builder::add;
