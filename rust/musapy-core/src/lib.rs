//! musapy-core: 核心运行时（ADR L2-3）
//!
//! Phase 1: 错误模型 + ABI 版本管理
//! Phase 2+: 加入 device/dtype/stream/buffer/array 等模块

pub mod error;
pub mod abi;

pub use error::{
    DeviceError, DtypeError, InteropError, KernelError, MemoryError, MusapyError, Result,
    ShapeError, StreamError,
};

/// musapy-core 版本号
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
