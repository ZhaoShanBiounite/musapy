//! musapy-core: 核心运行时（ADR L2-3）
//!
//! Phase 1: 错误模型 + ABI 版本管理
//! Phase 2: 核心数据结构（device/dtype/layout/stream/buffer/array）
//! Phase 3: MUSA FFI + 内存与流基础设施

pub mod abi;
pub mod array;
pub mod buffer;
pub mod debug;
pub mod device;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod mem_stats;
pub mod musa_ffi;
pub mod resolution;
pub mod stream;
// deferred-free 队列仅在默认内存路径编译（ADR L3-11）。
// 启用 stream-ordered feature 后走 musaFreeAsync，本模块不编译。
#[cfg(not(feature = "stream-ordered"))]
pub mod deferred_free;
// Buffer Pool：GPU 内存复用池（Phase C-lite），同样仅默认路径。
#[cfg(not(feature = "stream-ordered"))]
pub mod buffer_pool;

pub use array::{Array, DtypeResolution};
pub use buffer::{Buffer, BufferRef};
pub use debug::{is_debug, set_debug};
pub use device::{Device, DeviceResolution, ResolutionSource, SourceLocation};
pub use dtype::{Dtype, promote};
pub use error::{
    DeviceError, DtypeError, InteropError, KernelError, MemoryError, MusapyError, Result,
    ShapeError, StreamError,
};
pub use layout::{Layout, Shape};
pub use stream::{Event, OpContext, PythonFrame, Stream};

/// musapy-core 版本号
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
