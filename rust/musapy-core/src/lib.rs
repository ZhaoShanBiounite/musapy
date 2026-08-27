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
pub mod dlpack;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod mem_stats;
pub mod musa_ffi;
pub mod resolution;
pub mod stream;
// MUSA-X 数学库 FFI + 句柄管理（v0.3，ADR-003 003-D1/D2）。
// 这些模块引用 mublas/murand/mufft/musparse 符号，但 musapy-core 只链 musart
// （L2-3）——链接指令由 musapy-ops/build.rs 发出。故双重隔离：
//   1. feature `math-libs`：musapy-core 单独构建（cargo test -p musapy-core）
//      不开启；由 musapy-ops / musapy-python 依赖侧启用。
//   2. `any(musapy_mock_musa, not(test))`：workspace 级 cargo test 会做 feature
//      统一，core 的测试二进制也会带上 math-libs——但真实模式下它只链 musart，
//      无法解析 MUSA-X 符号。故真实模式排除 core 自身测试构建；单测一律在
//      mock 模式跑（stub 无外部符号），真机验证走 pytest 冒烟（P1.7）。
//      mock 模式下 musapy_mock_musa 使模块照常进入测试构建。
#[cfg(all(feature = "math-libs", any(musapy_mock_musa, not(test))))]
pub mod math_handle;
#[cfg(all(feature = "math-libs", any(musapy_mock_musa, not(test))))]
pub mod musa_x_ffi;
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
    DeviceError, DtypeError, IndexError, InteropError, KernelError, LinAlgError, MemoryError,
    MusapyError, Result, ShapeError, StreamError,
};
pub use layout::{Layout, Shape};
pub use stream::{Event, OpContext, PythonFrame, Stream};

/// musapy-core 版本号
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
