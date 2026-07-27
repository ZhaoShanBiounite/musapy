//! Deferred-free 全局队列（ADR L3-11，默认内存路径）
//!
//! 适用场景：MUSA Runtime 3.x/4.x 的 libmusart.so 不导出
//! `musaMallocAsync`/`musaFreeAsync` 符号（ADR L3-9 实测确认），无法做
//! stream-ordered free。本模块提供 fallback：延迟到 `Stream::synchronize`
//! 成功后批量 `musaFree`。
//!
//! 工作流程（ADR L3-11）：
//!   1. `Buffer::drop` 不立即 free，而是在 dealloc_stream 上 wait 所有
//!      read/write events 后，把 `(ptr, device)` 入此全局队列
//!   2. `Stream::synchronize` 成功后调用 `reclaim_all()`，对队列中每个
//!      buffer 调用 `musaSetDevice(id)` + `musaFree(ptr)`
//!
//! 安全保证：
//!   - 入队前已 wait events，synchronize 后 events 一定已完成
//!   - 所以 reclaim 时 buffer 一定不再被任何流使用
//!
//! 与 stream-ordered 路径的关系（ADR L3-9）：
//!   - 本模块仅在默认路径（无 `stream-ordered` feature）编译
//!   - 启用 `stream-ordered` feature 后，Buffer 走 `musaFreeAsync` 路径，
//!     本模块不被编译，自动失效

#![cfg(not(feature = "stream-ordered"))]

use crate::device::Device;
use crate::error::Result;
use crate::musa_ffi;
use parking_lot::Mutex;
use std::ptr::NonNull;
use std::sync::OnceLock;

// ============================================================
// 1. 队列数据结构
// ============================================================

/// 一个待释放的 GPU 内存块。
struct DeferredEntry {
    /// GPU 内存指针（由 musaMalloc 分配，非空）。
    ptr: NonNull<u8>,
    /// 所属设备（reclaim 时需 musaSetDevice 切换到该设备）。
    device: Device,
    /// 分配大小（字节），用于内存统计（ADR L3-28）。
    size: usize,
}

// GPU 指针可跨线程释放：musaFree 绑定的是"当前设备"而非分配线程，
// reclaim 前会 musaSetDevice 切到正确设备，故 Send 安全。
// 队列受 Mutex 保护，无并发访问，无需 Sync。
unsafe impl Send for DeferredEntry {}

/// 全局 deferred-free 队列。
///
/// 用 `OnceLock<Mutex<Vec<...>>>` 延迟初始化，避免 static 顺序问题。
/// `parking_lot::Mutex` 不可重入但低开销，适合此处的短临界区。
static DEFERRED: OnceLock<Mutex<Vec<DeferredEntry>>> = OnceLock::new();

fn queue() -> &'static Mutex<Vec<DeferredEntry>> {
    DEFERRED.get_or_init(|| Mutex::new(Vec::new()))
}

// ============================================================
// 2. 公开 API
// ============================================================

/// 把一个待释放的 buffer 入队（ADR L3-11 步骤 1）。
///
/// **调用前置条件**：调用方（`Buffer::drop`）必须已在 dealloc_stream 上
/// wait 完所有 read/write events。本函数只负责记录，不做同步。
///
/// 入队后指针所有权转移给队列，调用方不得再使用该指针。
pub fn enqueue(ptr: NonNull<u8>, device: Device, size: usize) {
    crate::mem_stats::record_cached(size);
    queue().lock().push(DeferredEntry { ptr, device, size });
}

/// 当前队列中待释放的 buffer 数量（调试/统计用）。
pub fn pending_count() -> usize {
    queue().lock().len()
}

/// 批量释放队列中所有 buffer（ADR L3-11 步骤 2）。
///
/// 应在 `Stream::synchronize` 成功后调用：synchronize 保证流上所有 op
/// 完成，加上入队前已 wait events，此刻 buffer 一定不再被任何流使用。
///
/// 对每个 entry：`musaSetDevice(id)` → `musaFree(ptr)`。
/// 单个 free 失败不中断整体 reclaim（记录警告，继续下一个），全部尝试后
/// 若有失败则返回第一个错误。
pub fn reclaim_all() -> Result<()> {
    let entries: Vec<DeferredEntry> = std::mem::take(&mut *queue().lock());

    let mut first_err: Option<crate::error::MusapyError> = None;
    for entry in entries {
        if let Err(e) = reclaim_one(entry) {
            // 保留第一个错误，继续尝试其余（避免一个坏 buffer 阻塞队列）
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
    }

    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// ============================================================
// 3. 内部实现
// ============================================================

/// 释放单个 entry：切换设备 + musaFree。
fn reclaim_one(entry: DeferredEntry) -> Result<()> {
    let device_id = match &entry.device {
        Device::Musa(id) => *id as i32,
        // CPU buffer 不走 deferred-free（Buffer::drop 直接 std::alloc::dealloc），
        // 理论上不会到达这里；防御性处理。
        Device::Cpu => return Ok(()),
    };

    // musaFree 绑定当前设备，必须先 set
    musa_ffi::set_device(device_id)?;

    unsafe {
        musa_ffi::check_musa(
            musa_ffi::musaFree(entry.ptr.as_ptr() as *mut std::ffi::c_void),
            "musaFree",
        )?;
    }

    // 内存统计：回收成功（ADR L3-28）
    crate::mem_stats::record_reclaimed(entry.size);
    Ok(())
}

// ============================================================
// 4. 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_count_starts_low() {
        // 队列是全局的，可能有其他测试残留；只断言能读取不 panic。
        let _ = pending_count();
    }

    #[test]
    fn reclaim_all_on_empty_is_ok() {
        // 清空状态下的 reclaim 不应报错。
        // 注意：全局队列可能含其他测试入队的 entry，先 drain 再断言空 reclaim。
        let _: Vec<_> = std::mem::take(&mut *queue().lock());
        assert!(reclaim_all().is_ok());
        assert_eq!(pending_count(), 0);
    }
}
