//! 内存统计原子计数器（ADR L3-28）
//!
//! 零锁设计：所有计数器用 `AtomicUsize`，适合频繁监控。
//! 由 `Buffer::alloc` / `Buffer::drop` / `deferred_free` 插桩维护。
//!
//! 计数器语义：
//!   - `allocated_*`：当前存活 Buffer 占用的内存（alloc 增，drop 减）
//!   - `cached_*`：deferred-free 队列中待释放的内存（enqueue 增，reclaim 减）
//!   - `peak_bytes`：`allocated_bytes` 的历史峰值（CAS 更新）

use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================
// 全局原子计数器
// ============================================================

/// 当前存活 Buffer 的总字节数。
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// 当前存活 Buffer 的数量。
static ALLOCATED_BUFFERS: AtomicUsize = AtomicUsize::new(0);

/// deferred-free 队列中的总字节数（已 drop 但未 reclaim）。
static CACHED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// deferred-free 队列中的 buffer 数量。
static CACHED_BUFFERS: AtomicUsize = AtomicUsize::new(0);

/// `allocated_bytes` 的历史峰值。
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

// ============================================================
// 插桩 API（由 Buffer / deferred_free 调用）
// ============================================================

/// 记录一次成功分配（`Buffer::alloc` 成功后调用，size > 0）。
///
/// 递增 `allocated_bytes` / `allocated_buffers`，并 CAS 更新 `peak_bytes`。
pub fn record_alloc(size: usize) {
    ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
    ALLOCATED_BUFFERS.fetch_add(1, Ordering::Relaxed);

    // CAS 更新峰值
    let current = ALLOCATED_BYTES.load(Ordering::Relaxed);
    loop {
        let peak = PEAK_BYTES.load(Ordering::Relaxed);
        if current <= peak {
            break;
        }
        match PEAK_BYTES.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(_) => continue, // 被其他线程更新，重试
        }
    }
}

/// 记录一次释放（`Buffer::drop` 时调用）。
///
/// 递减 `allocated_bytes` / `allocated_buffers`。
pub fn record_dealloc(size: usize) {
    ALLOCATED_BYTES.fetch_sub(size, Ordering::Relaxed);
    ALLOCATED_BUFFERS.fetch_sub(1, Ordering::Relaxed);
}

/// 记录一次入队（`deferred_free::enqueue` 时调用）。
///
/// 递增 `cached_bytes` / `cached_buffers`。
pub fn record_cached(size: usize) {
    CACHED_BYTES.fetch_add(size, Ordering::Relaxed);
    CACHED_BUFFERS.fetch_add(1, Ordering::Relaxed);
}

/// 记录一次回收（`deferred_free::reclaim_one` 成功后调用）。
///
/// 递减 `cached_bytes` / `cached_buffers`。
pub fn record_reclaimed(size: usize) {
    CACHED_BYTES.fetch_sub(size, Ordering::Relaxed);
    CACHED_BUFFERS.fetch_sub(1, Ordering::Relaxed);
}

// ============================================================
// 查询 API
// ============================================================

/// 内存统计快照。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// 当前存活 Buffer 的总字节数。
    pub allocated_bytes: usize,
    /// 当前存活 Buffer 的数量。
    pub allocated_buffers: usize,
    /// deferred-free 队列中的总字节数。
    pub cached_bytes: usize,
    /// deferred-free 队列中的 buffer 数量。
    pub cached_buffers: usize,
    /// `allocated_bytes` 的历史峰值。
    pub peak_bytes: usize,
}

/// 读取所有计数器的快照（Relaxed 序，非精确一致但足够监控用）。
pub fn snapshot() -> MemorySnapshot {
    MemorySnapshot {
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        allocated_buffers: ALLOCATED_BUFFERS.load(Ordering::Relaxed),
        cached_bytes: CACHED_BYTES.load(Ordering::Relaxed),
        cached_buffers: CACHED_BUFFERS.load(Ordering::Relaxed),
        peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_dealloc_roundtrip() {
        let before = snapshot();
        record_alloc(1024);
        let mid = snapshot();
        assert_eq!(mid.allocated_bytes, before.allocated_bytes + 1024);
        assert_eq!(mid.allocated_buffers, before.allocated_buffers + 1);

        record_dealloc(1024);
        let after = snapshot();
        assert_eq!(after.allocated_bytes, before.allocated_bytes);
        assert_eq!(after.allocated_buffers, before.allocated_buffers);
    }

    #[test]
    fn cached_reclaimed_roundtrip() {
        let before = snapshot();
        record_cached(512);
        let mid = snapshot();
        assert_eq!(mid.cached_bytes, before.cached_bytes + 512);
        assert_eq!(mid.cached_buffers, before.cached_buffers + 1);

        record_reclaimed(512);
        let after = snapshot();
        assert_eq!(after.cached_bytes, before.cached_bytes);
        assert_eq!(after.cached_buffers, before.cached_buffers);
    }

    #[test]
    fn peak_tracks_maximum() {
        let before = snapshot();
        // 分配一大块，峰值应更新
        record_alloc(999_999);
        let mid = snapshot();
        assert!(mid.peak_bytes >= before.peak_bytes);
        assert!(mid.peak_bytes >= 999_999);

        // 释放后峰值不降
        record_dealloc(999_999);
        let after = snapshot();
        assert_eq!(after.peak_bytes, mid.peak_bytes);
    }
}
