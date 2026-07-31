//! Buffer Pool：GPU 内存复用池（Phase C-lite）
//!
//! 按 (Device, SizeClass) 分桶缓存已释放的 GPU buffer，
//! 下次同 size-class 分配时直接复用，避免 musaMalloc/musaFree 开销。
//!
//! 设计约束：
//! - 仅缓存 GPU buffer（CPU buffer 走 std::alloc，开销可忽略）
//! - SizeClass = round_up_pow2(size)，最小 512 bytes
//! - 每设备缓存上限 512 MB（超出则 fallback 到 deferred_free）
//! - 复用时若 stream 不同，需 wait on stored event（跨 stream 安全）
//! - 仅在默认内存路径编译（stream-ordered 走 musaMallocAsync，无需池化）

#![cfg(not(feature = "stream-ordered"))]

use crate::device::Device;
use crate::stream::Event;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::OnceLock;

/// 每设备最大缓存字节数（512 MB）
const MAX_CACHED_PER_DEVICE: usize = 512 * 1024 * 1024;

/// 最小 size class（512 bytes）
const MIN_SIZE_CLASS: usize = 512;

/// 池中条目
struct PoolEntry {
    ptr: NonNull<u8>,
    /// 实际分配大小（可能 > size_class）
    actual_size: usize,
    /// 最后使用的 stream id（复用时判断是否需要 wait）
    last_stream_id: u64,
    /// 最后写操作事件（跨 stream 复用时需 wait）
    last_event: Option<Event>,
}

// GPU ptr 可跨线程（MUSA 内存模型）
unsafe impl Send for PoolEntry {}

/// 池内部状态
struct BufferPoolInner {
    /// (Device, SizeClass) → 空闲条目列表
    buckets: HashMap<(Device, usize), Vec<PoolEntry>>,
    /// 每设备已缓存字节数
    cached_per_device: HashMap<Device, usize>,
}

/// 全局池（懒初始化）
static POOL: OnceLock<Mutex<BufferPoolInner>> = OnceLock::new();

fn pool() -> &'static Mutex<BufferPoolInner> {
    POOL.get_or_init(|| {
        Mutex::new(BufferPoolInner {
            buckets: HashMap::new(),
            cached_per_device: HashMap::new(),
        })
    })
}

/// 将 size 向上取整到 2 的幂（最小 MIN_SIZE_CLASS）
fn size_class(size: usize) -> usize {
    let s = size.max(MIN_SIZE_CLASS);
    s.next_power_of_two()
}

/// 尝试从池中复用 buffer。
///
/// 命中条件：同 device + 同 size_class 有空闲条目。
/// 复用时：若 last_stream_id != 当前 stream id，wait on stored event。
///
/// 返回 `Some((ptr, actual_size))` 表示命中，`None` 表示需要新分配。
pub fn try_reuse(
    size: usize,
    device: &Device,
    stream_id: u64,
    stream: &crate::stream::Stream,
) -> Option<(NonNull<u8>, usize)> {
    // CPU 不走池
    if matches!(device, Device::Cpu) {
        return None;
    }

    let sc = size_class(size);
    let mut guard = pool().lock();

    let key = (device.clone(), sc);
    let bucket = guard.buckets.get_mut(&key)?;

    // 找到第一个 actual_size >= 请求 size 的条目（同 size_class 内可能有更小的）
    let idx = bucket.iter().position(|e| e.actual_size >= size)?;
    let entry = bucket.swap_remove(idx);

    // 更新设备缓存计数
    if let Some(cached) = guard.cached_per_device.get_mut(device) {
        *cached = cached.saturating_sub(entry.actual_size);
    }

    drop(guard); // 释放锁后再做 stream wait（避免持锁调 driver）

    // 跨 stream 复用：等待上次写操作完成
    if entry.last_stream_id != stream_id {
        if let Some(event) = &entry.last_event {
            if let Err(e) = stream.wait_event(event) {
                eprintln!("warn: buffer_pool reuse wait_event failed: {}", e);
            }
        }
    }

    Some((entry.ptr, entry.actual_size))
}

/// 将 buffer 归还到池中。
///
/// 若设备缓存已达上限，返回 false（调用方应 fallback 到 deferred_free）。
pub fn return_to_pool(
    ptr: NonNull<u8>,
    actual_size: usize,
    device: Device,
    stream_id: u64,
    event: Option<Event>,
) -> bool {
    if matches!(device, Device::Cpu) {
        return false;
    }

    let mut guard = pool().lock();

    // 检查设备缓存上限
    let cached = guard.cached_per_device.entry(device.clone()).or_insert(0);
    if *cached + actual_size > MAX_CACHED_PER_DEVICE {
        return false; // 池满，caller 走 deferred_free
    }
    *cached += actual_size;

    let sc = size_class(actual_size);
    let key = (device, sc);
    guard.buckets.entry(key).or_default().push(PoolEntry {
        ptr,
        actual_size,
        last_stream_id: stream_id,
        last_event: event,
    });

    true
}

/// 池统计：(cached_bytes, cached_count)
pub fn pool_stats() -> (usize, usize) {
    let guard = pool().lock();
    let bytes: usize = guard.cached_per_device.values().sum();
    let count: usize = guard.buckets.values().map(|v| v.len()).sum();
    (bytes, count)
}

/// 清空池（真正 musaFree 所有缓存 buffer）。
///
/// 用于测试或 shutdown。调用前需确保所有 stream 已 synchronize。
pub fn drain_all() {
    let mut guard = pool().lock();
    for ((device, _sc), entries) in guard.buckets.drain() {
        for entry in entries {
            if let Device::Musa(id) = device {
                let _ = crate::musa_ffi::set_device(id as i32);
                unsafe {
                    crate::musa_ffi::musaFree(entry.ptr.as_ptr() as *mut std::ffi::c_void);
                }
            }
        }
    }
    guard.cached_per_device.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_class_rounding() {
        assert_eq!(size_class(1), 512);
        assert_eq!(size_class(512), 512);
        assert_eq!(size_class(513), 1024);
        assert_eq!(size_class(1024), 1024);
        assert_eq!(size_class(1025), 2048);
        assert_eq!(size_class(4 * 1024 * 1024), 4 * 1024 * 1024);
        assert_eq!(size_class(3 * 1024 * 1024), 4 * 1024 * 1024);
    }
}
