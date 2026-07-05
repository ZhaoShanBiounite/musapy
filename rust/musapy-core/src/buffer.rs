//! Buffer 与 BufferRef：读写引用分离（ADR L1-10, L3-10）
//!
//! 职责：
//!   1. Buffer：GPU/CPU 内存块，含 raw ptr + device + 读写事件追踪
//!   2. BufferRef(Arc<Buffer>)：只读共享视图，op 输入自动降级为此类型
//!   3. 读写事件记录：record_read/record_write（Phase 3 接 musaEventRecord）
//!
//! Phase 2 约束（ADR L2-3）：
//!   - 不调用 musaMallocAsync/musaFreeAsync（Phase 3 实现）
//!   - 不调用 musaEventRecord/musaStreamWaitEvent（Phase 3 实现）
//!   - raw ptr 用 null 占位，alloc 由 Phase 3 填充
//!   - 但 Arc 语义、读写引用分离、别名检测逻辑必须定型
//!
//! 设计依据：
//!   - L1-10：Arc<Buffer> 可写唯一所有权 / BufferRef 只读共享
//!   - L3-10：释放流选择策略 b（最后使用的流）
//!   - L2-5：同一 BufferRef 不能既是输入又是 out（编译期检测）

use crate::device::Device;
use crate::stream::Stream;
use parking_lot::Mutex;
use std::fmt;
use std::sync::Arc;

// ============================================================
// 1. Event（Phase 3 占位）
// ============================================================

/// MUSA 事件（ADR L3-10，用于跨流同步）。
///
/// **Phase 2 占位**：`raw` 为 0，不调用 musaEventCreate/Record。
/// **Phase 3**：替换为真实 `musaEvent_t` + create/record/wait。
#[derive(Debug)]
pub struct Event {
    #[allow(dead_code)]
    raw: usize,
}

impl Event {
    /// Phase 2 占位构造。Phase 3 调用 musaEventCreate。
    pub fn new() -> Self {
        Self { raw: 0 }
    }
}

impl Default for Event {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// 2. Buffer（ADR L1-10, L3-10）
// ============================================================

/// GPU/CPU 内存块（ADR L1-10）。
///
/// **所有权语义**：
/// - `Arc<Buffer>`：可写，唯一所有权（逻辑上，通过 alias 检测保证）
/// - `BufferRef(Arc<Buffer>)`：只读共享视图
///
/// **读写事件追踪**（ADR L3-10）：
/// - `last_write_event`：最近一次写操作的事件
/// - `read_events`：所有未完成读操作的事件（释放流等待这些事件后才能 free）
/// - `dealloc_stream`：释放流（策略 b：最后使用的流，跨流使用时更新）
///
/// **Phase 2 约束**：
/// - `ptr` 为 null（Phase 3 用 musaMallocAsync 分配真实内存）
/// - `record_read`/`record_write` 只记录 Event 占位（Phase 3 接 musaEventRecord）
/// - `Drop` 不调用 musaFreeAsync（Phase 3 实现 stream-ordered free）
#[derive(Debug)]
pub struct Buffer {
    /// 内存指针（Phase 3 由 musaMallocAsync 填充）。
    /// Phase 2 为 null，仅用于类型结构定型。
    ptr: std::ptr::NonNull<u8>,
    /// 字节大小。
    size: usize,
    /// 所属设备。
    device: Device,
    /// 释放流（策略 b：最后使用的流）。
    /// Drop 时在此流上执行 stream-ordered free。
    dealloc_stream: Mutex<Option<Arc<Stream>>>,
    /// 最近一次写操作的事件（Drop 时释放流需等待）。
    last_write_event: Mutex<Option<Event>>,
    /// 所有未完成读操作的事件（Drop 时释放流需等待）。
    /// L3-10 优化：只存尚未被 dealloc_stream 等待过的事件。
    read_events: Mutex<Vec<Event>>,
}

// Buffer 的 Drop：Phase 2 无操作（不调用 musaFreeAsync）。
// Phase 3 会实现 stream-ordered free（L3-9, L3-10）。
// 这里不 impl Drop，留待 Phase 3。

impl Buffer {
    /// Phase 2 占位构造（不分配真实内存）。
    ///
    /// **Phase 3**：替换为 `alloc(size, device, stream)`，调用 musaMallocAsync。
    /// 这里用 dangling ptr 占位，让 NonNull 字段类型满足。
    ///
    /// # Safety
    /// 占位构造，ptr 为 dangling，不实际读写。Phase 3 替换为真实分配。
    pub fn placeholder(size: usize, device: Device) -> Self {
        // Phase 2 占位：dangling ptr，不分配真实内存。
        // Phase 3 替换为 musaMallocAsync 返回的真实 ptr。
        let ptr = unsafe { std::ptr::NonNull::dangling() };
        Self {
            ptr,
            size,
            device,
            dealloc_stream: Mutex::new(None),
            last_write_event: Mutex::new(None),
            read_events: Mutex::new(Vec::new()),
        }
    }

    /// 字节大小。
    pub fn size(&self) -> usize {
        self.size
    }

    /// 所属设备。
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 原始指针（Phase 3 kernel launch 用）。
    pub fn ptr(&self) -> std::ptr::NonNull<u8> {
        self.ptr
    }

    /// 记录读操作（ADR L3-10）。
    ///
    /// 在 reader_stream 上记录事件，加入 read_events。
    /// **Phase 2 占位**：只创建 Event 占位，不调用 musaEventRecord。
    /// **Phase 3**：
    ///   1. 在 reader_stream 上 musaEventRecord
    ///   2. 若 reader_stream != dealloc_stream，更新 dealloc_stream = reader_stream
    ///   3. 事件加入 read_events（L3-10 优化：已等待过的不存）
    pub fn record_read(&self, reader_stream: &Arc<Stream>) {
        let _ = reader_stream; // Phase 2 暂未使用
        let event = Event::new();
        self.read_events.lock().push(event);
    }

    /// 记录写操作（ADR L3-10）。
    ///
    /// 在 writer_stream 上记录事件，替换 last_write_event。
    /// **Phase 2 占位**：只创建 Event 占位。
    /// **Phase 3**：
    ///   1. 在 writer_stream 上 musaEventRecord
    ///   2. 更新 dealloc_stream = writer_stream（策略 b：最后使用的流）
    ///   3. 替换 last_write_event
    pub fn record_write(&self, writer_stream: &Arc<Stream>) {
        let _ = writer_stream; // Phase 2 暂未使用
        let event = Event::new();
        *self.last_write_event.lock() = Some(event);
    }

    /// 当前 read_events 数量（调试用）。
    pub fn read_event_count(&self) -> usize {
        self.read_events.lock().len()
    }

    /// 是否有 last_write_event（调试用）。
    pub fn has_write_event(&self) -> bool {
        self.last_write_event.lock().is_some()
    }
}

impl fmt::Display for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Buffer(size={}, device={})",
            self.size, self.device
        )
    }
}

// ============================================================
// 3. BufferRef（ADR L1-10）
// ============================================================

/// 只读共享视图（ADR L1-10）。
///
/// Op 输入自动降级为 BufferRef；输出是新 Buffer。
/// 这使得 kernel 可以安全使用 `__restrict__`（编译器可假设无别名）。
///
/// **别名检测**（ADR L2-5）：同一 BufferRef 不能既是输入又是 out。
/// 通过 BufferRef 之间的 PartialEq 比较 Arc 指针实现。
#[derive(Debug, Clone)]
pub struct BufferRef(Arc<Buffer>);

impl BufferRef {
    /// 从 Arc<Buffer> 创建只读引用。
    pub fn new(buffer: Arc<Buffer>) -> Self {
        Self(buffer)
    }

    /// 访问底层 Buffer。
    pub fn buffer(&self) -> &Buffer {
        &self.0
    }

    /// 访问底层 Arc<Buffer>（kernel launch 需要 Arc 解引用到 ptr）。
    pub fn arc(&self) -> &Arc<Buffer> {
        &self.0
    }
}

impl fmt::Display for BufferRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BufferRef({})", self.0)
    }
}

// PartialEq：比较 Arc 指针地址（用于别名检测，ADR L2-5）
impl PartialEq for BufferRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for BufferRef {}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buffer(size: usize, device: Device) -> Arc<Buffer> {
        Arc::new(Buffer::placeholder(size, device))
    }

    // --- Buffer 基本属性 ---

    #[test]
    fn buffer_size_and_device() {
        let b = make_buffer(1024, Device::Musa(0));
        assert_eq!(b.size(), 1024);
        assert_eq!(*b.device(), Device::Musa(0));
    }

    #[test]
    fn buffer_display() {
        let b = make_buffer(512, Device::Cpu);
        assert_eq!(b.to_string(), "Buffer(size=512, device=cpu)");
    }

    #[test]
    fn buffer_ptr_nonnull() {
        // Phase 2 占位 ptr 是 dangling，但不为 null
        let b = make_buffer(64, Device::Musa(0));
        assert!(!b.ptr().as_ptr().is_null());
    }

    // --- Buffer 读写事件 ---

    #[test]
    fn buffer_initial_no_events() {
        let b = make_buffer(64, Device::Musa(0));
        assert_eq!(b.read_event_count(), 0);
        assert!(!b.has_write_event());
    }

    #[test]
    fn buffer_record_read_adds_event() {
        let stream = Arc::new(Stream::new(Device::Musa(0), 0));
        let b = make_buffer(64, Device::Musa(0));
        b.record_read(&stream);
        b.record_read(&stream);
        assert_eq!(b.read_event_count(), 2);
    }

    #[test]
    fn buffer_record_write_sets_event() {
        let stream = Arc::new(Stream::new(Device::Musa(0), 0));
        let b = make_buffer(64, Device::Musa(0));
        assert!(!b.has_write_event());
        b.record_write(&stream);
        assert!(b.has_write_event());
    }

    #[test]
    fn buffer_record_write_replaces_previous() {
        let stream = Arc::new(Stream::new(Device::Musa(0), 0));
        let b = make_buffer(64, Device::Musa(0));
        b.record_write(&stream);
        b.record_write(&stream);
        // last_write_event 只保留一个
        assert!(b.has_write_event());
    }

    // --- BufferRef 基本属性 ---

    #[test]
    fn buffer_ref_display() {
        let b = make_buffer(128, Device::Musa(1));
        let r = BufferRef::new(b);
        assert_eq!(r.to_string(), "BufferRef(Buffer(size=128, device=musa:1))");
    }

    #[test]
    fn buffer_ref_access_buffer() {
        let b = make_buffer(256, Device::Cpu);
        let r = BufferRef::new(b.clone());
        assert_eq!(r.buffer().size(), 256);
        assert_eq!(*r.buffer().device(), Device::Cpu);
    }

    // --- BufferRef 别名检测（ADR L2-5）---

    #[test]
    fn buffer_ref_eq_same_arc() {
        let b = make_buffer(64, Device::Musa(0));
        let r1 = BufferRef::new(b.clone());
        let r2 = BufferRef::new(b.clone());
        // 同一 Arc 的两个 BufferRef 应相等
        assert_eq!(r1, r2);
    }

    #[test]
    fn buffer_ref_neq_different_arc() {
        let b1 = make_buffer(64, Device::Musa(0));
        let b2 = make_buffer(64, Device::Musa(0)); // 相同 size/device，但不同 Arc
        let r1 = BufferRef::new(b1);
        let r2 = BufferRef::new(b2);
        // 不同 Arc（即使内容相同）应不相等
        assert_ne!(r1, r2);
    }

    #[test]
    fn buffer_ref_eq_after_clone() {
        let b = make_buffer(64, Device::Musa(0));
        let r1 = BufferRef::new(b);
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }

    // --- 模拟别名检测场景（ADR L2-5）---

    #[test]
    fn alias_detection_scenario() {
        // 模拟 ms.add(a, b, out=a) 的别名检测
        // a 既是输入又是 out，应该被检测到
        let buf = make_buffer(64, Device::Musa(0));
        let a = BufferRef::new(buf.clone());
        let b = BufferRef::new(make_buffer(64, Device::Musa(0)));
        // out 是 buf 的可写引用
        let out_ref = BufferRef::new(buf);

        // 别名检测：out_ref 与 a 相等 → 别名！
        assert_eq!(out_ref, a, "alias detected: out is same as input a");
        // out_ref 与 b 不相等 → 无别名
        assert_ne!(out_ref, b);
    }

    #[test]
    fn no_alias_scenario() {
        // 模拟 ms.add(a, b, out=c) 的正常场景
        let a = BufferRef::new(make_buffer(64, Device::Musa(0)));
        let b = BufferRef::new(make_buffer(64, Device::Musa(0)));
        let c = BufferRef::new(make_buffer(64, Device::Musa(0)));

        // 三个都互不相等
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    // --- Arc 引用计数 ---

    #[test]
    fn buffer_arc_refcount() {
        let b = make_buffer(64, Device::Musa(0));
        assert_eq!(Arc::strong_count(&b), 1);

        let r1 = BufferRef::new(b.clone());
        assert_eq!(Arc::strong_count(&b), 2);

        let r2 = r1.clone();
        assert_eq!(Arc::strong_count(&b), 3);

        drop(r1);
        drop(r2);
        assert_eq!(Arc::strong_count(&b), 1);
    }
}