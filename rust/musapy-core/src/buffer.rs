//! Buffer 与 BufferRef：读写引用分离（ADR L1-10, L3-9, L3-10）
//!
//! Phase 3 第二批：真实内存分配/释放
//!   - Buffer::alloc 调用 musaMallocAsync
//!   - Buffer::Drop 实现 stream-ordered free（策略 b）
//!   - record_read/record_write 接真实 musaEventRecord + 更新 dealloc_stream
//!
//! 设计依据：
//!   - L1-10：Arc<Buffer> 可写唯一所有权 / BufferRef 只读共享
//!   - L3-9：stream-ordered alloc/free（musaMallocAsync/musaFreeAsync）
//!   - L3-10：释放流选择策略 b（最后使用的流）
//!   - L2-5：同一 BufferRef 不能既是输入又是 out（编译期检测）

use crate::device::Device;
use crate::error::{MemoryError, Result};
use crate::musa_ffi;
use crate::stream::{Event, Stream};
use parking_lot::Mutex;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

// ============================================================
// 1. Buffer（ADR L1-10, L3-9, L3-10）
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
/// **Phase 3 实现**：
/// - `alloc()` 调用 `musaMallocAsync` 分配 GPU 内存
/// - `Drop` 在 `dealloc_stream` 上等待所有事件后调用 `musaFreeAsync`
/// - CPU 设备：用 `std::alloc` 分配主机内存
pub struct Buffer {
    /// 内存指针。None 表示未分配或已释放（防止 Drop 重复释放）。
    ptr: Option<NonNull<u8>>,
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

// GPU 内存指针可跨线程访问（MUSA 内存模型），可变状态已用 Mutex 保护。
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// 真实分配 GPU/CPU 内存（ADR L3-9, L3-11）。
    ///
    /// - MUSA 设备：
    ///   - 默认路径：`musaSetDevice(id)` + `musaMalloc`（同步，所有 SDK 版本可用）
    ///   - stream-ordered feature：`musaMallocAsync(ptr, size, stream)`（仅 5.x+）
    /// - CPU 设备：`std::alloc::alloc`（8 字节对齐）
    ///
    /// 分配失败返回 `MemoryError::OutOfMemory`。
    pub fn alloc(size: usize, device: Device, stream: &Arc<Stream>) -> Result<Self> {
        if size == 0 {
            // 0 字节分配：返回 null ptr 的占位 Buffer
            return Ok(Self {
                ptr: None,
                size: 0,
                device,
                dealloc_stream: Mutex::new(Some(stream.clone())),
                last_write_event: Mutex::new(None),
                read_events: Mutex::new(Vec::new()),
            });
        }

        let ptr = match &device {
            Device::Cpu => Self::alloc_cpu(size)?,
            Device::Musa(id) => Self::alloc_musa(*id, size, stream)?,
        };

        // 内存统计插桩（ADR L3-28）
        crate::mem_stats::record_alloc(size);

        Ok(Self {
            ptr: Some(ptr),
            size,
            device,
            dealloc_stream: Mutex::new(Some(stream.clone())),
            last_write_event: Mutex::new(None),
            read_events: Mutex::new(Vec::new()),
        })
    }

    /// CPU 内存分配（8 字节对齐）。
    fn alloc_cpu(size: usize) -> Result<NonNull<u8>> {
        let layout = std::alloc::Layout::from_size_align(size, 8)
            .map_err(|_| MemoryError::OutOfMemory(format!("invalid layout: {} bytes", size)))?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        NonNull::new(ptr).ok_or_else(|| {
            MemoryError::OutOfMemory(format!("CPU alloc failed: {} bytes", size)).into()
        })
    }

    /// MUSA GPU 内存分配。
    ///
    /// 双路径（ADR L3-9, L3-11）：
    /// - stream-ordered feature：`musaMallocAsync`（stream-ordered，5.x+）
    /// - 默认：`musaSetDevice(id)` + `musaMalloc`（同步，3.x/4.x/5.x 全可用）
    ///
    /// `musaMalloc` 绑定调用线程的当前设备，所以必须先 `musaSetDevice`。
    #[cfg(feature = "stream-ordered")]
    fn alloc_musa(_device_id: u32, size: usize, stream: &Arc<Stream>) -> Result<NonNull<u8>> {
        let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaMallocAsync(&mut dev_ptr, size, stream.raw()),
                "musaMallocAsync",
            )?;
        }
        Self::check_musa_ptr(dev_ptr, size)
    }

    /// MUSA GPU 内存分配（默认路径，ADR L3-11）。
    #[cfg(not(feature = "stream-ordered"))]
    fn alloc_musa(device_id: u32, size: usize, _stream: &Arc<Stream>) -> Result<NonNull<u8>> {
        // musaMalloc 绑定当前设备，必须先 set（修复 Musa(1) 落到设备 0 的隐患）
        musa_ffi::set_device(device_id as i32)?;
        let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            musa_ffi::check_musa(musa_ffi::musaMalloc(&mut dev_ptr, size), "musaMalloc")?;
        }
        Self::check_musa_ptr(dev_ptr, size)
    }

    /// 校验 musaMalloc/musaMallocAsync 返回的指针非空，转 NonNull。
    fn check_musa_ptr(dev_ptr: *mut std::ffi::c_void, size: usize) -> Result<NonNull<u8>> {
        if dev_ptr.is_null() {
            return Err(MemoryError::OutOfMemory(format!(
                "MUSA alloc returned null: {} bytes",
                size
            ))
            .into());
        }
        // dev_ptr 已检查非 null，NonNull::new_unchecked 安全
        Ok(unsafe { NonNull::new_unchecked(dev_ptr as *mut u8) })
    }

    /// 占位构造（仅测试用，不分配真实内存）。
    ///
    /// 生产代码应使用 `alloc()`。这个方法保留给不需要真实内存的单元测试。
    pub fn placeholder(size: usize, device: Device) -> Self {
        Self {
            ptr: None,
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

    /// 原始指针（kernel launch 用）。
    ///
    /// 返回 `None` 表示未分配（placeholder 或 0 字节）。
    pub fn ptr(&self) -> Option<NonNull<u8>> {
        self.ptr
    }

    /// 记录读操作（ADR L3-10）。
    ///
    /// 在 reader_stream 上记录事件，加入 read_events。
    /// 若 reader_stream != dealloc_stream，更新 dealloc_stream = reader_stream（策略 b）。
    pub fn record_read(&self, reader_stream: &Arc<Stream>) {
        match Event::new() {
            Ok(event) => {
                if let Err(e) = event.record(reader_stream) {
                    eprintln!("warn: record_read event record failed: {}", e);
                    return;
                }
                self.read_events.lock().push(event);
                // 策略 b：更新释放流为最后使用的流
                *self.dealloc_stream.lock() = Some(reader_stream.clone());
            }
            Err(e) => {
                eprintln!("warn: record_read event create failed: {}", e);
            }
        }
    }

    /// 记录写操作（ADR L3-10）。
    ///
    /// 在 writer_stream 上记录事件，替换 last_write_event。
    /// 更新 dealloc_stream = writer_stream（策略 b）。
    pub fn record_write(&self, writer_stream: &Arc<Stream>) {
        match Event::new() {
            Ok(event) => {
                if let Err(e) = event.record(writer_stream) {
                    eprintln!("warn: record_write event record failed: {}", e);
                    return;
                }
                *self.last_write_event.lock() = Some(event);
                // 策略 b：更新释放流为最后使用的流
                *self.dealloc_stream.lock() = Some(writer_stream.clone());
            }
            Err(e) => {
                eprintln!("warn: record_write event create failed: {}", e);
            }
        }
    }

    /// 当前 read_events 数量（调试用）。
    pub fn read_event_count(&self) -> usize {
        self.read_events.lock().len()
    }

    /// 是否有 last_write_event（调试用）。
    pub fn has_write_event(&self) -> bool {
        self.last_write_event.lock().is_some()
    }

    /// 若此 buffer 有未完成的写操作事件，让 target_stream 等待之（ADR L1-8 自动 stream wait）。
    ///
    /// 用于 op 执行时，当输入 buffer 是在另一个 stream 上写入的，
    /// 让输出 stream 自动等待输入的写操作完成。
    /// CPU stream 的 `wait_event` 是 no-op，不影响 CPU 路径。
    pub fn wait_last_write_on(&self, target_stream: &Arc<Stream>) -> Result<()> {
        let guard = self.last_write_event.lock();
        if let Some(event) = guard.as_ref() {
            target_stream.wait_event(event)?;
        }
        Ok(())
    }
}

/// Buffer 释放（ADR L3-9, L3-10 策略 b, L3-11）。
///
/// 释放流程（两路径共用的前 3 步）：
/// 1. 取出 dealloc_stream（最后使用的流，策略 b）
/// 2. 在 dealloc_stream 上等待所有 read_events
/// 3. 在 dealloc_stream 上等待 last_write_event
///
/// 然后按路径分流：
/// - stream-ordered feature：`musaFreeAsync(ptr, dealloc_stream)`（立即 stream-ordered free）
/// - 默认路径：`deferred_free::enqueue(ptr, device)`（入队，等 synchronize 批量 musaFree）
///
/// 等待事件保证：释放前所有使用此 buffer 的操作都已完成。
/// CPU buffer 用 std::alloc::dealloc 释放。
impl Drop for Buffer {
    fn drop(&mut self) {
        let ptr = match self.ptr.take() {
            Some(p) => p,
            None => return, // 未分配或已释放
        };

        // 取出 dealloc_stream
        let dealloc_stream = self.dealloc_stream.lock().clone();

        match &self.device {
            Device::Cpu => {
                // CPU：直接 dealloc（不需要 stream 同步）
                crate::mem_stats::record_dealloc(self.size);
                let layout = std::alloc::Layout::from_size_align(self.size, 8).unwrap();
                unsafe {
                    std::alloc::dealloc(ptr.as_ptr(), layout);
                }
            }
            Device::Musa(_) => {
                if let Some(stream) = dealloc_stream {
                    // 等待所有 read_events（策略 b：在最后使用的流上等待）
                    let read_events: Vec<Event> = self.read_events.lock().drain(..).collect();
                    for ev in read_events {
                        if let Err(e) = stream.wait_event(&ev) {
                            eprintln!("warn: free wait read_event failed: {}", e);
                        }
                    }
                    // 等待 last_write_event
                    if let Some(ev) = self.last_write_event.lock().take() {
                        if let Err(e) = stream.wait_event(&ev) {
                            eprintln!("warn: free wait write_event failed: {}", e);
                        }
                    }

                    // 路径分流
                    #[cfg(feature = "stream-ordered")]
                    {
                        // stream-ordered free（5.x+）：musaFreeAsync 立即流序释放
                        crate::mem_stats::record_dealloc(self.size);
                        unsafe {
                            if let Err(e) = musa_ffi::check_musa(
                                musa_ffi::musaFreeAsync(
                                    ptr.as_ptr() as *mut std::ffi::c_void,
                                    stream.raw(),
                                ),
                                "musaFreeAsync",
                            ) {
                                eprintln!("warn: musaFreeAsync failed: {}", e);
                            }
                        }
                    }
                    #[cfg(not(feature = "stream-ordered"))]
                    {
                        // 默认路径（3.x/4.x/5.x）：入 deferred-free 队列，
                        // 等 Stream::synchronize 成功后批量 musaFree（ADR L3-11）。
                        // 入队前已 wait events，synchronize 后 buffer 必不再被使用。
                        crate::mem_stats::record_dealloc(self.size);
                        crate::deferred_free::enqueue(ptr, self.device.clone(), self.size);
                    }
                } else {
                    // 无 dealloc_stream（placeholder 或未初始化）：跳过 free
                    eprintln!("warn: Buffer dropped without dealloc_stream, skipping free");
                }
            }
        }
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("ptr", &format_args!("{:?}", self.ptr))
            .field("size", &self.size)
            .field("device", &self.device)
            .field("has_write_event", &self.has_write_event())
            .field("read_event_count", &self.read_event_count())
            .finish()
    }
}

impl fmt::Display for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Buffer(size={}, device={})", self.size, self.device)
    }
}

// ============================================================
// 2. BufferRef（ADR L1-10，不变）
// ============================================================

#[derive(Clone)]
pub struct BufferRef(Arc<Buffer>);

impl BufferRef {
    pub fn new(buffer: Arc<Buffer>) -> Self {
        Self(buffer)
    }

    pub fn buffer(&self) -> &Buffer {
        &self.0
    }

    pub fn arc(&self) -> &Arc<Buffer> {
        &self.0
    }
}

impl fmt::Debug for BufferRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BufferRef").field(&self.0).finish()
    }
}

impl fmt::Display for BufferRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BufferRef({})", self.0)
    }
}

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

    fn make_stream(device: Device) -> Arc<Stream> {
        Arc::new(Stream::new(device, 0).unwrap())
    }

    // --- Buffer::alloc（真实分配）---

    #[test]
    fn buffer_alloc_musa() {
        let stream = make_stream(Device::Musa(0));
        let buf = Buffer::alloc(1024, Device::Musa(0), &stream).expect("alloc failed");
        assert_eq!(buf.size(), 1024);
        assert_eq!(*buf.device(), Device::Musa(0));
        assert!(buf.ptr().is_some(), "ptr should be non-null after alloc");
    }

    #[test]
    fn buffer_alloc_cpu() {
        let stream = make_stream(Device::Cpu);
        let buf = Buffer::alloc(512, Device::Cpu, &stream).expect("alloc failed");
        assert_eq!(buf.size(), 512);
        assert_eq!(*buf.device(), Device::Cpu);
        assert!(buf.ptr().is_some());
    }

    #[test]
    fn buffer_alloc_zero_size() {
        let stream = make_stream(Device::Musa(0));
        let buf = Buffer::alloc(0, Device::Musa(0), &stream).expect("alloc failed");
        assert_eq!(buf.size(), 0);
        assert!(buf.ptr().is_none(), "0-size buffer has null ptr");
    }

    #[test]
    fn buffer_alloc_and_drop_musa() {
        // 验证 alloc + drop 不崩溃（stream-ordered free）
        let stream = make_stream(Device::Musa(0));
        {
            let _buf = Buffer::alloc(2048, Device::Musa(0), &stream).expect("alloc failed");
            // _buf drop 时执行 stream-ordered free
        }
        // 同步流，确保 free 完成
        stream.synchronize().unwrap();
    }

    #[test]
    fn buffer_alloc_and_drop_cpu() {
        let stream = make_stream(Device::Cpu);
        {
            let _buf = Buffer::alloc(256, Device::Cpu, &stream).expect("alloc failed");
        }
        // CPU buffer drop 时直接 dealloc
    }

    // --- placeholder（保留给不需要真实内存的测试）---

    #[test]
    fn buffer_placeholder() {
        let b = Buffer::placeholder(64, Device::Musa(0));
        assert_eq!(b.size(), 64);
        assert!(b.ptr().is_none(), "placeholder has null ptr");
    }

    // --- 读写事件 + dealloc_stream 更新 ---

    #[test]
    fn buffer_record_read_updates_dealloc_stream() {
        let stream1 = make_stream(Device::Musa(0));
        let stream2 = make_stream(Device::Musa(0));
        let buf = Buffer::alloc(64, Device::Musa(0), &stream1).unwrap();

        // 初始 dealloc_stream = stream1（alloc 时设置）
        buf.record_read(&stream2);
        // record_read 后 dealloc_stream 应更新为 stream2
        // （通过 drop 不崩溃间接验证）
    }

    #[test]
    fn buffer_record_write_updates_dealloc_stream() {
        let stream1 = make_stream(Device::Musa(0));
        let stream2 = make_stream(Device::Musa(0));
        let buf = Buffer::alloc(64, Device::Musa(0), &stream1).unwrap();

        buf.record_write(&stream2);
        // dealloc_stream 应更新为 stream2
    }

    #[test]
    fn buffer_record_read_adds_event() {
        let stream = make_stream(Device::Musa(0));
        let buf = Buffer::alloc(64, Device::Musa(0), &stream).unwrap();
        assert_eq!(buf.read_event_count(), 0);
        buf.record_read(&stream);
        assert_eq!(buf.read_event_count(), 1);
    }

    #[test]
    fn buffer_record_write_sets_event() {
        let stream = make_stream(Device::Musa(0));
        let buf = Buffer::alloc(64, Device::Musa(0), &stream).unwrap();
        assert!(!buf.has_write_event());
        buf.record_write(&stream);
        assert!(buf.has_write_event());
    }

    #[test]
    fn buffer_record_write_replaces_previous() {
        let stream = make_stream(Device::Musa(0));
        let buf = Buffer::alloc(64, Device::Musa(0), &stream).unwrap();
        buf.record_write(&stream);
        buf.record_write(&stream);
        assert!(buf.has_write_event());
    }

    // --- 跨流场景（策略 b 核心）---

    #[test]
    fn buffer_cross_stream_free_safe() {
        // 在 stream1 分配，在 stream2 读写，drop 时应在 stream2 释放
        let stream1 = make_stream(Device::Musa(0));
        let stream2 = make_stream(Device::Musa(0));

        {
            let buf = Buffer::alloc(1024, Device::Musa(0), &stream1).unwrap();
            // 在 stream2 上读
            buf.record_read(&stream2);
            // 在 stream2 上写
            buf.record_write(&stream2);
            // drop：dealloc_stream 已更新为 stream2
            // 会在 stream2 上等待事件后 free
        }

        // 同步两个流，确保所有操作完成
        stream1.synchronize().unwrap();
        stream2.synchronize().unwrap();
    }

    // --- BufferRef（不变）---

    #[test]
    fn buffer_ref_display() {
        let stream = make_stream(Device::Musa(0));
        let b = Arc::new(Buffer::alloc(128, Device::Musa(0), &stream).unwrap());
        let r = BufferRef::new(b);
        assert_eq!(r.to_string(), "BufferRef(Buffer(size=128, device=musa:0))");
    }

    #[test]
    fn buffer_ref_access_buffer() {
        let stream = make_stream(Device::Cpu);
        let b = Arc::new(Buffer::alloc(256, Device::Cpu, &stream).unwrap());
        let r = BufferRef::new(b.clone());
        assert_eq!(r.buffer().size(), 256);
        assert_eq!(*r.buffer().device(), Device::Cpu);
    }

    #[test]
    fn buffer_ref_eq_same_arc() {
        let stream = make_stream(Device::Musa(0));
        let b = Arc::new(Buffer::alloc(64, Device::Musa(0), &stream).unwrap());
        let r1 = BufferRef::new(b.clone());
        let r2 = BufferRef::new(b.clone());
        assert_eq!(r1, r2);
    }

    #[test]
    fn buffer_ref_neq_different_arc() {
        let stream = make_stream(Device::Musa(0));
        let b1 = Arc::new(Buffer::alloc(64, Device::Musa(0), &stream).unwrap());
        let b2 = Arc::new(Buffer::alloc(64, Device::Musa(0), &stream).unwrap());
        let r1 = BufferRef::new(b1);
        let r2 = BufferRef::new(b2);
        assert_ne!(r1, r2);
    }

    #[test]
    fn buffer_ref_eq_after_clone() {
        let stream = make_stream(Device::Musa(0));
        let b = Arc::new(Buffer::alloc(64, Device::Musa(0), &stream).unwrap());
        let r1 = BufferRef::new(b);
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }

    #[test]
    fn alias_detection_scenario() {
        let stream = make_stream(Device::Musa(0));
        let buf = Arc::new(Buffer::alloc(64, Device::Musa(0), &stream).unwrap());
        let a = BufferRef::new(buf.clone());
        let b = BufferRef::new(Arc::new(
            Buffer::alloc(64, Device::Musa(0), &stream).unwrap(),
        ));
        let out_ref = BufferRef::new(buf);

        assert_eq!(out_ref, a, "alias detected: out is same as input a");
        assert_ne!(out_ref, b);
    }

    #[test]
    fn no_alias_scenario() {
        let stream = make_stream(Device::Musa(0));
        let a = BufferRef::new(Arc::new(
            Buffer::alloc(64, Device::Musa(0), &stream).unwrap(),
        ));
        let b = BufferRef::new(Arc::new(
            Buffer::alloc(64, Device::Musa(0), &stream).unwrap(),
        ));
        let c = BufferRef::new(Arc::new(
            Buffer::alloc(64, Device::Musa(0), &stream).unwrap(),
        ));

        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn buffer_arc_refcount() {
        let stream = make_stream(Device::Musa(0));
        let b = Arc::new(Buffer::alloc(64, Device::Musa(0), &stream).unwrap());
        assert_eq!(Arc::strong_count(&b), 1);

        let r1 = BufferRef::new(b.clone());
        assert_eq!(Arc::strong_count(&b), 2);

        let r2 = r1.clone();
        assert_eq!(Arc::strong_count(&b), 3);

        drop(r1);
        drop(r2);
        assert_eq!(Arc::strong_count(&b), 1);
    }

    // --- 内存统计相关（为下轮 BufferPool 预留）---

    #[test]
    fn buffer_alloc_large() {
        // 分配较大内存验证 musaMallocAsync 能力
        let stream = make_stream(Device::Musa(0));
        let size = 4 * 1024 * 1024; // 4MB
        let buf = Buffer::alloc(size, Device::Musa(0), &stream).expect("alloc 4MB failed");
        assert_eq!(buf.size(), size);
        assert!(buf.ptr().is_some());
    }

    #[test]
    fn buffer_multiple_alloc_free() {
        // 多次分配释放，验证不泄漏（不崩溃）
        let stream = make_stream(Device::Musa(0));
        for _ in 0..10 {
            let _buf = Buffer::alloc(1024, Device::Musa(0), &stream).unwrap();
        }
        stream.synchronize().unwrap();
    }
}
