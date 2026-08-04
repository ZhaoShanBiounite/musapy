//! 流、事件与算子上下文（ADR L1-7, L1-8, L1-9, L3-2, L3-3, L3-10）
//!
//! Phase 3：真实 MUSA FFI 实现
//!   - Stream: musaStreamCreateWithPriority / musaStreamDestroy / musaStreamSynchronize
//!   - Event: musaEventCreate / musaEventDestroy / musaEventRecord / musaStreamWaitEvent
//!   - CPU 设备：跳过 FFI（raw = null），synchronize 为 no-op

use crate::device::Device;
use crate::dtype::Dtype;
use crate::error::{Result, ShapeError, StreamError};
use crate::layout::Shape;
use crate::musa_ffi;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

// ============================================================
// 1. PythonFrame（debug 模式，ADR L3-26，不变）
// ============================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PythonFrame {
    pub filename: String,
    pub lineno: u32,
    pub function: String,
}

impl fmt::Display for PythonFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} ({})", self.filename, self.lineno, self.function)
    }
}

// ============================================================
// 2. OpContext（ADR L3-2，不变）
// ============================================================

#[derive(Clone, Debug)]
pub struct OpContext {
    pub op_name: &'static str,
    pub input_shapes: Vec<Shape>,
    pub input_devices: Vec<Device>,
    pub input_dtypes: Vec<Dtype>,
    pub output_shape: Shape,
    pub stream_id: u64,
    pub python_frame: Option<PythonFrame>,
    pub timestamp: Instant,
}

impl OpContext {
    pub fn new(
        op_name: &'static str,
        input_shapes: Vec<Shape>,
        input_devices: Vec<Device>,
        input_dtypes: Vec<Dtype>,
        output_shape: Shape,
        stream_id: u64,
    ) -> Self {
        Self {
            op_name,
            input_shapes,
            input_devices,
            input_dtypes,
            output_shape,
            stream_id,
            python_frame: None,
            timestamp: Instant::now(),
        }
    }

    pub fn with_frame(mut self, frame: PythonFrame) -> Self {
        self.python_frame = Some(frame);
        self
    }
}

impl fmt::Display for OpContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "op '{}'", self.op_name)?;
        write!(f, " on stream {}", self.stream_id)?;
        if !self.input_shapes.is_empty() {
            write!(f, ", inputs={:?}", self.input_shapes)?;
        }
        if let Some(ref frame) = self.python_frame {
            write!(f, " at {}", frame)?;
        }
        Ok(())
    }
}

// ============================================================
// 2.5 索引越界检查槽（P1 方案二：GPU 侧错误标志）
// ============================================================
//
// gather/scatter 的越界检查不再在 launch 前做「同步 D2H + host 逐元素比较」
// （~10ms/op 的性能瓶颈），改为 device 侧报告：kernel 发现越界索引时跳过
// 读/写并置位错误槽，host 在下一次 synchronize() 批量读回并带算子上下文报错。
//
// 错误槽布局（16B，对齐满足 i64 访问）：
//   [0..4)   flag: i32 —— kernel atomicOr 1 置位（初始/复位 = 0）
//   [4..8)   pos:  i32 —— 首个越界条目的展平序号（atomicCAS 哨兵 = -1）
//   [8..16)  val:  i64 —— 越界索引值（仅 CAS 胜出线程写入）
//
// 槽位从 arena（连续设备内存）按轮分配：两轮 synchronize 之间发出的槽位
// 必然连续，drain 时每个 arena 一次 D2H 读回整段，避免逐槽拷贝开销。
// 干净轮次（无越界）槽位天然复位（flag 只会被 kernel 置 1）；报错轮次
// drain 后整段重新初始化。

/// 单个检查槽的字节数：flag i32 + pos i32 + val i64。
const INDEX_CHECK_SLOT_BYTES: usize = 16;

struct IndexCheckArena {
    ptr: *mut u8,
    /// 槽位总数。
    capacity: usize,
    /// 本轮已发出的槽位数（drain 时归零）。
    cursor: usize,
}

// arena 指针只在 Mutex 保护下访问，可跨线程。
unsafe impl Send for IndexCheckArena {}

struct PendingIndexCheck {
    /// arena 在 `arenas` Vec 中的下标。
    arena: usize,
    /// arena 内槽位下标。
    slot: usize,
    /// 算子上下文（如 "gather(axis=1, axis_len=3, n_indices=2)"），用于报错归因。
    context: String,
}

#[derive(Default)]
struct IndexCheckState {
    arenas: Vec<IndexCheckArena>,
    pending: VecDeque<PendingIndexCheck>,
}

// ============================================================
// 3. Event（ADR L3-10，Phase 3 真实 FFI）
// ============================================================

/// MUSA 事件（ADR L3-10，用于跨流同步）。
///
/// Phase 3 真实实现：
/// - `new()` 调用 `musaEventCreate`
/// - `record(stream)` 调用 `musaEventRecord`
/// - `Drop` 调用 `musaEventDestroy`
///
/// Event 是 Send（可跨线程移动），但不是 Sync（不并发访问）。
/// 在 Buffer 中始终通过 Mutex 保护。
pub struct Event {
    raw: musa_ffi::musaEvent_t,
}

// Event 需要 Send 才能存入 Buffer（Buffer 在 Arc 中跨线程共享）
unsafe impl Send for Event {}

impl Event {
    /// 创建事件（调用 musaEventCreate）。
    pub fn new() -> Result<Self> {
        let mut raw: musa_ffi::musaEvent_t = std::ptr::null_mut();
        unsafe {
            musa_ffi::check_musa(musa_ffi::musaEventCreate(&mut raw), "musaEventCreate")?;
        }
        Ok(Self { raw })
    }

    /// 原始句柄（供 Stream::wait_event 使用）。
    pub fn raw(&self) -> musa_ffi::musaEvent_t {
        self.raw
    }

    /// 在指定流上记录事件（调用 musaEventRecord）。
    ///
    /// 记录后，其他流可通过 `stream.wait_event(event)` 等待此事件。
    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaEventRecord(self.raw, stream.raw()),
                "musaEventRecord",
            )
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // null 句柄（CPU 占位）不调用 destroy
        if !self.raw.is_null() {
            unsafe {
                // Drop 中忽略错误，无法返回
                let _ = musa_ffi::musaEventDestroy(self.raw);
            }
        }
    }
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Event(raw={:p})", self.raw)
    }
}

// ============================================================
// 4. Stream（ADR L1-7, L1-9, L3-3，Phase 3 真实 FFI）
// ============================================================

static STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// MUSA 流（ADR L1-7）。
///
/// Phase 3 真实实现：
/// - `new()` 调用 `musaStreamCreateWithPriority`（MUSA 设备）或 null（CPU）
/// - `synchronize()` 调用 `musaStreamSynchronize`
/// - `Drop` 调用 `musaStreamDestroy`
/// - `wait_event()` 调用 `musaStreamWaitEvent`
///
/// **线程安全**：MUSA 流可跨线程使用（需外部同步），故 impl Send + Sync。
pub struct Stream {
    /// MUSA 流句柄。CPU 设备或创建失败时为 null。
    raw: musa_ffi::musaStream_t,
    device: Device,
    priority: i32,
    id: u64,
    pending_ops: Mutex<VecDeque<OpContext>>,
    poisoned: AtomicBool,
    /// P1 方案二：gather/scatter 越界检查槽（arena + pending 队列）。
    index_checks: Mutex<IndexCheckState>,
}

// MUSA 流是线程安全的（可从多线程访问，需外部同步）
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// 创建流。
    ///
    /// - MUSA 设备：先 `musaSetDevice(id)` 绑定当前设备，再
    ///   `musaStreamCreateWithPriority`（stream 绑定到当前设备）。priority 对齐 MUSA 流优先级
    /// - CPU 设备：raw = null，不调用 FFI（CPU 无 MUSA 流概念）
    ///
    /// 创建失败返回 `StreamError::MusaCallFailed`。
    pub fn new(device: Device, priority: i32) -> Result<Self> {
        let raw = match &device {
            Device::Cpu => std::ptr::null_mut(),
            Device::Musa(id) => {
                // stream 创建绑定当前设备，必须先 set（修复 Musa(1) 流落到设备 0 的隐患）
                musa_ffi::set_device(*id as i32)?;
                let mut stream: musa_ffi::musaStream_t = std::ptr::null_mut();
                unsafe {
                    musa_ffi::check_musa(
                        musa_ffi::musaStreamCreateWithPriority(&mut stream, 0, priority),
                        "musaStreamCreateWithPriority",
                    )?;
                }
                stream
            }
        };

        Ok(Self {
            raw,
            device,
            priority,
            id: STREAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            pending_ops: Mutex::new(VecDeque::new()),
            poisoned: AtomicBool::new(false),
            index_checks: Mutex::new(IndexCheckState::default()),
        })
    }

    /// 原始流句柄（供 Event::record 使用）。
    pub fn raw(&self) -> musa_ffi::musaStream_t {
        self.raw
    }

    /// 所属设备。
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 流优先级（ADR L1-9）。
    pub fn priority(&self) -> i32 {
        self.priority
    }

    /// 全局唯一 ID。
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 是否被毒化（ADR L3-3）。
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// 标记流为 poisoned（ADR L3-3）。
    pub fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    /// 同步流（等待所有排队操作完成，ADR L3-1）。
    ///
    /// - CPU 流：no-op（无 GPU 操作）
    /// - MUSA 流：调用 `musaStreamSynchronize`
    ///
    /// 成功后清空 pending_ops（所有 op 已完成）。
    /// 失败时返回错误，并附带最早的 pending op 上下文（根因归因，ADR L3-2）。
    pub fn synchronize(&self) -> Result<()> {
        // CPU 流：no-op
        if self.raw.is_null() {
            self.clear_pending_ops();
            return Ok(());
        }

        // MUSA 流：调用 musaStreamSynchronize
        let result = unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaStreamSynchronize(self.raw),
                "musaStreamSynchronize",
            )
        };

        match result {
            Ok(()) => {
                // 成功：清空 pending ops
                self.clear_pending_ops();
                // 默认路径（ADR L3-11）：synchronize 保证流上所有 op 完成，
                // 此时批量回收 deferred-free 队列中的 buffer（musaFree）。
                // 入队的 buffer 在 drop 前已 wait events，此刻必不再被任何流使用。
                #[cfg(not(feature = "stream-ordered"))]
                {
                    if let Err(e) = crate::deferred_free::reclaim_all() {
                        eprintln!("warn: deferred_free reclaim failed: {}", e);
                    }
                }
                // P1 方案二：批量读回 gather/scatter 越界检查槽；发现越界则报错。
                // 不毒化流：越界条目已被 kernel 跳过，GPU 状态仍然一致。
                self.drain_index_checks()
            }
            Err(e) => {
                // 失败：标记 poisoned，附带 op 上下文
                self.poison();
                // kernel 崩溃时在途检查槽的结果不可信，直接丢弃（流已毒化不再复用）
                {
                    let mut st = self.index_checks.lock();
                    st.pending.clear();
                    for arena in st.arenas.iter_mut() {
                        arena.cursor = 0;
                    }
                }
                let op_ctx = self.oldest_op_context_string();
                let msg = if let Some(ctx) = op_ctx {
                    format!("{} | failed op: {}", e, ctx)
                } else {
                    e.to_string()
                };
                Err(StreamError::Poisoned(msg).into())
            }
        }
    }

    /// 等待指定事件（ADR L1-8, L3-10）。
    ///
    /// 调用 `musaStreamWaitEvent`，使当前流等待 event 完成后再执行后续操作。
    /// 用于跨流同步：`out` 的流等待 `in` 的流写完成。
    ///
    /// CPU 流：no-op。
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        if self.raw.is_null() {
            return Ok(());
        }
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaStreamWaitEvent(self.raw, event.raw(), 0),
                "musaStreamWaitEvent",
            )
        }
    }

    /// 记录 op 到 pending 队列（ADR L3-2）。
    pub fn record_op(&self, ctx: OpContext) {
        self.pending_ops.lock().push_back(ctx);
    }

    /// 取出并移除最早的 op。
    pub fn pop_oldest_op(&self) -> Option<OpContext> {
        self.pending_ops.lock().pop_front()
    }

    /// 查看最早的 op 上下文字符串（不移除，用于错误归因）。
    pub fn oldest_op_context_string(&self) -> Option<String> {
        let ops = self.pending_ops.lock();
        ops.front().map(|op| op.to_string())
    }

    /// 当前 pending op 数量。
    pub fn pending_count(&self) -> usize {
        self.pending_ops.lock().len()
    }

    /// 清空 pending 队列。
    pub fn clear_pending_ops(&self) {
        self.pending_ops.lock().clear();
    }

    // ── P1 方案二：索引越界检查槽 ────────────────────────────

    /// 领取一个 device 侧索引检查槽并注册到本流（P1 方案二）。
    ///
    /// 返回槽的设备指针（16B：`[flag: i32][pos: i32][val: i64]`），初始状态
    /// `flag=0, pos=-1`。gather/scatter kernel 遇越界索引时置位并记录，
    /// 下一次 `synchronize()` 批量读回并报错。`context` 用于错误归因。
    ///
    /// 仅 MUSA 流可用（CPU 流返回错误——CPU 路径在 host 端同步校验）。
    pub fn acquire_index_check(&self, context: String) -> Result<NonNull<u8>> {
        if self.raw.is_null() {
            return Err(StreamError::MusaCallFailed(
                "acquire_index_check: CPU stream has no device check slots".to_string(),
            )
            .into());
        }
        let mut st = self.index_checks.lock();
        // 找一个有空位的 arena，或扩容新 arena（旧 arena 的槽可能仍被本轮
        // 在途 kernel 引用，不能释放/复用，只能追加）。
        let (ai, slot) = match st
            .arenas
            .iter_mut()
            .enumerate()
            .find(|(_, a)| a.cursor < a.capacity)
        {
            Some((ai, a)) => {
                let s = a.cursor;
                a.cursor += 1;
                (ai, s)
            }
            None => {
                let cap = st
                    .arenas
                    .last()
                    .map(|a| a.capacity * 2)
                    .unwrap_or(16)
                    .max(16);
                let arena = Self::alloc_check_arena(&self.device, cap)?;
                st.arenas.push(arena);
                let ai = st.arenas.len() - 1;
                st.arenas[ai].cursor = 1;
                (ai, 0)
            }
        };
        st.pending.push_back(PendingIndexCheck {
            arena: ai,
            slot,
            context,
        });
        let ptr = unsafe { st.arenas[ai].ptr.add(slot * INDEX_CHECK_SLOT_BYTES) };
        Ok(NonNull::new(ptr).expect("check arena ptr non-null"))
    }

    /// 分配并初始化一个检查槽 arena（flag=0, pos=-1, val=0）。
    fn alloc_check_arena(device: &Device, capacity: usize) -> Result<IndexCheckArena> {
        let Device::Musa(id) = device else {
            unreachable!("acquire_index_check guards for MUSA streams");
        };
        musa_ffi::set_device(*id as i32)?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaMalloc(&mut ptr, capacity * INDEX_CHECK_SLOT_BYTES),
                "musaMalloc(index_check arena)",
            )?;
        }
        if let Err(e) = Self::init_check_arena(ptr, capacity) {
            unsafe {
                let _ = musa_ffi::musaFree(ptr);
            }
            return Err(e);
        }
        Ok(IndexCheckArena {
            ptr: ptr as *mut u8,
            capacity,
            cursor: 0,
        })
    }

    /// H2D 写入 arena 初始/复位模式（每槽 flag=0, pos=-1, val=0）。
    fn init_check_arena(ptr: *mut std::ffi::c_void, capacity: usize) -> Result<()> {
        let mut init = Vec::with_capacity(capacity * INDEX_CHECK_SLOT_BYTES);
        for _ in 0..capacity {
            init.extend_from_slice(&0i32.to_le_bytes());
            init.extend_from_slice(&(-1i32).to_le_bytes());
            init.extend_from_slice(&0i64.to_le_bytes());
        }
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaMemcpy(
                    ptr,
                    init.as_ptr() as *const std::ffi::c_void,
                    init.len(),
                    musa_ffi::musaMemcpyKind::HostToDevice,
                ),
                "musaMemcpy(index_check init)",
            )
        }
    }

    /// 批量读回本轮所有检查槽（每个 arena 一次 D2H），报告首个越界错误。
    ///
    /// 干净轮次：槽位天然复位（flag 只会被 kernel 置 1），cursor 归零即可。
    /// 报错轮次：整段重新初始化后再复用。
    fn drain_index_checks(&self) -> Result<()> {
        let mut st = self.index_checks.lock();
        if st.pending.is_empty() {
            // 无检查在途：cursor 也必为 0，直接返回
            return Ok(());
        }

        // Pass 1：逐 arena 读回本轮已发出的槽位段。
        let mut host_bufs: Vec<Vec<u8>> = Vec::with_capacity(st.arenas.len());
        for arena in st.arenas.iter() {
            if arena.cursor == 0 {
                host_bufs.push(Vec::new());
                continue;
            }
            let nbytes = arena.cursor * INDEX_CHECK_SLOT_BYTES;
            let mut host = vec![0u8; nbytes];
            unsafe {
                musa_ffi::check_musa(
                    musa_ffi::musaMemcpy(
                        host.as_mut_ptr() as *mut std::ffi::c_void,
                        arena.ptr as *const std::ffi::c_void,
                        nbytes,
                        musa_ffi::musaMemcpyKind::DeviceToHost,
                    ),
                    "musaMemcpy(index_check drain)",
                )?;
            }
            host_bufs.push(host);
        }

        // Pass 2：按注册顺序（FIFO）扫描，取首个越界条目报错。
        let mut first_error: Option<String> = None;
        for check in st.pending.iter() {
            let base = check.slot * INDEX_CHECK_SLOT_BYTES;
            let buf = &host_bufs[check.arena];
            let flag = i32::from_le_bytes(buf[base..base + 4].try_into().unwrap());
            if flag != 0 && first_error.is_none() {
                let pos = i32::from_le_bytes(buf[base + 4..base + 8].try_into().unwrap());
                let val = i64::from_le_bytes(buf[base + 8..base + 16].try_into().unwrap());
                first_error = Some(format!(
                    "{}: index {} out of bounds at flattened entry {} \
                     (detected on device, reported at synchronization)",
                    check.context, val, pos
                ));
            }
        }

        // Pass 3：复位。报错轮次需重新初始化（槽已被置脏）；干净轮次直接归零 cursor。
        let need_reinit = first_error.is_some();
        for arena in st.arenas.iter_mut() {
            if arena.cursor == 0 {
                continue;
            }
            if need_reinit {
                Self::init_check_arena(arena.ptr as *mut std::ffi::c_void, arena.cursor)?;
            }
            arena.cursor = 0;
        }
        st.pending.clear();

        match first_error {
            Some(msg) => Err(ShapeError::Mismatch(msg).into()),
            None => Ok(()),
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // null 句柄（CPU）不调用 destroy
        if !self.raw.is_null() {
            unsafe {
                let _ = musa_ffi::musaStreamDestroy(self.raw);
            }
            // 释放索引检查 arena（get_mut 免锁：drop 时无并发访问）
            let st = self.index_checks.get_mut();
            for arena in st.arenas.drain(..) {
                unsafe {
                    let _ = musa_ffi::musaFree(arena.ptr as *mut std::ffi::c_void);
                }
            }
        }
    }
}

impl fmt::Debug for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stream")
            .field("raw", &format_args!("{:p}", self.raw))
            .field("device", &self.device)
            .field("priority", &self.priority)
            .field("id", &self.id)
            .field("pending_count", &self.pending_count())
            .field("poisoned", &self.is_poisoned())
            .finish()
    }
}

impl fmt::Display for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stream(id={}, device={}, priority={})",
            self.id, self.device, self.priority
        )?;
        if self.is_poisoned() {
            write!(f, " [POISONED]")?;
        }
        Ok(())
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- PythonFrame（不变）---

    #[test]
    fn python_frame_display() {
        let f = PythonFrame {
            filename: "test.py".to_string(),
            lineno: 42,
            function: "main".to_string(),
        };
        assert_eq!(f.to_string(), "test.py:42 (main)");
    }

    // --- OpContext（不变）---

    #[test]
    fn op_context_new() {
        let ctx = OpContext::new(
            "add",
            vec![vec![3], vec![3]],
            vec![Device::Musa(0), Device::Musa(0)],
            vec![Dtype::Float32, Dtype::Float32],
            vec![3],
            42,
        );
        assert_eq!(ctx.op_name, "add");
        assert_eq!(ctx.stream_id, 42);
        assert!(ctx.python_frame.is_none());
    }

    #[test]
    fn op_context_with_frame() {
        let frame = PythonFrame {
            filename: "script.py".to_string(),
            lineno: 10,
            function: "run".to_string(),
        };
        let ctx = OpContext::new("mul", vec![], vec![], vec![], vec![], 0).with_frame(frame);
        assert!(ctx.python_frame.is_some());
        assert!(ctx.to_string().contains("script.py:10"));
    }

    #[test]
    fn op_context_display() {
        let ctx = OpContext::new(
            "add",
            vec![vec![3], vec![3]],
            vec![Device::Musa(0), Device::Musa(0)],
            vec![Dtype::Float32, Dtype::Float32],
            vec![3],
            1,
        );
        let s = ctx.to_string();
        assert!(s.contains("add"));
        assert!(s.contains("stream 1"));
    }

    // --- Stream 基本属性（new 现在返回 Result）---

    #[test]
    fn stream_new_musa() {
        let s = Stream::new(Device::Musa(0), 0).expect("stream creation failed");
        assert_eq!(*s.device(), Device::Musa(0));
        assert_eq!(s.priority(), 0);
        assert!(!s.is_poisoned());
        assert_eq!(s.pending_count(), 0);
        // 真实 MUSA 环境 raw 非空
        assert!(!s.raw().is_null());
    }

    #[test]
    fn stream_new_cpu() {
        // CPU 流：raw 为 null，不调用 FFI
        let s = Stream::new(Device::Cpu, 0).unwrap();
        assert_eq!(*s.device(), Device::Cpu);
        assert!(s.raw().is_null());
    }

    #[test]
    fn stream_id_is_unique() {
        let s1 = Stream::new(Device::Musa(0), 0).unwrap();
        let s2 = Stream::new(Device::Musa(0), 0).unwrap();
        assert_ne!(s1.id(), s2.id());
    }

    #[test]
    fn stream_display() {
        let s = Stream::new(Device::Musa(0), -1).unwrap();
        let display = s.to_string();
        assert!(display.contains("device=musa:0"));
        assert!(display.contains("priority=-1"));
        assert!(!display.contains("POISONED"));
    }

    // --- Stream poison ---

    #[test]
    fn stream_poison() {
        let s = Stream::new(Device::Musa(0), 0).unwrap();
        assert!(!s.is_poisoned());
        s.poison();
        assert!(s.is_poisoned());
        let display = s.to_string();
        assert!(display.contains("POISONED"));
    }

    // --- Stream synchronize ---

    #[test]
    fn stream_synchronize_musa() {
        // 真实 MUSA 流：synchronize 应成功（无排队操作）
        let s = Stream::new(Device::Musa(0), 0).unwrap();
        assert!(s.synchronize().is_ok());
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn stream_synchronize_cpu() {
        // CPU 流：synchronize 是 no-op
        let s = Stream::new(Device::Cpu, 0).unwrap();
        assert!(s.synchronize().is_ok());
    }

    #[test]
    fn stream_synchronize_clears_pending() {
        let s = Stream::new(Device::Musa(0), 0).unwrap();
        s.record_op(OpContext::new(
            "add",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));
        s.record_op(OpContext::new(
            "mul",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));
        assert_eq!(s.pending_count(), 2);

        s.synchronize().unwrap();
        assert_eq!(s.pending_count(), 0);
    }

    // --- Stream pending_ops 队列 ---

    #[test]
    fn stream_record_and_pop_op() {
        let s = Stream::new(Device::Musa(0), 0).unwrap();
        assert_eq!(s.pending_count(), 0);

        let ctx = OpContext::new("add", vec![], vec![], vec![], vec![], s.id());
        s.record_op(ctx);
        assert_eq!(s.pending_count(), 1);

        let popped = s.pop_oldest_op().unwrap();
        assert_eq!(popped.op_name, "add");
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn stream_pop_empty_returns_none() {
        let s = Stream::new(Device::Musa(0), 0).unwrap();
        assert!(s.pop_oldest_op().is_none());
    }

    #[test]
    fn stream_fifo_order() {
        let s = Stream::new(Device::Musa(0), 0).unwrap();

        s.record_op(OpContext::new(
            "op1",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));
        s.record_op(OpContext::new(
            "op2",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));
        s.record_op(OpContext::new(
            "op3",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));

        assert_eq!(s.pop_oldest_op().unwrap().op_name, "op1");
        assert_eq!(s.pop_oldest_op().unwrap().op_name, "op2");
        assert_eq!(s.pop_oldest_op().unwrap().op_name, "op3");
        assert!(s.pop_oldest_op().is_none());
    }

    #[test]
    fn stream_oldest_op_context_string() {
        let s = Stream::new(Device::Musa(0), 0).unwrap();
        s.record_op(OpContext::new(
            "op1",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));
        s.record_op(OpContext::new(
            "op2",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));

        let ctx_str = s.oldest_op_context_string().unwrap();
        assert!(ctx_str.contains("op1"));
        assert_eq!(s.pending_count(), 2); // 不移除
    }

    #[test]
    fn stream_clear_pending() {
        let s = Stream::new(Device::Musa(0), 0).unwrap();
        s.record_op(OpContext::new(
            "op1",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));
        s.record_op(OpContext::new(
            "op2",
            vec![],
            vec![],
            vec![],
            vec![],
            s.id(),
        ));
        assert_eq!(s.pending_count(), 2);

        s.clear_pending_ops();
        assert_eq!(s.pending_count(), 0);
    }

    // --- Event（Phase 3 真实 FFI）---

    #[test]
    fn event_new() {
        let e = Event::new().expect("event creation failed");
        // 真实 MUSA 环境 raw 非空
        assert!(!e.raw().is_null());
    }

    #[test]
    fn event_record_on_stream() {
        let s = Stream::new(Device::Musa(0), 0).unwrap();
        let e = Event::new().unwrap();
        // 记录事件应成功
        assert!(e.record(&s).is_ok());
    }

    #[test]
    fn stream_wait_event() {
        let s1 = Stream::new(Device::Musa(0), 0).unwrap();
        let s2 = Stream::new(Device::Musa(0), 0).unwrap();
        let e = Event::new().unwrap();

        // s1 记录事件
        e.record(&s1).unwrap();
        // s2 等待 s1 的事件
        assert!(s2.wait_event(&e).is_ok());
    }

    #[test]
    fn event_record_on_cpu_stream() {
        // CPU 流：record 是 no-op（raw 为 null，FFI 跳过）
        // 但 Event 本身是真实 MUSA event
        let s = Stream::new(Device::Cpu, 0).unwrap();
        let e = Event::new().unwrap();
        // record 会调用 musaEventRecord(event, null_stream)
        // 这在真实 MUSA 上可能失败或成功，取决于实现
        // 不做强断言，只要不 panic 即可
        let _ = e.record(&s);
    }
}
