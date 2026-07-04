//! 流与算子上下文（ADR L1-7, L1-8, L1-9, L3-2, L3-3）
//!
//! 职责：
//!   1. OpContext：算子执行上下文（错误归因用，ADR L3-2）
//!   2. PythonFrame：Python 调用栈帧（debug 模式，ADR L3-26）
//!   3. Stream：MUSA 流（Phase 2 占位，Phase 3 接 musaStream_t）
//!
//! Phase 2 约束（ADR L2-3）：
//!   - 不调用任何 MUSA runtime API（musaStreamCreate/Destroy 等）
//!   - raw 字段用 usize 占位，Phase 3 替换为 musaStream_t
//!   - pending_ops 队列可用（纯 Rust 逻辑，不依赖 MUSA）

use crate::device::Device;
use crate::dtype::Dtype;
use crate::layout::Shape;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

// ============================================================
// 1. PythonFrame（debug 模式，ADR L3-26）
// ============================================================

/// Python 调用栈帧（debug 模式下捕获）。
///
/// Phase 2 只定义结构；Phase 4 的 P4.10 通过 PyO3 的 `PyFrame` 真实捕获。
/// debug 模式关闭时 OpContext 的 python_frame 为 None。
#[derive(Clone, Debug)]
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
// 2. OpContext（ADR L3-2）
// ============================================================

/// 算子执行上下文（ADR L3-2）。
///
/// 每个 op 排队到 stream 时记录一个 OpContext，存入 stream 的 pending_ops 队列。
/// `stream.synchronize()` 失败时，从队列取出最近的 OpContext 定位根因。
///
/// 字段说明：
/// - `op_name`：算子名（如 "add"）， &'static str 零分配
/// - `input_shapes/devices/dtypes`：输入张量的元数据
/// - `output_shape`：输出张量形状
/// - `stream_id`：所属流的 ID（跨流调试时引用）
/// - `python_frame`：debug 模式下的 Python 调用位置
/// - `timestamp`：入队时间（性能分析和时序调试）
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
    /// 创建 OpContext（python_frame=None，timestamp=now）。
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

    /// 附带 Python 调用栈帧（debug 模式）。
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
// 3. Stream（ADR L1-7, L1-9, L3-3）
// ============================================================

/// 全局 Stream ID 生成器。
static STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// MUSA 流（ADR L1-7）。
///
/// 每个设备有一个默认流（由 runtime 拥有），用户也可创建自定义流。
/// null stream 不暴露（ADR L1-7 rationale：避免隐式同步的 CUDA 历史包袱）。
///
/// **Phase 2 占位**：`raw` 字段用 `usize`，不调用 musaStreamCreate。
/// **Phase 3**：替换为真实 `musaStream_t` + create/destroy/synchronize。
#[derive(Debug)]
pub struct Stream {
    /// 占位：Phase 3 替换为 `musaStream_t`。
    /// 0 表示未初始化（Phase 2 所有流都是 0）。
    #[allow(dead_code)]
    raw: usize,
    device: Device,
    priority: i32,
    /// 全局唯一 ID（用于 OpContext 跨流引用）。
    id: u64,
    /// 待完成的 op 队列（ADR L3-2，synchronize 失败时用于错误归因）。
    pending_ops: Mutex<VecDeque<OpContext>>,
    /// 流是否被毒化（ADR L3-3，kernel 执行失败后标记）。
    poisoned: AtomicBool,
}

impl Stream {
    /// 创建流（Phase 2 占位，不调用 MUSA API）。
    ///
    /// Phase 3 会替换为真实实现：
    ///   - 调用 musaStreamCreateWithPriority
    ///   - raw 字段存真实 musaStream_t
    ///   - capability probe（L3-11）
    pub fn new(device: Device, priority: i32) -> Self {
        Self {
            raw: 0,
            device,
            priority,
            id: STREAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            pending_ops: Mutex::new(VecDeque::new()),
            poisoned: AtomicBool::new(false),
        }
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
    ///
    /// kernel 执行失败后标记为 poisoned，后续 op 立即返回 PoisonedStream。
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// 标记流为 poisoned（ADR L3-3）。
    ///
    /// Phase 3 的 `stream.reset()`（实验性）会调用此方法。
    pub fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    /// 记录一个 op 到 pending 队列（ADR L3-2）。
    ///
    /// Phase 6 的 OpBuilder 在 kernel launch 后调用此方法。
    /// poisoned 流不应调用（Phase 6 的 OpBuilder 会检查并提前返回）。
    pub fn record_op(&self, ctx: OpContext) {
        self.pending_ops.lock().push_back(ctx);
    }

    /// 取出并移除最早的 op（用于错误归因，ADR L3-2）。
    ///
    /// `synchronize()` 成功后调用，清理已完成 op。
    /// `synchronize()` 失败时用 `peek_oldest_op()` 查看根因（不移除）。
    pub fn pop_oldest_op(&self) -> Option<OpContext> {
        self.pending_ops.lock().pop_front()
    }

    /// 查看最早的 op（不移除，用于错误归因）。
    pub fn peek_oldest_op(&self) -> Option<OpContext>
    where
        OpContext: Clone,
    {
        self.pending_ops.lock().front().cloned()
    }

    /// 当前 pending op 数量。
    pub fn pending_count(&self) -> usize {
        self.pending_ops.lock().len()
    }

    /// 清空 pending 队列（`synchronize()` 成功后批量清理）。
    pub fn clear_pending_ops(&self) {
        self.pending_ops.lock().clear();
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
    use crate::device::Device;
    use crate::dtype::Dtype;

    // --- PythonFrame ---

    #[test]
    fn python_frame_display() {
        let f = PythonFrame {
            filename: "test.py".to_string(),
            lineno: 42,
            function: "main".to_string(),
        };
        assert_eq!(f.to_string(), "test.py:42 (main)");
    }

    // --- OpContext ---

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

    // --- Stream 基本属性 ---

    #[test]
    fn stream_new() {
        let s = Stream::new(Device::Musa(0), 0);
        assert_eq!(*s.device(), Device::Musa(0));
        assert_eq!(s.priority(), 0);
        assert!(!s.is_poisoned());
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn stream_id_is_unique() {
        let s1 = Stream::new(Device::Cpu, 0);
        let s2 = Stream::new(Device::Cpu, 0);
        assert_ne!(s1.id(), s2.id());
    }

    #[test]
    fn stream_display() {
        let s = Stream::new(Device::Musa(0), -1);
        let display = s.to_string();
        assert!(display.contains("device=musa:0"));
        assert!(display.contains("priority=-1"));
        assert!(!display.contains("POISONED"));
    }

    // --- Stream poison ---

    #[test]
    fn stream_poison() {
        let s = Stream::new(Device::Musa(0), 0);
        assert!(!s.is_poisoned());
        s.poison();
        assert!(s.is_poisoned());
        let display = s.to_string();
        assert!(display.contains("POISONED"));
    }

    // --- Stream pending_ops 队列 ---

    #[test]
    fn stream_record_and_pop_op() {
        let s = Stream::new(Device::Musa(0), 0);
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
        let s = Stream::new(Device::Cpu, 0);
        assert!(s.pop_oldest_op().is_none());
    }

    #[test]
    fn stream_fifo_order() {
        let s = Stream::new(Device::Musa(0), 0);

        s.record_op(OpContext::new("op1", vec![], vec![], vec![], vec![], s.id()));
        s.record_op(OpContext::new("op2", vec![], vec![], vec![], vec![], s.id()));
        s.record_op(OpContext::new("op3", vec![], vec![], vec![], vec![], s.id()));

        assert_eq!(s.pop_oldest_op().unwrap().op_name, "op1");
        assert_eq!(s.pop_oldest_op().unwrap().op_name, "op2");
        assert_eq!(s.pop_oldest_op().unwrap().op_name, "op3");
        assert!(s.pop_oldest_op().is_none());
    }

    #[test]
    fn stream_peek_oldest() {
        let s = Stream::new(Device::Musa(0), 0);
        s.record_op(OpContext::new("op1", vec![], vec![], vec![], vec![], s.id()));
        s.record_op(OpContext::new("op2", vec![], vec![], vec![], vec![], s.id()));

        // peek 不移除
        let peeked = s.peek_oldest_op().unwrap();
        assert_eq!(peeked.op_name, "op1");
        assert_eq!(s.pending_count(), 2);

        // pop 移除
        let popped = s.pop_oldest_op().unwrap();
        assert_eq!(popped.op_name, "op1");
        assert_eq!(s.pending_count(), 1);
    }

    #[test]
    fn stream_clear_pending() {
        let s = Stream::new(Device::Musa(0), 0);
        s.record_op(OpContext::new("op1", vec![], vec![], vec![], vec![], s.id()));
        s.record_op(OpContext::new("op2", vec![], vec![], vec![], vec![], s.id()));
        assert_eq!(s.pending_count(), 2);

        s.clear_pending_ops();
        assert_eq!(s.pending_count(), 0);
    }
}