//! 调试模式运行时标志（ADR L3-26）
//!
//! 单一二进制，运行时标志。`ms.set_debug(True)` 或 `MUSAPY_DEBUG=1` 环境变量
//! 或 `with ms.debug():` 上下文。
//!
//! Debug 模式启用时：
//!   - OpContext 记录 `python_frame`（调用者 Python 帧）
//!   - （v0.1-alpha 仅实现 python_frame；其余 debug 特性后续迭代）
//!
//! 实现：Rust `if is_debug()` 分支，release 构建编译器消除。Debug 关闭时零开销。

use crate::stream::PythonFrame;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

// ============================================================
// 全局 debug 标志
// ============================================================

/// 全局调试标志。
static DEBUG: AtomicBool = AtomicBool::new(false);

/// 懒初始化：首次 `is_debug()` 调用时读取 `MUSAPY_DEBUG` 环境变量。
static DEBUG_INIT: Once = Once::new();

/// 设置全局调试标志（ADR L3-26）。
pub fn set_debug(enabled: bool) {
    DEBUG.store(enabled, Ordering::Relaxed);
}

/// 读取全局调试标志。
///
/// 首次调用时从 `MUSAPY_DEBUG` 环境变量初始化（`1` 或 `true` 启用）。
pub fn is_debug() -> bool {
    DEBUG_INIT.call_once(|| {
        if std::env::var("MUSAPY_DEBUG")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            DEBUG.store(true, Ordering::Relaxed);
        }
    });
    DEBUG.load(Ordering::Relaxed)
}

// ============================================================
// Debug context guard（ADR L3-26，`with ms.debug():`）
// ============================================================

/// Debug context guard。Drop 时恢复进入前的 debug 标志。
///
/// 由 `push_debug_context` 创建，PyO3 层在其上实现
/// `__enter__`/`__exit__` 以支持 `with ms.debug():`。
pub struct DebugGuard {
    previous: bool,
}

impl Drop for DebugGuard {
    fn drop(&mut self) {
        DEBUG.store(self.previous, Ordering::Relaxed);
    }
}

/// 进入 debug context（ADR L3-26）。启用 debug 并返回 guard，Drop 时恢复。
///
/// ```ignore
/// {
///     let _g = push_debug_context();
///     assert!(is_debug());
/// }
/// // guard Drop 后恢复之前的状态
/// ```
pub fn push_debug_context() -> DebugGuard {
    let previous = is_debug();
    DEBUG.store(true, Ordering::Relaxed);
    DebugGuard { previous }
}

// ============================================================
// Python 帧传递（PyO3 层 → op_builder）
// ============================================================

thread_local! {
    /// 当前线程的 Python 调用帧（debug 模式下由 PyO3 层设置）。
    ///
    /// 工作流：
    ///   1. PyO3 pyfunction 检测 `is_debug()` → 用 `sys._getframe()` 捕获调用帧
    ///   2. 调用 `set_debug_frame(Some(frame))` 写入 thread-local
    ///   3. op_builder 创建 OpContext 时调用 `take_debug_frame()` 取走
    ///   4. 帧信息附加到 OpContext.python_frame
    static DEBUG_PYTHON_FRAME: RefCell<Option<PythonFrame>> = RefCell::new(None);
}

/// 设置当前线程的 Python 调用帧（PyO3 层调用）。
pub fn set_debug_frame(frame: Option<PythonFrame>) {
    DEBUG_PYTHON_FRAME.with(|f| *f.borrow_mut() = frame);
}

/// 取走当前线程的 Python 调用帧（op_builder 调用，一次性）。
pub fn take_debug_frame() -> Option<PythonFrame> {
    DEBUG_PYTHON_FRAME.with(|f| f.borrow_mut().take())
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_get_debug() {
        let original = is_debug();
        set_debug(true);
        assert!(is_debug());
        set_debug(false);
        assert!(!is_debug());
        // 恢复
        set_debug(original);
    }

    #[test]
    fn debug_guard_restores() {
        let original = is_debug();
        set_debug(false);
        {
            let _g = push_debug_context();
            assert!(is_debug());
        }
        assert!(!is_debug());
        set_debug(original);
    }

    #[test]
    fn debug_guard_restores_previous_true() {
        let original = is_debug();
        set_debug(true);
        {
            let _g = push_debug_context();
            assert!(is_debug());
        }
        // 恢复到之前的 true
        assert!(is_debug());
        set_debug(original);
    }

    #[test]
    fn debug_frame_set_and_take() {
        let frame = PythonFrame {
            filename: "test.py".to_string(),
            lineno: 42,
            function: "test_fn".to_string(),
        };
        set_debug_frame(Some(frame.clone()));
        let taken = take_debug_frame();
        assert_eq!(taken, Some(frame));
        // 第二次 take 返回 None（一次性）
        assert_eq!(take_debug_frame(), None);
    }
}
