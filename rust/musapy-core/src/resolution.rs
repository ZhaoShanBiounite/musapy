//! Thread-local 默认与 5 级解析链（ADR L0-6, L0-7, L0-8, L0-9, L0-11, L1-7, L2-7）
//!
//! 职责：
//!   1. thread-local 默认 device/dtype/stream 栈（per-thread 隔离，零锁）
//!   2. 全局 SEED：新线程继承父线程当前的默认（值快照，之后解耦）
//!   3. 5 级解析链：Arg > Context > InputArray > GlobalDefault > (AutoProbe/报错)
//!   4. auto-probe（偏好 MUSA，仅显式调用）
//!   5. device/dtype/stream context manager guard（Drop 时 pop）
//!
//! 设计依据：
//!   - L0-6：device 5 级优先级链
//!   - L0-7：dtype 对称解析，但 dtype 总有 float32 兜底（无 DeviceNotConfigured）
//!   - L0-8：每次解析生成 DeviceResolution/DtypeResolution（可追溯）
//!   - L0-9：从未 set_default_device 时首次 array() 抛 DeviceNotConfigured，
//!           不静默用 auto-probe
//!   - L0-11：默认 device/dtype 用 thread-local 栈；新线程 start 时继承父线程
//!            当前默认（值快照，之后解耦）；不做广播 API
//!   - L1-13："Musa > CPU" 层级仅用于 auto-probe 偏好，不影响 op 行为
//!   - L2-7：device()/dtype()/stream() context 对称且可组合

use crate::array::DtypeResolution;
use crate::device::{Device, DeviceResolution, ResolutionSource};
use crate::dtype::Dtype;
use crate::error::{DeviceError, Result};
use crate::musa_ffi;
use crate::stream::Stream;
use parking_lot::Mutex;
use std::cell::RefCell;
use std::sync::Arc;

// ============================================================
// 1. thread-local 栈 + 全局 SEED（ADR L0-11）
// ============================================================

thread_local! {
    /// 默认 device 栈（L0-11）。栈底 = set_default_device 设的；栈顶 = context push 的。
    static DEVICE_STACK: RefCell<Vec<Device>> = RefCell::new(Vec::new());
    /// 默认 dtype 栈（L0-7，对称于 device）。
    static DTYPE_STACK: RefCell<Vec<Dtype>> = RefCell::new(Vec::new());
    /// 当前 stream 栈（L1-7）。栈顶 = 当前 context 的 stream。
    static STREAM_STACK: RefCell<Vec<Arc<Stream>>> = RefCell::new(Vec::new());
    /// 标记当前线程是否已从 device SEED snapshot 过初始值（L0-11：snapshot 后解耦）。
    static DEVICE_SEED_SNAPSHOT_TAKEN: RefCell<bool> = RefCell::new(false);
    /// 标记当前线程是否已从 dtype SEED snapshot 过初始值。
    static DTYPE_SEED_SNAPSHOT_TAKEN: RefCell<bool> = RefCell::new(false);
}

/// 全局 device SEED：新线程继承父线程当前的默认（值快照，L0-11）。
///
/// `set_default_device` 写此 SEED；新线程首次访问 device 栈为空时从此 snapshot。
/// snapshot 后线程与 SEED 解耦：后续兄弟线程/父线程的变更不影响本线程。
/// 注意：只有首次 set_default_device 或从未 snapshot 的线程会更新 SEED。
static SEED_DEVICE: Mutex<Option<Device>> = Mutex::new(None);

/// 全局 dtype SEED（对称于 SEED_DEVICE）。
static SEED_DTYPE: Mutex<Option<Dtype>> = Mutex::new(None);

// ============================================================
// 2. device 默认栈 API（P4.1, P4.4）
// ============================================================

/// 设置全局默认 device（ADR L0-6 级 4，L0-11）。
///
/// 压入当前线程的 device 栈底（替换已有默认），并更新全局 SEED 供新线程继承。
/// 后续 `resolve_device` 无更高优先级来源时用此值。
/// L0-11：snapshot 后线程与 SEED 解耦，只有未 snapshot 的线程首次 set_default_device 才会更新 SEED。
pub fn set_default_device(device: Device) {
    DEVICE_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        let snapshot_taken = DEVICE_SEED_SNAPSHOT_TAKEN.with(|taken| *taken.borrow());
        if s.is_empty() {
            s.push(device.clone());
            // 首次设置时，若未 snapshot 过，更新 SEED 供新线程继承
            if !snapshot_taken {
                *SEED_DEVICE.lock() = Some(device);
            }
        } else {
            // 替换栈底（全局默认），保留上方 context 层
            s[0] = device.clone();
            // 若未 snapshot 过，更新 SEED
            if !snapshot_taken {
                *SEED_DEVICE.lock() = Some(device);
            }
        }
    });
}

/// 获取当前线程的默认 device（ADR L0-6 级 4）。
///
/// 线程栈非空 → 返回栈底（全局默认）。
/// 线程栈空 → 从全局 SEED snapshot 一份压栈（L0-11 继承），之后解耦。
/// SEED 也空（从未 set_default_device）→ 返回 None（触发 L0-9 DeviceNotConfigured）。
pub fn get_default_device() -> Option<Device> {
    DEVICE_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        if !s.is_empty() {
            return Some(s[0].clone());
        }
        // 栈空：从 SEED snapshot（L0-11 继承）
        let seed = SEED_DEVICE.lock().clone();
        if let Some(d) = seed {
            s.push(d.clone());
            DEVICE_SEED_SNAPSHOT_TAKEN.with(|taken| *taken.borrow_mut() = true);
            return Some(d);
        }
        None
    })
}

/// 获取当前 device context 的栈顶（ADR L0-6 级 2，`with ms.device()` push 的）。
///
/// 注意：这与全局默认（栈底）不同。context 层在栈底之上。
/// 栈只有 1 层（仅 set_default_device，无 context）时返回 None（context 未用）。
pub fn get_context_device() -> Option<Device> {
    DEVICE_STACK.with(|stack| {
        let s = stack.borrow();
        // 栈深 > 1 表示有 context 层；栈顶是最近 push 的 context
        if s.len() > 1 {
            Some(s.last().cloned().unwrap())
        } else {
            None
        }
    })
}

/// 压入 device context 到栈（内部，`with ms.device()` 用）。
fn push_device_stack(device: Device) {
    DEVICE_STACK.with(|stack| stack.borrow_mut().push(device));
}

/// 弹出 device context（guard Drop 时调用）。
fn pop_device_context() {
    DEVICE_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

// ============================================================
// 3. dtype 默认栈 API（P4.2, 对称于 device）
// ============================================================

/// 设置全局默认 dtype（ADR L0-7 级 4）。
/// L0-11：snapshot 后线程与 SEED 解耦，只有未 snapshot 的线程首次 set_default_dtype 才会更新 SEED。
pub fn set_default_dtype(dtype: Dtype) {
    DTYPE_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        let snapshot_taken = DTYPE_SEED_SNAPSHOT_TAKEN.with(|taken| *taken.borrow());
        if s.is_empty() {
            s.push(dtype.clone());
            // 首次设置时，若未 snapshot 过，更新 SEED 供新线程继承
            if !snapshot_taken {
                *SEED_DTYPE.lock() = Some(dtype);
            }
        } else {
            // 替换栈底（全局默认）
            s[0] = dtype.clone();
            // 若未 snapshot 过，更新 SEED
            if !snapshot_taken {
                *SEED_DTYPE.lock() = Some(dtype);
            }
        }
    });
}

/// 获取当前线程的默认 dtype（ADR L0-7 级 4）。
///
/// 与 device 对称，但 dtype 总有 float32 兜底（L0-7：不会 DeviceNotConfigured）。
/// 此函数返回 Option（None 表示未设），兜底逻辑在 resolve_dtype 里做。
pub fn get_default_dtype() -> Option<Dtype> {
    DTYPE_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        if !s.is_empty() {
            return Some(s[0]);
        }
        // 栈空：从 SEED snapshot（L0-11 继承）
        let seed = *SEED_DTYPE.lock();
        if let Some(d) = seed {
            s.push(d);
            DTYPE_SEED_SNAPSHOT_TAKEN.with(|taken| *taken.borrow_mut() = true);
            return Some(d);
        }
        None
    })
}

/// 获取当前 dtype context 的栈顶（ADR L0-7 级 2）。
pub fn get_context_dtype() -> Option<Dtype> {
    DTYPE_STACK.with(|stack| {
        let s = stack.borrow();
        if s.len() > 1 {
            Some(*s.last().unwrap())
        } else {
            None
        }
    })
}

fn push_dtype_stack(dtype: Dtype) {
    DTYPE_STACK.with(|stack| stack.borrow_mut().push(dtype));
}

fn pop_dtype_context() {
    DTYPE_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

// ============================================================
// 4. stream 栈 API（P4.3, ADR L1-7）
// ============================================================

/// 获取当前 stream context 的栈顶（ADR L1-7，`with ms.stream()` push 的）。
///
/// stream 无全局默认栈底（无 set_default_stream）；新线程无 stream 上下文。
/// 返回 None 表示用 op 默认 stream（由 runtime 决定）。
pub fn get_current_stream() -> Option<Arc<Stream>> {
    STREAM_STACK.with(|stack| stack.borrow().last().cloned())
}

fn push_stream_stack(stream: Arc<Stream>) {
    STREAM_STACK.with(|stack| stack.borrow_mut().push(stream));
}

fn pop_stream_context() {
    STREAM_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

// ============================================================
// 5. 5 级解析函数（P4.5, P4.6, ADR L0-6, L0-7, L0-8, L0-9）
// ============================================================

/// 解析 device（ADR L0-6 5 级优先级链，L0-8 可追溯，L0-9）。
///
/// 优先级（数字越小越高）：
/// 1. `arg`：函数调用 `device=` 参数
/// 2. Context：`with ms.device()` push 的（get_context_device）
/// 3. InputArray：输入 Array 的 device（inputs[0]，ufunc 风格）
/// 4. GlobalDefault：set_default_device 设的（get_default_device）
/// 5. —— 若都无，抛 `DeviceNotConfigured`（L0-9，不 fallback auto-probe）
///
/// `arg`：显式 device 参数（None 表示未传）。
/// `inputs`：输入 Array 的 device 列表（空表示无输入，如 ms.array 创建）。
///
/// 返回带 source 的 `DeviceResolution`（L0-8 可追溯）。
pub fn resolve_device(arg: Option<Device>, inputs: &[Device]) -> Result<DeviceResolution> {
    // 级 1：Arg
    if let Some(d) = arg {
        return Ok(DeviceResolution::new(d, ResolutionSource::Arg));
    }
    // 级 2：Context
    if let Some(d) = get_context_device() {
        return Ok(DeviceResolution::new(d, ResolutionSource::Context));
    }
    // 级 3：InputArray（ufunc 风格，跟随第一个输入）
    if let Some(d) = inputs.first() {
        return Ok(DeviceResolution::new(d.clone(), ResolutionSource::InputArray));
    }
    // 级 4：GlobalDefault
    if let Some(d) = get_default_device() {
        return Ok(DeviceResolution::new(d, ResolutionSource::GlobalDefault));
    }
    // 级 5：L0-9 —— 抛 DeviceNotConfigured，不静默 auto-probe
    Err(DeviceError::NotConfigured.into())
}

/// 解析 dtype（ADR L0-7，对称于 device，L0-8）。
///
/// 与 resolve_device 同链，但级 5 永不抛错（L0-7：dtype 总有 float32 兜底）。
/// 级 5 用 `Dtype::default()`（Float32），source 标记 `AutoProbe`（L0-7 级 5 固定）。
pub fn resolve_dtype(arg: Option<Dtype>, inputs: &[Dtype]) -> Result<DtypeResolution> {
    // 级 1：Arg
    if let Some(d) = arg {
        return Ok(DtypeResolution::new(d, ResolutionSource::Arg));
    }
    // 级 2：Context
    if let Some(d) = get_context_dtype() {
        return Ok(DtypeResolution::new(d, ResolutionSource::Context));
    }
    // 级 3：InputArray
    if let Some(d) = inputs.first() {
        return Ok(DtypeResolution::new(*d, ResolutionSource::InputArray));
    }
    // 级 4：GlobalDefault
    if let Some(d) = get_default_dtype() {
        return Ok(DtypeResolution::new(d, ResolutionSource::GlobalDefault));
    }
    // 级 5：float32 兜底（L0-7，dtype 无 DeviceNotConfigured）
    Ok(DtypeResolution::new(
        Dtype::default(),
        ResolutionSource::AutoProbe,
    ))
}

// ============================================================
// 6. auto-probe（P4.7, ADR L0-6 级 5, L1-13）
// ============================================================

/// 自动探测设备（ADR L0-6 级 5, L1-13）。
///
/// 偏好 MUSA：有 MUSA GPU（get_device_count > 0）→ `Musa(0)`，否则 `Cpu`。
/// "Musa > CPU" 层级仅用于此探测偏好（L1-13），不影响 op 行为。
///
/// **仅显式调用**（L0-9）：不会在启动时自动运行。用户需显式
/// `set_default_device(auto_probe())` 才启用。FFI 失败时保守返回 Cpu。
pub fn auto_probe() -> Device {
    match musa_ffi::get_device_count() {
        Ok(count) if count > 0 => Device::Musa(0),
        _ => Device::Cpu,
    }
}

// ============================================================
// 7. context manager guard（P4.8, ADR L2-7）
// ============================================================

/// device context guard（ADR L2-7）。Drop 时自动 pop device 栈。
///
/// 由 `push_device_context` 创建，PyO3 层（Phase 5）在其上实现
/// `__enter__`/`__exit__` 以支持 `with ms.device(...):`。
pub struct DeviceGuard {
    _private: (),
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        pop_device_context();
    }
}

/// 进入 device context（ADR L2-7）。返回 guard，Drop 时自动退出。
///
/// ```ignore
/// {
///     let _g = push_device_context(Device::Musa(0));
///     // 此范围内 resolve_device 的级 2 会用 Musa(0)
/// }
/// // guard Drop 后栈恢复
/// ```
pub fn push_device_context(device: Device) -> DeviceGuard {
    push_device_stack(device);
    DeviceGuard { _private: () }
}

/// dtype context guard（ADR L2-7）。Drop 时自动 pop dtype 栈。
pub struct DtypeGuard {
    _private: (),
}

impl Drop for DtypeGuard {
    fn drop(&mut self) {
        pop_dtype_context();
    }
}

/// 进入 dtype context（ADR L2-7）。返回 guard，Drop 时自动退出。
pub fn push_dtype_context(dtype: Dtype) -> DtypeGuard {
    push_dtype_stack(dtype);
    DtypeGuard { _private: () }
}

/// stream context guard（ADR L2-7）。Drop 时自动 pop stream 栈。
pub struct StreamGuard {
    _private: (),
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        pop_stream_context();
    }
}

/// 进入 stream context（ADR L2-7）。返回 guard，Drop 时自动退出。
pub fn push_stream_context(stream: Arc<Stream>) -> StreamGuard {
    push_stream_stack(stream);
    StreamGuard { _private: () }
}

// ============================================================
// 8. 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// 全局测试锁：resolution 的状态（thread-local 栈 + SEED）是进程级共享的，
    /// 并发测试会互相干扰（如 clear_thread_state 清掉他人正在用的 SEED）。
    /// 用此锁串行化所有 resolution 测试，确保隔离。
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    /// 测试辅助：清空当前线程的 device/dtype 栈与 SEED，确保测试隔离。
    fn clear_thread_state() {
        DEVICE_STACK.with(|s| s.borrow_mut().clear());
        DTYPE_STACK.with(|s| s.borrow_mut().clear());
        STREAM_STACK.with(|s| s.borrow_mut().clear());
        *SEED_DEVICE.lock() = None;
        *SEED_DTYPE.lock() = None;
        DEVICE_SEED_SNAPSHOT_TAKEN.with(|taken| *taken.borrow_mut() = false);
        DTYPE_SEED_SNAPSHOT_TAKEN.with(|taken| *taken.borrow_mut() = false);
    }

    // --- device 栈基本操作 ---

    #[test]
    fn set_and_get_default_device() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_device(Device::Musa(0));
        assert_eq!(get_default_device(), Some(Device::Musa(0)));
    }

    #[test]
    fn get_default_device_none_when_unset() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        assert_eq!(get_default_device(), None);
    }

    #[test]
    fn set_default_device_replaces() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_device(Device::Musa(0));
        set_default_device(Device::Cpu);
        assert_eq!(get_default_device(), Some(Device::Cpu));
    }

    // --- dtype 栈基本操作 ---

    #[test]
    fn set_and_get_default_dtype() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_dtype(Dtype::Float64);
        assert_eq!(get_default_dtype(), Some(Dtype::Float64));
    }

    #[test]
    fn get_default_dtype_none_when_unset() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        assert_eq!(get_default_dtype(), None);
    }

    // --- resolve_device 5 级优先级 ---

    #[test]
    fn resolve_device_level1_arg_wins() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_device(Device::Cpu);
        let r = resolve_device(Some(Device::Musa(0)), &[]).unwrap();
        assert_eq!(r.device, Device::Musa(0));
        assert_eq!(r.source, ResolutionSource::Arg);
    }

    #[test]
    fn resolve_device_level2_context_beats_input_and_default() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_device(Device::Cpu); // 级 4
        {
            let _g = push_device_context(Device::Musa(1)); // 级 2
            let inputs = vec![Device::Cpu]; // 级 3
            let r = resolve_device(None, &inputs).unwrap();
            assert_eq!(r.device, Device::Musa(1));
            assert_eq!(r.source, ResolutionSource::Context);
        }
    }

    #[test]
    fn resolve_device_level3_input_beats_default() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_device(Device::Cpu); // 级 4
        let inputs = vec![Device::Musa(0)]; // 级 3
        let r = resolve_device(None, &inputs).unwrap();
        assert_eq!(r.device, Device::Musa(0));
        assert_eq!(r.source, ResolutionSource::InputArray);
    }

    #[test]
    fn resolve_device_level4_global_default() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_device(Device::Musa(0)); // 级 4
        let r = resolve_device(None, &[]).unwrap();
        assert_eq!(r.device, Device::Musa(0));
        assert_eq!(r.source, ResolutionSource::GlobalDefault);
    }

    #[test]
    fn resolve_device_level5_not_configured_error() {
        // L0-9：从未 set_default_device 时抛 DeviceNotConfigured，不 fallback auto-probe
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        let err = resolve_device(None, &[]).unwrap_err();
        match err {
            crate::error::MusapyError::Device(DeviceError::NotConfigured) => {}
            other => panic!("expected DeviceNotConfigured, got {:?}", other),
        }
    }

    // --- resolve_dtype 5 级 ---

    #[test]
    fn resolve_dtype_level1_arg_wins() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_dtype(Dtype::Float64);
        let r = resolve_dtype(Some(Dtype::Int32), &[]).unwrap();
        assert_eq!(r.dtype, Dtype::Int32);
        assert_eq!(r.source, ResolutionSource::Arg);
    }

    #[test]
    fn resolve_dtype_level2_context() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_dtype(Dtype::Float64); // 级 4
        {
            let _g = push_dtype_context(Dtype::Int16); // 级 2
            let r = resolve_dtype(None, &[]).unwrap();
            assert_eq!(r.dtype, Dtype::Int16);
            assert_eq!(r.source, ResolutionSource::Context);
        }
    }

    #[test]
    fn resolve_dtype_level3_input() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_dtype(Dtype::Float64);
        let inputs = vec![Dtype::Int8];
        let r = resolve_dtype(None, &inputs).unwrap();
        assert_eq!(r.dtype, Dtype::Int8);
        assert_eq!(r.source, ResolutionSource::InputArray);
    }

    #[test]
    fn resolve_dtype_level4_global_default() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_dtype(Dtype::Float64);
        let r = resolve_dtype(None, &[]).unwrap();
        assert_eq!(r.dtype, Dtype::Float64);
        assert_eq!(r.source, ResolutionSource::GlobalDefault);
    }

    #[test]
    fn resolve_dtype_level5_float32_fallback() {
        // L0-7：dtype 总有 float32 兜底，不抛错
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        let r = resolve_dtype(None, &[]).unwrap();
        assert_eq!(r.dtype, Dtype::Float32);
        assert_eq!(r.source, ResolutionSource::AutoProbe);
    }

    // --- context guard Drop 恢复 ---

    #[test]
    fn device_guard_restores_on_drop() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_device(Device::Cpu);
        assert!(get_context_device().is_none());
        {
            let _g = push_device_context(Device::Musa(0));
            assert_eq!(get_context_device(), Some(Device::Musa(0)));
        }
        // guard Drop 后 context 层消失
        assert!(get_context_device().is_none());
        // 全局默认仍在
        assert_eq!(get_default_device(), Some(Device::Cpu));
    }

    #[test]
    fn dtype_guard_restores_on_drop() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_dtype(Dtype::Float64);
        {
            let _g = push_dtype_context(Dtype::Int32);
            assert_eq!(get_context_dtype(), Some(Dtype::Int32));
        }
        assert!(get_context_dtype().is_none());
    }

    #[test]
    fn nested_device_contexts() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        set_default_device(Device::Cpu);
        {
            let _g1 = push_device_context(Device::Musa(0));
            {
                let _g2 = push_device_context(Device::Musa(1));
                assert_eq!(get_context_device(), Some(Device::Musa(1)));
            }
            // 内层 Drop，回到外层
            assert_eq!(get_context_device(), Some(Device::Musa(0)));
        }
        assert!(get_context_device().is_none());
    }

    // --- auto-probe ---

    #[test]
    fn auto_probe_returns_musa_on_real_hardware() {
        // 真实 MUSA 环境（本机 MTT S4000）：应返回 Musa(0)
        // mock 模式 get_device_count 返回 1，也是 Musa(0)
        let d = auto_probe();
        assert_eq!(d, Device::Musa(0));
    }

    // --- 线程隔离 + 继承（ADR L0-11，计划验收测试）---

    #[test]
    fn thread_local_isolation_and_inheritance() {
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        std::thread::scope(|s| {
            set_default_device(Device::Musa(0));
            s.spawn(|| {
                // 从父线程 spawn 时继承 Musa(0)
                assert_eq!(get_default_device(), Some(Device::Musa(0)));
                set_default_device(Device::Cpu);
                assert_eq!(get_default_device(), Some(Device::Cpu));
            });
            s.spawn(|| {
                // 也继承 Musa(0)，不受兄弟线程影响
                assert_eq!(get_default_device(), Some(Device::Musa(0)));
            });
            // 父线程仍是 Musa(0)
            assert_eq!(get_default_device(), Some(Device::Musa(0)));
        });
    }

    #[test]
    fn sibling_threads_decouple_after_snapshot() {
        // L0-11：snapshot 后解耦。兄弟线程改自己的默认不影响父线程。
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        std::thread::scope(|s| {
            set_default_device(Device::Musa(0));
            let h = s.spawn(|| {
                // 继承 Musa(0)
                assert_eq!(get_default_device(), Some(Device::Musa(0)));
                set_default_device(Device::Cpu);
                assert_eq!(get_default_device(), Some(Device::Cpu));
            });
            h.join().unwrap();
            // 子线程改了它的默认，父线程不受影响
            assert_eq!(get_default_device(), Some(Device::Musa(0)));
        });
    }

    #[test]
    fn new_thread_without_seed_gets_none() {
        // 父线程未 set_default_device → SEED 空 → 新线程 get_default_device 返回 None
        let _g = TEST_LOCK.lock().unwrap();
        clear_thread_state();
        std::thread::scope(|s| {
            s.spawn(|| {
                assert_eq!(get_default_device(), None);
            });
        });
    }
}
