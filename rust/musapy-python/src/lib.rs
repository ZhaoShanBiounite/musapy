//! musapy-python: PyO3 绑定层（ADR L2-6, L2-7）
//!
//! Phase 1: 最小 PyO3 模块，暴露 __version__ 和 startup_report()
//! Phase 5: PyDevice/PyDtype/PyStream/PyArray + ms.array() + context managers

pub mod array;
pub mod device;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod stream;

use musapy_core::{Device, debug, mem_stats, resolution};
use pyo3::prelude::*;

// ============================================================
// 辅助：从 Python 参数解析 Device
// ============================================================

fn parse_device_arg(obj: &Bound<'_, PyAny>) -> PyResult<Device> {
    let py = obj.py();
    if let Ok(s) = obj.extract::<String>() {
        Device::parse(&s).map_err(error::to_pyerr)
    } else if let Ok(d) = obj.extract::<Py<device::PyDevice>>() {
        let d_ref = d.borrow(py);
        Ok(d_ref.inner.clone())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "device must be a string (e.g. \"musa:0\") or Device",
        ))
    }
}

// ============================================================
// Context managers（ADR L2-7：device/dtype/stream 对称可组合）
// ============================================================

/// `with ms.device("musa:0"):` 的 context manager。
#[pyclass(name = "_DeviceContext", module = "musapy")]
pub struct DeviceContext {
    guard: Option<resolution::DeviceGuard>,
}

#[pymethods]
impl DeviceContext {
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }
    fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        self.guard.take(); // Drop -> pop stack
    }
}

/// `with ms.dtype(ms.float32):` 的 context manager。
#[pyclass(name = "_DtypeContext", module = "musapy")]
pub struct DtypeContext {
    guard: Option<resolution::DtypeGuard>,
}

#[pymethods]
impl DtypeContext {
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }
    fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        self.guard.take();
    }
}

/// `with ms.stream(s):` 的 context manager。
#[pyclass(name = "_StreamContext", module = "musapy")]
pub struct StreamContext {
    guard: Option<resolution::StreamGuard>,
}

#[pymethods]
impl StreamContext {
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }
    fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        self.guard.take();
    }
}

/// `with ms.debug():` 的 context manager（ADR L3-26）。
#[pyclass(name = "_DebugContext", module = "musapy")]
pub struct DebugContext {
    guard: Option<debug::DebugGuard>,
}

#[pymethods]
impl DebugContext {
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }
    fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        self.guard.take(); // Drop -> 恢复之前的 debug 标志
    }
}

// ============================================================
// 模块级函数
// ============================================================

/// 设置全局默认 device（ADR L0-6 级 4）。
///
/// `ms.set_default_device("musa:0")` 或 `ms.set_default_device(ms.Device("cpu"))`
#[pyfunction]
fn set_default_device(device: &Bound<'_, PyAny>) -> PyResult<()> {
    let dev = parse_device_arg(device)?;
    resolution::set_default_device(dev);
    Ok(())
}

/// 设置全局默认 dtype（ADR L0-7 级 4）。
#[pyfunction]
fn set_default_dtype(dtype: dtype::PyDtype) {
    resolution::set_default_dtype(dtype.0);
}

/// device context manager（ADR L2-7）。
///
/// `with ms.device("musa:0"):` — 临时切换 device context
#[pyfunction(name = "device")]
fn device_context(device: &Bound<'_, PyAny>) -> PyResult<DeviceContext> {
    let dev = parse_device_arg(device)?;
    let guard = resolution::push_device_context(dev);
    Ok(DeviceContext { guard: Some(guard) })
}

/// dtype context manager（ADR L2-7）。
#[pyfunction(name = "dtype")]
fn dtype_context(dtype: dtype::PyDtype) -> DtypeContext {
    let guard = resolution::push_dtype_context(dtype.0);
    DtypeContext { guard: Some(guard) }
}

/// stream context manager（ADR L2-7）。
#[pyfunction(name = "stream")]
fn stream_context(py: Python<'_>, stream: Py<stream::PyStream>) -> StreamContext {
    let s = stream.borrow(py);
    let guard = resolution::push_stream_context(s.inner());
    StreamContext { guard: Some(guard) }
}

// ============================================================
// 内存 / 设备查询（P5.9, ADR L3-28, L1-3）
// ============================================================

/// 格式化字节数为人类可读字符串。
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

/// 内存统计摘要（ADR L3-28）。
///
/// `ms.memory_summary()` — 显示 allocated / cached / peak。
/// `ms.memory_summary(device="musa:0")` — 额外显示该设备的 VRAM 信息。
#[pyfunction]
#[pyo3(signature = (device=None))]
fn memory_summary(device: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let snap = mem_stats::snapshot();
    let mut out = String::new();
    out.push_str("musapy memory summary\n");
    out.push_str(&format!(
        "  Allocated: {} ({} buffers)\n",
        format_bytes(snap.allocated_bytes),
        snap.allocated_buffers
    ));
    out.push_str(&format!(
        "  Cached (deferred-free): {} ({} buffers)\n",
        format_bytes(snap.cached_bytes),
        snap.cached_buffers
    ));
    out.push_str(&format!(
        "  Peak allocated: {}\n",
        format_bytes(snap.peak_bytes)
    ));

    // 可选：显示指定设备的 VRAM 信息
    if let Some(dev_obj) = device {
        let dev = parse_device_arg(dev_obj)?;
        if let Device::Musa(id) = dev {
            match musapy_core::musa_ffi::get_device_properties(id as i32) {
                Ok(props) => {
                    let used = props.total_memory.saturating_sub(props.free_memory);
                    out.push_str(&format!(
                        "  Device musa:{} — {:.1} MB used / {:.0} MB total VRAM ({:.1} MB free)\n",
                        id,
                        used as f64 / (1024.0 * 1024.0),
                        props.total_memory as f64 / (1024.0 * 1024.0),
                        props.free_memory as f64 / (1024.0 * 1024.0),
                    ));
                }
                Err(e) => {
                    out.push_str(&format!("  Device musa:{} — query failed: {}\n", id, e));
                }
            }
        }
    }

    Ok(out)
}

/// 设备能力摘要（ADR L1-3）。
///
/// 遍历所有 MUSA 设备，显示名称、arch、VRAM、CU 数。
#[pyfunction]
fn device_summary() -> PyResult<String> {
    let mut out = String::new();

    // CPU 行
    out.push_str("cpu — host memory\n");

    // MUSA 设备
    match musapy_core::musa_ffi::get_device_count() {
        Ok(count) => {
            for id in 0..count {
                match musapy_core::musa_ffi::get_device_properties(id) {
                    Ok(props) => {
                        out.push_str(&format!(
                            "musa:{} — {}, arch=mp_{}, {} VRAM, {} CUs\n",
                            id,
                            props.name,
                            format!("{}{}", props.arch_major, props.arch_minor),
                            format_bytes(props.total_memory),
                            props.multiprocessor_count,
                        ));
                    }
                    Err(e) => {
                        out.push_str(&format!("musa:{} — query failed: {}\n", id, e));
                    }
                }
            }
        }
        Err(e) => {
            out.push_str(&format!("musa — device count query failed: {}\n", e));
        }
    }

    Ok(out)
}

// ============================================================
// 调试模式（P5.10, ADR L3-26）
// ============================================================

/// 设置全局调试模式（ADR L3-26）。
///
/// `ms.set_debug(True)` 启用 debug：OpContext 记录 python_frame。
#[pyfunction]
fn set_debug(enabled: bool) {
    debug::set_debug(enabled);
}

/// debug context manager（ADR L3-26）。
///
/// `with ms.debug():` — 临时启用 debug 模式，退出后恢复。
#[pyfunction(name = "debug")]
fn debug_context() -> DebugContext {
    let guard = debug::push_debug_context();
    DebugContext { guard: Some(guard) }
}

/// musapy Python 扩展模块入口。
///
/// pyproject.toml 里 module-name = "musapy._core"，
/// 所以本函数名必须是 `_core`，对应 C 符号 `PyInit__core`。
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- 版本信息 + 启动报告 ---
    m.add("__version__", musapy_core::VERSION)?;
    m.add_function(wrap_pyfunction!(startup_report, m)?)?;

    // --- 异常类（P5.11）---
    error::register_exceptions(m)?;

    // --- 类注册 ---
    m.add_class::<device::PyDevice>()?;
    m.add_class::<dtype::PyDtype>()?;
    m.add_class::<stream::PyStream>()?;
    m.add_class::<array::PyArray>()?;

    // context managers
    m.add_class::<DeviceContext>()?;
    m.add_class::<DtypeContext>()?;
    m.add_class::<StreamContext>()?;
    m.add_class::<DebugContext>()?;

    // --- dtype 常量（P5.3）---
    dtype::register_constants(m)?;

    // --- 模块级函数 ---
    m.add_function(wrap_pyfunction!(ops::array, m)?)?;
    m.add_function(wrap_pyfunction!(ops::add, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sub, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mul, m)?)?;
    m.add_function(wrap_pyfunction!(ops::div, m)?)?;
    m.add_function(wrap_pyfunction!(ops::pow, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sin, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cos, m)?)?;
    m.add_function(wrap_pyfunction!(ops::exp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::log, m)?)?;
    m.add_function(wrap_pyfunction!(ops::abs, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sign, m)?)?;
    m.add_function(wrap_pyfunction!(ops::neg, m)?)?;
    m.add_function(wrap_pyfunction!(ops::clamp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::eq, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ne, m)?)?;
    m.add_function(wrap_pyfunction!(ops::lt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::gt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::le, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ge, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sum, m)?)?;
    m.add_function(wrap_pyfunction!(ops::prod, m)?)?;
    m.add_function(wrap_pyfunction!(ops::max, m)?)?;
    m.add_function(wrap_pyfunction!(ops::min, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mean, m)?)?;
    m.add_function(wrap_pyfunction!(ops::argmax, m)?)?;
    m.add_function(wrap_pyfunction!(ops::argmin, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cumsum, m)?)?;
    // Phase 5: creation ops
    m.add_function(wrap_pyfunction!(ops::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ones, m)?)?;
    m.add_function(wrap_pyfunction!(ops::full, m)?)?;
    m.add_function(wrap_pyfunction!(ops::eye, m)?)?;
    m.add_function(wrap_pyfunction!(ops::arange, m)?)?;
    m.add_function(wrap_pyfunction!(ops::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(ops::zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(set_default_device, m)?)?;
    m.add_function(wrap_pyfunction!(set_default_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(device_context, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_context, m)?)?;
    m.add_function(wrap_pyfunction!(stream_context, m)?)?;
    m.add_function(wrap_pyfunction!(memory_summary, m)?)?;
    m.add_function(wrap_pyfunction!(device_summary, m)?)?;
    m.add_function(wrap_pyfunction!(set_debug, m)?)?;
    m.add_function(wrap_pyfunction!(debug_context, m)?)?;

    Ok(())
}

/// 返回 musapy 启动期 ABI 校验报告字符串。
#[pyfunction]
fn startup_report() -> PyResult<String> {
    match musapy_core::abi::run_startup_checks() {
        Ok(r) => Ok(r.to_string()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}
