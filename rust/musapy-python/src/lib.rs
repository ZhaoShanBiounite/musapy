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

use musapy_core::{resolution, Device};
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

    // --- dtype 常量（P5.3）---
    dtype::register_constants(m)?;

    // --- 模块级函数 ---
    m.add_function(wrap_pyfunction!(ops::array, m)?)?;
    m.add_function(wrap_pyfunction!(ops::add, m)?)?;
    m.add_function(wrap_pyfunction!(set_default_device, m)?)?;
    m.add_function(wrap_pyfunction!(set_default_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(device_context, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_context, m)?)?;
    m.add_function(wrap_pyfunction!(stream_context, m)?)?;

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
