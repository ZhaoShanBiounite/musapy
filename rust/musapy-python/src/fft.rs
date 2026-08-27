//! ms.fft 命名空间的 _core 层（ADR-003 003-D7，v0.3 Phase 5）
//!
//! 3 个 pyfunction（fft/ifft/rfft）平铺注册到 _core，由 python/musapy/fft.py
//! 包装为 `ms.fft.*` 命名空间（对齐 np.fft 肌肉记忆）。axis=-1 起步；
//! axis != -1 由 ops 层抛错（multi-axis/fftn 推迟到 v0.3 后期）。

use crate::array::PyArray;
use crate::error;
use crate::ops::extract_caller_frame;
use musapy_core::debug;
use pyo3::prelude::*;

/// 捕获 Python 调用帧（debug 模式，ADR L3-26；与 linalg/random ops 同惯例）。
fn capture_frame(py: Python<'_>) {
    if debug::is_debug()
        && let Some(frame) = extract_caller_frame(py)
    {
        debug::set_debug_frame(Some(frame));
    }
}

/// `ms.fft.fft(a, n=None, axis=-1, norm="backward", out=None)` — 复数 FFT。
#[pyfunction]
#[pyo3(signature = (a, n=None, axis=-1, norm=None, out=None))]
pub fn fft(
    py: Python<'_>,
    a: &PyArray,
    n: Option<usize>,
    axis: i32,
    norm: Option<String>,
    out: Option<&PyArray>,
) -> PyResult<PyArray> {
    capture_frame(py);
    let norm = musapy_ops::FftNorm::parse(norm.as_deref()).map_err(error::to_pyerr)?;
    let result =
        musapy_ops::fft(&a.inner, n, axis, norm, out.map(|o| &o.inner)).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.fft.ifft(a, n=None, axis=-1, norm="backward", out=None)` — 逆变换。
#[pyfunction]
#[pyo3(signature = (a, n=None, axis=-1, norm=None, out=None))]
pub fn ifft(
    py: Python<'_>,
    a: &PyArray,
    n: Option<usize>,
    axis: i32,
    norm: Option<String>,
    out: Option<&PyArray>,
) -> PyResult<PyArray> {
    capture_frame(py);
    let norm = musapy_ops::FftNorm::parse(norm.as_deref()).map_err(error::to_pyerr)?;
    let result = musapy_ops::ifft(&a.inner, n, axis, norm, out.map(|o| &o.inner))
        .map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.fft.rfft(a, n=None, axis=-1, norm="backward", out=None)` — 实输入 FFT，
/// 输出形状 (..., N//2+1)。
#[pyfunction]
#[pyo3(signature = (a, n=None, axis=-1, norm=None, out=None))]
pub fn rfft(
    py: Python<'_>,
    a: &PyArray,
    n: Option<usize>,
    axis: i32,
    norm: Option<String>,
    out: Option<&PyArray>,
) -> PyResult<PyArray> {
    capture_frame(py);
    let norm = musapy_ops::FftNorm::parse(norm.as_deref()).map_err(error::to_pyerr)?;
    let result = musapy_ops::rfft(&a.inner, n, axis, norm, out.map(|o| &o.inner))
        .map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}
