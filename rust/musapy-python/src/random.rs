//! ms.random 命名空间的 _core 层（ADR-003 003-D7，v0.3 Phase 4）
//!
//! 5 个生成 pyfunction 平铺注册到 _core，由 python/musapy/random.py 包装为
//! `ms.random.*` 命名空间（对齐 np.random 肌肉记忆）。seed 语义与
//! shape=None（0-dim）约定见 random.py 与 musapy-ops/random.rs 模块注释。

use crate::array::PyArray;
use crate::dtype::PyDtype;
use crate::error;
use crate::ops::{extract_caller_frame, parse_device_opt, parse_shape};
use musapy_core::debug;
use pyo3::prelude::*;

/// 捕获 Python 调用帧（debug 模式，ADR L3-26；与 linalg ops 同惯例）。
fn capture_frame(py: Python<'_>) {
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }
}

/// `ms.random.rand(shape, dtype=float32, device=None, seed=None)` — uniform [0,1)。
#[pyfunction]
#[pyo3(signature = (shape, *, dtype=None, device=None, seed=None))]
pub fn rand(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
    seed: Option<u64>,
) -> PyResult<PyArray> {
    capture_frame(py);
    let shape = parse_shape(shape)?;
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::rand(&shape, dtype_arg, device_arg, seed).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.random.randn(shape, dtype=float32, device=None, seed=None)` — N(0,1)。
#[pyfunction]
#[pyo3(signature = (shape, *, dtype=None, device=None, seed=None))]
pub fn randn(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
    seed: Option<u64>,
) -> PyResult<PyArray> {
    capture_frame(py);
    let shape = parse_shape(shape)?;
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::randn(&shape, dtype_arg, device_arg, seed).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.random.uniform(low=0.0, high=1.0, shape=None, ...)` — [low, high)。
///
/// shape=None → 0-dim 标量数组（NumPy 对齐）。
#[pyfunction]
#[pyo3(signature = (low=0.0, high=1.0, shape=None, *, dtype=None, device=None, seed=None))]
pub fn uniform(
    py: Python<'_>,
    low: f64,
    high: f64,
    shape: Option<&Bound<'_, PyAny>>,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
    seed: Option<u64>,
) -> PyResult<PyArray> {
    capture_frame(py);
    let shape = match shape {
        Some(s) => parse_shape(s)?,
        None => vec![],
    };
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result =
        musapy_ops::uniform(&shape, low, high, dtype_arg, device_arg, seed).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.random.normal(loc=0.0, scale=1.0, shape=None, ...)` — N(loc, scale²)。
///
/// shape=None → 0-dim 标量数组（NumPy 对齐）。
#[pyfunction]
#[pyo3(signature = (loc=0.0, scale=1.0, shape=None, *, dtype=None, device=None, seed=None))]
pub fn normal(
    py: Python<'_>,
    loc: f64,
    scale: f64,
    shape: Option<&Bound<'_, PyAny>>,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
    seed: Option<u64>,
) -> PyResult<PyArray> {
    capture_frame(py);
    let shape = match shape {
        Some(s) => parse_shape(s)?,
        None => vec![],
    };
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result =
        musapy_ops::normal(&shape, loc, scale, dtype_arg, device_arg, seed).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.random.bernoulli(p=0.5, shape=None, device=None, seed=None)` — → bool。
///
/// shape=None → 0-dim 标量数组（NumPy 对齐）。
#[pyfunction]
#[pyo3(signature = (p=0.5, shape=None, *, device=None, seed=None))]
pub fn bernoulli(
    py: Python<'_>,
    p: f64,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<PyObject>,
    seed: Option<u64>,
) -> PyResult<PyArray> {
    capture_frame(py);
    let shape = match shape {
        Some(s) => parse_shape(s)?,
        None => vec![],
    };
    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::bernoulli(&shape, p, device_arg, seed).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}
