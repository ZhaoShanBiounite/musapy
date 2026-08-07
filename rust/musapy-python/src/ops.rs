//! ms.array() — Python 数组创建工厂（ADR L0-6, L0-7, L0-8, L0-9）。
//!
//! 使用 5 级解析链确定 device 和 dtype，分配 Buffer，H2D 拷贝数据，
//! 构造 Array 并返回 PyArray。

use crate::array::PyArray;
use crate::device::PyDevice;
use crate::dtype::PyDtype;
use crate::error;
use musapy_core::musa_ffi;
use musapy_core::resolution;
use musapy_core::{
    Buffer, BufferRef, Device, DeviceResolution, Dtype, DtypeResolution, Layout, Stream,
};
use musapy_core::{PythonFrame, debug};
use musapy_ops;
use pyo3::prelude::*;
use std::sync::Arc;

/// `ms.array(data, dtype=None, device=None)` — 创建数组。
///
/// 解析链：
///   device: Arg > Context > GlobalDefault > (DeviceNotConfigured error)
///   dtype:  Arg > Context > GlobalDefault > float32 fallback
///
/// 数据输入：支持 Python list/tuple（按 dtype 提取元素）。
#[pyfunction]
#[pyo3(signature = (data, dtype=None, device=None))]
pub fn array(
    py: Python<'_>,
    data: &Bound<PyAny>,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
) -> PyResult<PyArray> {
    // --- 1. 解析 device 参数 ---
    let device_arg: Option<Device> = match &device {
        None => None,
        Some(obj) => {
            let obj = obj.bind(py);
            if let Ok(s) = obj.extract::<String>() {
                Some(Device::parse(&s).map_err(error::to_pyerr)?)
            } else if let Ok(d) = obj.extract::<Py<PyDevice>>() {
                let d_ref = d.borrow(py);
                Some(d_ref.inner.clone())
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "device must be a string (e.g. \"musa:0\") or Device",
                ));
            }
        }
    };

    // --- 2. 解析 dtype 参数 ---
    let dtype_arg: Option<Dtype> = dtype.map(|d| d.0);

    // --- 3. 5 级解析（ADR L0-6, L0-7）---
    let device_res: DeviceResolution =
        resolution::resolve_device(device_arg, &[]).map_err(error::to_pyerr)?;
    let dtype_res: DtypeResolution =
        resolution::resolve_dtype(dtype_arg, &[]).map_err(error::to_pyerr)?;

    let resolved_device = device_res.device.clone();
    let resolved_dtype = dtype_res.dtype;

    // --- 4. 从 Python 数据提取 raw bytes + shape ---
    let (bytes, shape) = extract_data(data, resolved_dtype)?;

    // --- 5. 分配 Buffer ---
    let nbytes = bytes.len();
    let stream = Arc::new(Stream::new(resolved_device.clone(), 0).map_err(error::to_pyerr)?);
    let buffer =
        Buffer::alloc(nbytes, resolved_device.clone(), &stream).map_err(error::to_pyerr)?;
    let buffer_arc = Arc::new(buffer);
    let data_ref = BufferRef::new(buffer_arc);

    // --- 6. H2D 拷贝 ---
    copy_to_buffer(&data_ref, &bytes, &resolved_device)?;

    // --- 7. 构造 Array ---
    let layout = Layout::from_shape(shape);
    let array = musapy_core::Array::new(
        data_ref,
        layout,
        resolved_dtype,
        stream,
        device_res,
        dtype_res,
    );

    Ok(PyArray::from_array(array))
}

/// `ms.add(a, b, out=None)` — 逐元素加法（ADR L1-12, L2-4）。
///
/// 无 `out=` 时分配新 Buffer 返回新 Array。
/// 有 `out=` 时写入 out 的 Buffer，在 out 的 stream 上执行（ADR L1-8）。
#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn add(py: Python<'_>, a: &PyArray, b: &PyArray, out: Option<&PyArray>) -> PyResult<PyArray> {
    // Debug 模式：捕获 Python 调用帧（ADR L3-26）
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }

    let result =
        musapy_ops::add(&a.inner, &b.inner, out.map(|o| &o.inner)).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

// ── Phase 2: Binary elementwise ops ─────────────────────────

macro_rules! py_binary_op {
    ($name:ident, $doc:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, b, out=None))]
        #[doc = $doc]
        pub fn $name(py: Python<'_>, a: &PyArray, b: &PyArray, out: Option<&PyArray>) -> PyResult<PyArray> {
            if debug::is_debug() {
                if let Some(frame) = extract_caller_frame(py) {
                    debug::set_debug_frame(Some(frame));
                }
            }
            let result = musapy_ops::$name(&a.inner, &b.inner, out.map(|o| &o.inner))
                .map_err(error::to_pyerr)?;
            Ok(PyArray::from_array(result))
        }
    };
}

py_binary_op!(sub, "ms.sub(a, b, out=None) — 逐元素减法");
py_binary_op!(mul, "ms.mul(a, b, out=None) — 逐元素乘法");
py_binary_op!(div, "ms.div(a, b, out=None) — 逐元素除法");
py_binary_op!(pow, "ms.pow(a, b, out=None) — 逐元素幂运算");

// ── Phase 2: Unary elementwise ops ──────────────────────────

macro_rules! py_unary_op {
    ($name:ident, $doc:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, out=None))]
        #[doc = $doc]
        pub fn $name(py: Python<'_>, a: &PyArray, out: Option<&PyArray>) -> PyResult<PyArray> {
            if debug::is_debug() {
                if let Some(frame) = extract_caller_frame(py) {
                    debug::set_debug_frame(Some(frame));
                }
            }
            let result = musapy_ops::$name(&a.inner, out.map(|o| &o.inner))
                .map_err(error::to_pyerr)?;
            Ok(PyArray::from_array(result))
        }
    };
}

py_unary_op!(sin, "ms.sin(a, out=None) — 逐元素正弦");
py_unary_op!(cos, "ms.cos(a, out=None) — 逐元素余弦");
py_unary_op!(exp, "ms.exp(a, out=None) — 逐元素指数");
py_unary_op!(log, "ms.log(a, out=None) — 逐元素自然对数");
py_unary_op!(abs, "ms.abs(a, out=None) — 逐元素绝对值");
py_unary_op!(sign, "ms.sign(a, out=None) — 逐元素符号函数");
py_unary_op!(neg, "ms.neg(a, out=None) — 逐元素取反");

// ── Phase 3: Comparison ops ─────────────────────────────────

macro_rules! py_compare_op {
    ($name:ident, $doc:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, b, out=None))]
        #[doc = $doc]
        pub fn $name(py: Python<'_>, a: &PyArray, b: &PyArray, out: Option<&PyArray>) -> PyResult<PyArray> {
            if debug::is_debug() {
                if let Some(frame) = extract_caller_frame(py) {
                    debug::set_debug_frame(Some(frame));
                }
            }
            let result =
                musapy_ops::$name(&a.inner, &b.inner, out.map(|o| &o.inner)).map_err(error::to_pyerr)?;
            Ok(PyArray::from_array(result))
        }
    };
}

py_compare_op!(eq, "ms.eq(a, b, out=None) — 逐元素等于比较");
py_compare_op!(ne, "ms.ne(a, b, out=None) — 逐元素不等比较");
py_compare_op!(lt, "ms.lt(a, b, out=None) — 逐元素小于比较");
py_compare_op!(gt, "ms.gt(a, b, out=None) — 逐元素大于比较");
py_compare_op!(le, "ms.le(a, b, out=None) — 逐元素小于等于比较");
py_compare_op!(ge, "ms.ge(a, b, out=None) — 逐元素大于等于比较");

// ── Phase 2: Clamp + Astype ─────────────────────────────────

/// `ms.clamp(a, lo, hi, out=None)` — 逐元素截断到 [lo, hi]。
#[pyfunction]
#[pyo3(signature = (a, lo, hi, out=None))]
pub fn clamp(
    py: Python<'_>,
    a: &PyArray,
    lo: f64,
    hi: f64,
    out: Option<&PyArray>,
) -> PyResult<PyArray> {
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }
    let result =
        musapy_ops::clamp(&a.inner, lo, hi, out.map(|o| &o.inner)).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

// ── Phase 4: Reduction ──────────────────────────────────────

/// Reduction pyfunction 宏（axis + keepdims + out）。
macro_rules! py_reduce_op {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (a, axis=None, keepdims=false, out=None))]
        pub fn $name(
            py: Python<'_>,
            a: &PyArray,
            axis: Option<isize>,
            keepdims: bool,
            out: Option<&PyArray>,
        ) -> PyResult<PyArray> {
            if debug::is_debug() {
                if let Some(frame) = extract_caller_frame(py) {
                    debug::set_debug_frame(Some(frame));
                }
            }
            let result = musapy_ops::$name(&a.inner, axis, keepdims, out.map(|o| &o.inner))
                .map_err(error::to_pyerr)?;
            Ok(PyArray::from_array(result))
        }
    };
}

/// Argmax/Argmin pyfunction 宏（axis + out，无 keepdims）。
macro_rules! py_argreduce_op {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (a, axis=None, out=None))]
        pub fn $name(
            py: Python<'_>,
            a: &PyArray,
            axis: Option<isize>,
            out: Option<&PyArray>,
        ) -> PyResult<PyArray> {
            if debug::is_debug() {
                if let Some(frame) = extract_caller_frame(py) {
                    debug::set_debug_frame(Some(frame));
                }
            }
            let result = musapy_ops::$name(&a.inner, axis, out.map(|o| &o.inner))
                .map_err(error::to_pyerr)?;
            Ok(PyArray::from_array(result))
        }
    };
}

py_reduce_op!(sum, "ms.sum(a, axis=None, keepdims=False, out=None) — 沿轴求和");
py_reduce_op!(prod, "ms.prod(a, axis=None, keepdims=False, out=None) — 沿轴求积");
py_reduce_op!(max, "ms.max(a, axis=None, keepdims=False, out=None) — 沿轴最大值");
py_reduce_op!(min, "ms.min(a, axis=None, keepdims=False, out=None) — 沿轴最小值");
py_reduce_op!(mean, "ms.mean(a, axis=None, keepdims=False, out=None) — 沿轴均值");
py_argreduce_op!(argmax, "ms.argmax(a, axis=None, out=None) — 沿轴最大值索引");
py_argreduce_op!(argmin, "ms.argmin(a, axis=None, out=None) — 沿轴最小值索引");
py_argreduce_op!(cumsum, "ms.cumsum(a, axis=None, out=None) — 沿轴累积求和");

/// 从 Python 调用栈提取调用者帧信息（debug 模式用，ADR L3-26）。
///
/// 使用 `sys._getframe(0)`：C 扩展内调用时，frame(0) = 调用本扩展的 Python 代码。
pub(crate) fn extract_caller_frame(py: Python<'_>) -> Option<PythonFrame> {
    let sys = py.import("sys").ok()?;
    let frame = sys.call_method1("_getframe", (0,)).ok()?;
    let code = frame.getattr("f_code").ok()?;
    let filename: String = code.getattr("co_filename").ok()?.extract().ok()?;
    let lineno: u32 = frame.getattr("f_lineno").ok()?.extract().ok()?;
    let function: String = code.getattr("co_name").ok()?.extract().ok()?;
    Some(PythonFrame {
        filename,
        lineno,
        function,
    })
}

/// 从 Python 数据按 dtype 提取 raw bytes 和 shape。
///
/// 支持：
/// - 标量（int/float/bool）→ 0-dim，shape=[]
/// - 1D list/tuple → shape=[n]
/// - 嵌套 list/tuple → shape=[d0, d1, ...]（必须矩形）
fn extract_data(data: &Bound<PyAny>, dtype: Dtype) -> PyResult<(Vec<u8>, Vec<usize>)> {
    // 尝试作为序列提取（list/tuple）
    if let Ok(seq) = data.downcast::<pyo3::types::PySequence>() {
        let len = seq.len()?;
        if len == 0 {
            // 空列表 → shape=[0]
            return Ok((vec![], vec![0]));
        }
        // 检查第一个元素：如果也是序列 → 多维
        let first = seq.get_item(0)?;
        if first.downcast::<pyo3::types::PySequence>().is_ok()
            && !first.downcast::<pyo3::types::PyString>().is_ok()
        {
            // 多维：递归提取每个子数组，验证矩形
            return extract_nested(seq, dtype);
        }
        // 1D：直接提取
        return extract_flat(data, dtype);
    }

    // 标量 → 0-dim
    extract_scalar(data, dtype)
}

/// 提取多维嵌套序列。
fn extract_nested(
    seq: &Bound<pyo3::types::PySequence>,
    dtype: Dtype,
) -> PyResult<(Vec<u8>, Vec<usize>)> {
    let len = seq.len()?;
    let mut all_bytes: Vec<u8> = Vec::new();
    let mut sub_shape: Option<Vec<usize>> = None;

    for i in 0..len {
        let item = seq.get_item(i)?;
        let item_seq = item.downcast::<pyo3::types::PySequence>().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "inhomogeneous shape: all sub-arrays must have the same length",
            )
        })?;
        let (bytes, shape) = if item_seq.len()? > 0
            && item_seq
                .get_item(0)?
                .downcast::<pyo3::types::PySequence>()
                .is_ok()
            && !item_seq
                .get_item(0)?
                .downcast::<pyo3::types::PyString>()
                .is_ok()
        {
            extract_nested(item_seq, dtype)?
        } else {
            extract_flat(&item, dtype)?
        };

        // 验证所有子数组 shape 一致
        match &sub_shape {
            None => sub_shape = Some(shape),
            Some(expected) => {
                if *expected != shape {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "inhomogeneous shape: sub-array {} has shape {:?}, expected {:?}",
                        i, shape, expected
                    )));
                }
            }
        }
        all_bytes.extend_from_slice(&bytes);
    }

    let mut result_shape = vec![len];
    if let Some(ss) = sub_shape {
        result_shape.extend_from_slice(&ss);
    }
    Ok((all_bytes, result_shape))
}

/// 提取 1D flat 序列。
fn extract_flat(data: &Bound<PyAny>, dtype: Dtype) -> PyResult<(Vec<u8>, Vec<usize>)> {
    macro_rules! extract_typed {
        ($t:ty) => {{
            let v: Vec<$t> = data.extract()?;
            let shape = vec![v.len()];
            let nbytes = v.len() * std::mem::size_of::<$t>();
            let bytes =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, nbytes).to_vec() };
            Ok::<_, pyo3::PyErr>((bytes, shape))
        }};
    }

    match dtype {
        Dtype::Bool => extract_typed!(bool),
        Dtype::Int8 => extract_typed!(i8),
        Dtype::Int16 => extract_typed!(i16),
        Dtype::Int32 => extract_typed!(i32),
        Dtype::Int64 => extract_typed!(i64),
        Dtype::Uint8 => extract_typed!(u8),
        Dtype::Uint16 => extract_typed!(u16),
        Dtype::Uint32 => extract_typed!(u32),
        Dtype::Uint64 => extract_typed!(u64),
        Dtype::Float32 => extract_typed!(f32),
        Dtype::Float64 => extract_typed!(f64),
        Dtype::Float16 | Dtype::Bfloat16 => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "float16/bfloat16 array creation not yet supported",
        )),
        Dtype::Complex64 | Dtype::Complex128 => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "complex dtypes not yet supported for array creation",
            ))
        }
    }
}

/// 提取标量 → 0-dim array（shape=[]）。
fn extract_scalar(data: &Bound<PyAny>, dtype: Dtype) -> PyResult<(Vec<u8>, Vec<usize>)> {
    macro_rules! extract_one {
        ($t:ty) => {{
            let v: $t = data.extract()?;
            let bytes = unsafe {
                std::slice::from_raw_parts(&v as *const $t as *const u8, std::mem::size_of::<$t>())
                    .to_vec()
            };
            Ok::<_, pyo3::PyErr>((bytes, vec![]))
        }};
    }

    match dtype {
        Dtype::Bool => extract_one!(bool),
        Dtype::Int8 => extract_one!(i8),
        Dtype::Int16 => extract_one!(i16),
        Dtype::Int32 => extract_one!(i32),
        Dtype::Int64 => extract_one!(i64),
        Dtype::Uint8 => extract_one!(u8),
        Dtype::Uint16 => extract_one!(u16),
        Dtype::Uint32 => extract_one!(u32),
        Dtype::Uint64 => extract_one!(u64),
        Dtype::Float32 => extract_one!(f32),
        Dtype::Float64 => extract_one!(f64),
        Dtype::Float16 | Dtype::Bfloat16 => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "float16/bfloat16 array creation not yet supported",
        )),
        Dtype::Complex64 | Dtype::Complex128 => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "complex dtypes not yet supported for array creation",
            ))
        }
    }
}

/// 将 host bytes 拷贝到 Buffer（CPU 直接拷贝，MUSA 用 musaMemcpy H2D）。
fn copy_to_buffer(data_ref: &BufferRef, bytes: &[u8], device: &Device) -> PyResult<()> {
    let buffer = data_ref.buffer();
    if bytes.is_empty() {
        return Ok(());
    }
    match device {
        Device::Cpu => {
            if let Some(ptr) = buffer.ptr() {
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len());
                }
            }
            Ok(())
        }
        Device::Musa(_) => {
            if let Some(ptr) = buffer.ptr() {
                unsafe {
                    musa_ffi::check_musa(
                        musa_ffi::musaMemcpy(
                            ptr.as_ptr() as *mut std::ffi::c_void,
                            bytes.as_ptr() as *const std::ffi::c_void,
                            bytes.len(),
                            musa_ffi::musaMemcpyKind::HostToDevice,
                        ),
                        "musaMemcpy(H2D)",
                    )
                    .map_err(error::to_pyerr)?;
                }
            }
            Ok(())
        }
    }
}

// ============================================================
// Phase 5: Creation ops（zeros/ones/full/eye/arange/linspace/zeros_like/ones_like）
// ============================================================

/// 从 Python 参数解析 Device（复用 lib.rs 的逻辑）。
pub(crate) fn parse_device_opt(py: Python<'_>, device: &Option<PyObject>) -> PyResult<Option<Device>> {
    match device {
        None => Ok(None),
        Some(obj) => {
            let obj = obj.bind(py);
            if let Ok(s) = obj.extract::<String>() {
                Ok(Some(Device::parse(&s).map_err(error::to_pyerr)?))
            } else if let Ok(d) = obj.extract::<Py<PyDevice>>() {
                let d_ref = d.borrow(py);
                Ok(Some(d_ref.inner.clone()))
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "device must be a string (e.g. \"musa:0\") or Device",
                ))
            }
        }
    }
}

/// 从 Python 参数解析 shape（接受 int 或 tuple/list of ints）。
pub(crate) fn parse_shape(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(n) = shape.extract::<usize>() {
        Ok(vec![n])
    } else if let Ok(v) = shape.extract::<Vec<usize>>() {
        Ok(v)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "shape must be an int or a tuple/list of ints",
        ))
    }
}

/// `ms.zeros(shape, dtype=None, device=None)` — 创建全零数组。
#[pyfunction]
#[pyo3(signature = (shape, *, dtype=None, device=None))]
pub fn zeros(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
) -> PyResult<PyArray> {
    let shape = parse_shape(shape)?;
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::zeros(&shape, dtype_arg, device_arg).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.ones(shape, dtype=None, device=None)` — 创建全一数组。
#[pyfunction]
#[pyo3(signature = (shape, *, dtype=None, device=None))]
pub fn ones(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
) -> PyResult<PyArray> {
    let shape = parse_shape(shape)?;
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::ones(&shape, dtype_arg, device_arg).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.full(shape, fill_value, dtype=None, device=None)` — 创建填充指定值的数组。
#[pyfunction]
#[pyo3(signature = (shape, fill_value, *, dtype=None, device=None))]
pub fn full(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    fill_value: f64,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
) -> PyResult<PyArray> {
    let shape = parse_shape(shape)?;
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::full(&shape, fill_value, dtype_arg, device_arg).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.eye(n, m=None, k=0, dtype=None, device=None)` — 创建单位矩阵。
#[pyfunction]
#[pyo3(signature = (n, m=None, k=0, *, dtype=None, device=None))]
pub fn eye(
    py: Python<'_>,
    n: usize,
    m: Option<usize>,
    k: i32,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
) -> PyResult<PyArray> {
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::eye(n, m, k, dtype_arg, device_arg).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.arange(start, stop=None, step=1, dtype=None, device=None)` — 创建等差序列。
///
/// 单参数形式：`ms.arange(5)` → `[0, 1, 2, 3, 4]`（int64）。
/// 双参数形式：`ms.arange(2, 7)` → `[2, 3, 4, 5, 6]`。
/// 三参数形式：`ms.arange(0, 1, 0.2)` → `[0.0, 0.2, 0.4, 0.6, 0.8]`（float64）。
///
/// dtype 推断遵循 NumPy 行为：Python int 参数 → int64，Python float 参数 → float64。
#[pyfunction]
#[pyo3(signature = (start, stop=None, step=None, *, dtype=None, device=None))]
pub fn arange(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
) -> PyResult<PyArray> {
    // 检测 Python 参数类型（int vs float）用于 dtype 推断
    let start_is_float = start.is_instance_of::<pyo3::types::PyFloat>();
    let stop_is_float = stop.map_or(false, |s| s.is_instance_of::<pyo3::types::PyFloat>());
    let step_is_float = step.map_or(false, |s| s.is_instance_of::<pyo3::types::PyFloat>());

    let start_val: f64 = start.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("arange: start must be a number")
    })?;
    let stop_val: Option<f64> = match stop {
        Some(s) => Some(s.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("arange: stop must be a number")
        })?),
        None => None,
    };
    let step_val: f64 = match step {
        Some(s) => s.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("arange: step must be a number")
        })?,
        None => 1.0,
    };

    // dtype 推断（NumPy 行为）：
    // - 显式 dtype= 覆盖
    // - 任何参数是 Python float → float64
    // - 全部是 Python int → int64
    let dtype_arg: Option<musapy_core::Dtype> = match dtype {
        Some(d) => Some(d.0),
        None => {
            let any_float = start_is_float || stop_is_float || step_is_float;
            Some(if any_float {
                musapy_core::Dtype::Float64
            } else {
                musapy_core::Dtype::Int64
            })
        }
    };

    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::arange(start_val, stop_val, step_val, dtype_arg, device_arg).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.linspace(start, stop, num=50, dtype=None, device=None)` — 创建等间隔序列。
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, *, dtype=None, device=None))]
pub fn linspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    dtype: Option<PyDtype>,
    device: Option<PyObject>,
) -> PyResult<PyArray> {
    let dtype_arg = dtype.map(|d| d.0);
    let device_arg = parse_device_opt(py, &device)?;
    let result = musapy_ops::linspace(start, stop, num, dtype_arg, device_arg).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.zeros_like(a)` — 创建与输入同 shape/dtype/device 的全零数组。
#[pyfunction]
#[pyo3(signature = (a))]
pub fn zeros_like(a: &PyArray) -> PyResult<PyArray> {
    let result = musapy_ops::zeros_like(&a.inner).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.ones_like(a)` — 创建与输入同 shape/dtype/device 的全一数组。
#[pyfunction]
#[pyo3(signature = (a))]
pub fn ones_like(a: &PyArray) -> PyResult<PyArray> {
    let result = musapy_ops::ones_like(&a.inner).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

// ============================================================
// Phase 6: Indexing 算子（view 零拷贝）
// ============================================================

/// `ms.transpose(a, axes=None)` — 转置（零拷贝视图）。
///
/// `axes=None` 时完全反转维度顺序。
#[pyfunction]
#[pyo3(signature = (a, axes=None))]
pub fn transpose(a: &PyArray, axes: Option<Vec<usize>>) -> PyResult<PyArray> {
    let result = musapy_ops::transpose(&a.inner, axes.as_deref()).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.permute(a, dims)` — 按指定维度排列（零拷贝视图）。
#[pyfunction]
#[pyo3(signature = (a, dims))]
pub fn permute(a: &PyArray, dims: Vec<usize>) -> PyResult<PyArray> {
    let result = musapy_ops::permute(&a.inner, &dims).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.flip(a, axis)` — 翻转指定轴（零拷贝视图）。
#[pyfunction]
#[pyo3(signature = (a, axis))]
pub fn flip(a: &PyArray, axis: usize) -> PyResult<PyArray> {
    let result = musapy_ops::flip(&a.inner, axis).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.index_select(a, axis, index)` — 整数索引选择（降维，零拷贝视图）。
#[pyfunction]
#[pyo3(signature = (a, axis, index))]
pub fn index_select(a: &PyArray, axis: usize, index: usize) -> PyResult<PyArray> {
    let result = musapy_ops::index_select(&a.inner, axis, index).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.slice(a, specs)` — 多维切片（零拷贝视图）。
///
/// specs 是列表的列表：[[start, stop, step], ...]
#[pyfunction]
#[pyo3(signature = (a, specs))]
pub fn slice(a: &PyArray, specs: Vec<Vec<usize>>) -> PyResult<PyArray> {
    let slice_specs: Vec<musapy_ops::SliceSpec> = specs
        .into_iter()
        .map(|s| {
            if s.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each slice spec must be [start, stop, step]",
                ));
            }
            Ok(musapy_ops::SliceSpec {
                start: s[0],
                stop: s[1],
                step: s[2],
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let result = musapy_ops::slice(&a.inner, &slice_specs).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.contiguous(a)` — 物化为连续布局。
///
/// 已连续时零拷贝返回共享视图；否则分配新 buffer 逐元素拷贝。
#[pyfunction]
#[pyo3(signature = (a))]
pub fn contiguous(a: &PyArray) -> PyResult<PyArray> {
    let result = musapy_ops::contiguous(&a.inner).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.gather(a, indices, axis=0)` — 沿 axis 按 indices 取元素（copy）。
///
/// 等价 `np.take(a, indices, axis=axis)`。indices 为 1D int64。
///
/// GPU 越界语义（P1 去同步）：kernel 内检查索引，越界元素跳过并记录到
/// device 错误槽；异常延迟到下一次流同步（如 `tolist()`/`item()`）抛出
/// `ShapeError`，流本身不失效。CPU 路径仍为同步报错。
#[pyfunction]
#[pyo3(signature = (a, indices, axis=0))]
pub fn gather(a: &PyArray, indices: &PyArray, axis: usize) -> PyResult<PyArray> {
    let result = musapy_ops::gather(&a.inner, &indices.inner, axis).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.scatter(a, indices, values, axis=0)` — 沿 axis 把 values 写入 indices 位置（copy）。
///
/// 返回新数组，不修改原数组。重复 indices 写入顺序未定义。
///
/// GPU 越界语义同 `gather`：越界写入在 kernel 内跳过并记录，异常延迟到
/// 下一次流同步抛出 `ShapeError`，流不失效；CPU 路径同步报错。
#[pyfunction]
#[pyo3(signature = (a, indices, values, axis=0))]
pub fn scatter(a: &PyArray, indices: &PyArray, values: &PyArray, axis: usize) -> PyResult<PyArray> {
    let result =
        musapy_ops::scatter(&a.inner, &indices.inner, &values.inner, axis).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

// ── Phase 2 (v0.3): linalg ops（ADR-003 003-D3/D4/D6）────────

/// `ms.matmul(a, b, out=None)` — 矩阵乘法（NumPy `@` 语义）。
///
/// 支持 1D/2D 组合（含 `(n,)@(n,m)`/`(m,n)@(n,)`/`(n,)@(n,)` 的内积退化）；
/// 3D+ batch 推迟到 v0.4。GPU 走 muBLAS gemm（转置技巧），CPU 走
/// OpenBLAS/朴素实现。
#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn matmul(py: Python<'_>, a: &PyArray, b: &PyArray, out: Option<&PyArray>) -> PyResult<PyArray> {
    // Debug 模式：捕获 Python 调用帧（ADR L3-26）
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }

    let result =
        musapy_ops::matmul(&a.inner, &b.inner, out.map(|o| &o.inner)).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.dot(a, b, out=None)` — 点积（ADR-003 003-D6）。
///
/// `(n,)·(n,)` → 0-dim 标量；2D 组合委托 matmul；0-dim/3D+ 抛 ShapeError。
#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn dot(py: Python<'_>, a: &PyArray, b: &PyArray, out: Option<&PyArray>) -> PyResult<PyArray> {
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }

    let result =
        musapy_ops::dot(&a.inner, &b.inner, out.map(|o| &o.inner)).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.solve(a, b)` — 解线性方程组 `a @ x = b`。
///
/// `a` 必须为方阵；`b` 为 `(n,)` 或 `(n,k)`。奇异矩阵抛 `LinAlgError`
/// （003-D3）。GPU 走 muSOLVER getrf+getrs，CPU 走 OpenBLAS/朴素 LU。
#[pyfunction]
#[pyo3(signature = (a, b))]
pub fn solve(py: Python<'_>, a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }

    let result = musapy_ops::solve(&a.inner, &b.inner).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

// ── Phase 3 (v0.3): 分解类 linalg ops（ADR-003 003-D3/D6，GPU-only）──

/// `ms.lu(a)` → `(lu, piv)` — LU 分解（torch.linalg.lu 语义）。
///
/// `lu` 为 (m×n) 标准行主序布局（L 单位下三角 + U 上三角，LAPACK getrf
/// 布局，重建 `a = P·L·U`）；`piv` 为 `min(m,n)` 个 1-based int64 主元
/// （LAPACK ipiv 语义）。GPU 走 muSOLVER getrf（003-D4 共享 helper）。
#[pyfunction]
#[pyo3(signature = (a))]
pub fn lu(py: Python<'_>, a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }

    let (lu_arr, piv) = musapy_ops::lu(&a.inner).map_err(error::to_pyerr)?;
    Ok((PyArray::from_array(lu_arr), PyArray::from_array(piv)))
}

/// `ms.qr(a, mode="reduced")` → `(q, r)` — QR 分解（NumPy 语义）。
///
/// `mode`：`"reduced"` → q (m,k)、r (k,n)；`"complete"` → q (m,m)、r (m,n)
/// （r 下三角补零）。GPU 走 muSOLVER geqrf+orgqr。
#[pyfunction]
#[pyo3(signature = (a, mode="reduced"))]
pub fn qr(py: Python<'_>, a: &PyArray, mode: &str) -> PyResult<(PyArray, PyArray)> {
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }

    let (q_arr, r_arr) = musapy_ops::qr(&a.inner, mode).map_err(error::to_pyerr)?;
    Ok((PyArray::from_array(q_arr), PyArray::from_array(r_arr)))
}

/// `ms.svd(a, full_matrices=True, compute_uv=True)` → `(u, s, vh)`。
///
/// `s` 为 `min(m,n)` 个降序奇异值（1D）；`full_matrices=True` 时 u (m,m)、
/// vh (n,n)，否则 u (m,k)、vh (k,n)。`compute_uv=False` 仅返回 `s`
/// （NumPy 语义，非三元组）。GPU 走 muSOLVER gesvd（S 合理性校验兜底
/// info 失效，见 linalg.rs）。
#[pyfunction]
#[pyo3(signature = (a, full_matrices=true, compute_uv=true))]
pub fn svd(
    py: Python<'_>,
    a: &PyArray,
    full_matrices: bool,
    compute_uv: bool,
) -> PyResult<PyObject> {
    if debug::is_debug() {
        if let Some(frame) = extract_caller_frame(py) {
            debug::set_debug_frame(Some(frame));
        }
    }

    let (u, s, vh) = musapy_ops::svd(&a.inner, full_matrices, compute_uv).map_err(error::to_pyerr)?;
    let s_py = PyArray::from_array(s);
    if !compute_uv {
        return Ok(s_py.into_py(py));
    }
    let u_py = u.map(PyArray::from_array);
    let vh_py = vh.map(PyArray::from_array);
    Ok((u_py, s_py, vh_py).into_py(py))
}
