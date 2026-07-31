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
fn extract_caller_frame(py: Python<'_>) -> Option<PythonFrame> {
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
