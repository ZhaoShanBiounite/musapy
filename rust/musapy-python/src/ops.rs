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
use musapy_core::{Buffer, BufferRef, Device, DeviceResolution, Dtype, DtypeResolution, Layout, Stream};
use musapy_core::{debug, PythonFrame};
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
    let buffer = Buffer::alloc(nbytes, resolved_device.clone(), &stream).map_err(error::to_pyerr)?;
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

    let result = musapy_ops::add(&a.inner, &b.inner, out.map(|o| &o.inner))
        .map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

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

/// 从 Python list/tuple 按 dtype 提取 raw bytes 和 shape。
///
/// 当前支持 1D list/tuple。多维数组和 numpy 支持后续添加。
fn extract_data(data: &Bound<PyAny>, dtype: Dtype) -> PyResult<(Vec<u8>, Vec<usize>)> {
    macro_rules! extract_typed {
        ($t:ty) => {{
            let v: Vec<$t> = data.extract()?;
            let shape = vec![v.len()];
            let nbytes = v.len() * std::mem::size_of::<$t>();
            let bytes = unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, nbytes).to_vec()
            };
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
        Dtype::Complex64 | Dtype::Complex128 => Err(
            pyo3::exceptions::PyNotImplementedError::new_err(
                "complex dtypes not yet supported for array creation",
            ),
        ),
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
