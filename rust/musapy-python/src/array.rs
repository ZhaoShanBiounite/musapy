//! PyArray — Python 绑定的 Array 类（ADR L1-11, L3-27, L0-8）。
//!
//! 从 Buffer + Layout + resolution 构造，提供只读属性和 name 管理。
//! `device` getter 返回带 resolution source 的 PyDevice（L0-8 反馈原则）。
//! Phase 6: `__add__` / `tolist()` / `item()`（ADR L1-11 显式 sync + D2H）。

use crate::device::PyDevice;
use crate::dtype::PyDtype;
use crate::error;
use crate::stream::PyStream;
use musapy_core::musa_ffi;
use musapy_core::{Array, Device, Dtype};
use musapy_ops;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::ffi::c_void;

/// Python Array 类。
///
/// 通过 `ms.array(...)` 创建，不直接构造。
#[pyclass(name = "Array", module = "musapy")]
pub struct PyArray {
    pub(crate) inner: Array,
}

impl PyArray {
    /// 从 musapy-core Array 构造（内部，供 ops.rs 用）。
    pub fn from_array(array: Array) -> Self {
        Self { inner: array }
    }
}

#[pymethods]
impl PyArray {
    /// 形状元组，如 `(3,)` 或 `(2, 3)`。
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape())
    }

    /// 维度数。
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// 元素总数（shape 的乘积）。
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// dtype 对象。
    #[getter]
    fn dtype(&self) -> PyDtype {
        PyDtype::from_dtype(self.inner.dtype())
    }

    /// dtype 解析来源（如 "context"），无 resolution 时为 None。
    #[getter]
    fn dtype_resolution_source(&self) -> Option<String> {
        Some(self.inner.dtype_resolution().source.to_string())
    }

    /// 设备对象（带 resolution source）。
    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_resolution(self.inner.device_resolution())
    }

    /// 流对象。
    #[getter]
    fn stream(&self) -> PyStream {
        PyStream::from_stream(std::sync::Arc::clone(self.inner.stream()))
    }

    /// 数组名称（可空）。
    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name().map(|s| s.to_string())
    }

    /// 设置数组名称。
    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.set_name(name);
    }

    /// 清除数组名称。
    fn clear_name(&mut self) {
        self.inner.clear_name();
    }

    /// 字节数（size * element_size）。
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    /// 是否连续。
    #[getter]
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// 是否 0 维（标量）。
    #[getter]
    fn is_0d(&self) -> bool {
        self.inner.is_0d()
    }

    /// `a + b` — 逐元素加法（ADR L1-12）。
    ///
    /// 等价于 `ms.add(self, other)`，分配新 Buffer 返回。
    fn __add__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::add(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a - b` — 逐元素减法。
    fn __sub__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::sub(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a * b` — 逐元素乘法。
    fn __mul__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::mul(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a / b` — 逐元素除法。
    fn __truediv__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::div(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a ** b` — 逐元素幂运算。
    fn __pow__(&self, other: &PyArray, _modulo: Option<i32>) -> PyResult<PyArray> {
        let result = musapy_ops::pow(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `-a` — 逐元素取负（0 - a）。
    fn __neg__(&self) -> PyResult<PyArray> {
        let result = musapy_ops::neg(&self.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `abs(a)` — 逐元素绝对值。
    fn __abs__(&self) -> PyResult<PyArray> {
        let result = musapy_ops::abs(&self.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a == b` — 逐元素等于比较（广播 → bool 数组）。
    fn __eq__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::eq(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a != b` — 逐元素不等比较。
    fn __ne__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::ne(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a < b` — 逐元素小于比较。
    fn __lt__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::lt(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a > b` — 逐元素大于比较。
    fn __gt__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::gt(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a <= b` — 逐元素小于等于比较。
    fn __le__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::le(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a >= b` — 逐元素大于等于比较。
    fn __ge__(&self, other: &PyArray) -> PyResult<PyArray> {
        let result = musapy_ops::ge(&self.inner, &other.inner, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// `a.astype(dtype)` — 类型转换。
    fn astype(&self, dtype: PyDtype) -> PyResult<PyArray> {
        let result =
            musapy_ops::astype(&self.inner, dtype.0, None).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    /// 将数组数据取回 host 并转为 Python list（ADR L1-11: 显式 sync + D2H）。
    ///
    /// 多维数组返回嵌套 list（NumPy 行为）：
    /// `a.tolist()` → `[[1.0, 2.0], [3.0, 4.0]]`（shape=(2,2)）
    /// 0-dim 返回单元素 list：`[3.14]`
    fn tolist(&self, py: Python<'_>) -> PyResult<PyObject> {
        let bytes = self.sync_and_copy_to_host(py)?;
        let n = self.inner.size();
        let dtype = self.inner.dtype();
        let shape = self.inner.shape();

        // 先获取 flat list
        let flat = bytes_to_pylist(py, &bytes, n, dtype)?;

        // 0-dim 或 1D：直接返回 flat list
        if shape.len() <= 1 {
            return Ok(flat);
        }

        // 多维：递归嵌套
        nest_flat_list(py, flat.bind(py), shape, 0)
    }

    /// 0-dim 或 size=1 数组取标量值（ADR L1-11）。
    ///
    /// `a.item()` → `3.14`（Python float）
    fn item(&self, py: Python<'_>) -> PyResult<PyObject> {
        if self.inner.size() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "can only convert size-1 arrays to scalar",
            ));
        }
        let bytes = self.sync_and_copy_to_host(py)?;
        let dtype = self.inner.dtype();
        bytes_to_scalar(py, &bytes, dtype)
    }

    /// `Array(shape=(3,), dtype=float32, device=musa:0)`
    fn __repr__(&self) -> String {
        let shape = self.inner.shape();
        // 1D 时加尾逗号：(3,) 而非 (3)
        let shape_str = match shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", shape[0]),
            _ => {
                let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                format!("({})", parts.join(", "))
            }
        };
        format!(
            "Array(shape={}, dtype={}, device={})",
            shape_str,
            self.inner.dtype(),
            self.inner.device(),
        )
    }

    /// 与 `__repr__` 相同。
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================
// 内部辅助方法
// ============================================================

impl PyArray {
    /// 显式同步 stream + D2H 拷贝（ADR L1-11）。
    ///
    /// 先 `stream.synchronize()`，然后按 layout 拷贝逻辑元素到连续 bytes。
    /// 支持 offset 和非连续布局（stride-aware gather）。
    fn sync_and_copy_to_host(&self, _py: Python<'_>) -> PyResult<Vec<u8>> {
        // 1. stream 同步
        self.inner.stream().synchronize().map_err(error::to_pyerr)?;

        let n_elements = self.inner.size();
        let elem_size = self.inner.dtype().element_size();
        let nbytes = n_elements * elem_size;
        let mut bytes = vec![0u8; nbytes];
        if nbytes == 0 {
            return Ok(bytes);
        }

        let layout = self.inner.layout();

        // 2. 连续且无 offset：快速路径（直接 memcpy）
        if layout.is_contiguous() {
            let ptr = self.inner.data().buffer().ptr();
            match self.inner.device() {
                Device::Cpu => {
                    if let Some(p) = ptr {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                p.as_ptr(),
                                bytes.as_mut_ptr(),
                                nbytes,
                            );
                        }
                    }
                }
                Device::Musa(_) => {
                    if let Some(p) = ptr {
                        unsafe {
                            musa_ffi::check_musa(
                                musa_ffi::musaMemcpy(
                                    bytes.as_mut_ptr() as *mut c_void,
                                    p.as_ptr() as *const c_void,
                                    nbytes,
                                    musa_ffi::musaMemcpyKind::DeviceToHost,
                                ),
                                "musaMemcpy(D2H)",
                            )
                            .map_err(error::to_pyerr)?;
                        }
                    }
                }
            }
            return Ok(bytes);
        }

        // 3. 非连续或有 offset：需要 gather
        //    GPU 时先 D2H 整个 buffer，再在 host 端 gather
        let host_buffer: Vec<u8> = match self.inner.device() {
            Device::Cpu => {
                // CPU：直接使用 buffer 内存
                let buf_size = self.inner.data().buffer().size();
                let ptr = self.inner.data().buffer().ptr();
                let mut buf = vec![0u8; buf_size];
                if let Some(p) = ptr {
                    unsafe {
                        std::ptr::copy_nonoverlapping(p.as_ptr(), buf.as_mut_ptr(), buf_size);
                    }
                }
                buf
            }
            Device::Musa(_) => {
                let buf_size = self.inner.data().buffer().size();
                let ptr = self.inner.data().buffer().ptr();
                let mut buf = vec![0u8; buf_size];
                if let Some(p) = ptr {
                    unsafe {
                        musa_ffi::check_musa(
                            musa_ffi::musaMemcpy(
                                buf.as_mut_ptr() as *mut c_void,
                                p.as_ptr() as *const c_void,
                                buf_size,
                                musa_ffi::musaMemcpyKind::DeviceToHost,
                            ),
                            "musaMemcpy(D2H full buffer)",
                        )
                        .map_err(error::to_pyerr)?;
                    }
                }
                buf
            }
        };

        // 4. 按 strides gather 到连续输出
        let shape = layout.shape.as_slice();
        let strides = layout.strides.as_slice();
        let offset = layout.offset;
        let ndim = shape.len();

        if ndim == 0 {
            // 0-dim：单元素，offset 即位置
            let src_start = offset * elem_size;
            bytes.copy_from_slice(&host_buffer[src_start..src_start + elem_size]);
        } else {
            let mut coords = vec![0usize; ndim];
            for i in 0..n_elements {
                // 计算当前多维坐标的线性偏移
                let mut linear = offset as isize;
                for d in 0..ndim {
                    linear += coords[d] as isize * strides[d];
                }
                let src_start = linear as usize * elem_size;
                let dst_start = i * elem_size;
                bytes[dst_start..dst_start + elem_size]
                    .copy_from_slice(&host_buffer[src_start..src_start + elem_size]);

                // 递增坐标（C order：最右维最快）
                for d in (0..ndim).rev() {
                    coords[d] += 1;
                    if coords[d] < shape[d] {
                        break;
                    }
                    coords[d] = 0;
                }
            }
        }

        Ok(bytes)
    }
}

// ============================================================
// bytes → Python 转换辅助函数
// ============================================================

/// 将 flat Python list 按 shape 递归嵌套为多维 list。
///
/// 例如 flat=[1,2,3,4,5,6], shape=[2,3] → [[1,2,3],[4,5,6]]
fn nest_flat_list(
    py: Python<'_>,
    flat: &Bound<PyAny>,
    shape: &[usize],
    dim: usize,
) -> PyResult<PyObject> {
    use pyo3::types::PyList;

    let outer_len = shape[dim];
    // 每个子数组的元素数（剩余维度的乘积）
    let inner_size: usize = shape[dim + 1..].iter().product();

    let flat_list = flat.downcast::<PyList>()?;
    let result = PyList::empty(py);

    for i in 0..outer_len {
        let start = i * inner_size;
        let end = start + inner_size;
        let sub = flat_list.get_slice(start, end);
        if dim + 1 < shape.len() - 1 {
            // 还有更深层：继续递归嵌套
            let nested = nest_flat_list(py, &sub, shape, dim + 1)?;
            result.append(nested)?;
        } else {
            // 最内层：直接添加子 list
            result.append(sub)?;
        }
    }

    Ok(result.into())
}

/// 将原始字节按 dtype 解释为 Python list。
fn bytes_to_pylist(py: Python<'_>, bytes: &[u8], n: usize, dtype: Dtype) -> PyResult<PyObject> {
    if n == 0 {
        return Ok(PyList::empty(py).into());
    }

    macro_rules! to_list {
        ($t:ty) => {{
            let v: &[$t] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const $t, n) };
            return Ok(PyList::new(py, v.iter().copied())?.into());
        }};
    }

    match dtype {
        Dtype::Bool => to_list!(bool),
        Dtype::Int8 => to_list!(i8),
        Dtype::Int16 => to_list!(i16),
        Dtype::Int32 => to_list!(i32),
        Dtype::Int64 => to_list!(i64),
        Dtype::Uint8 => to_list!(u8),
        Dtype::Uint16 => to_list!(u16),
        Dtype::Uint32 => to_list!(u32),
        Dtype::Uint64 => to_list!(u64),
        Dtype::Float32 => to_list!(f32),
        Dtype::Float64 => to_list!(f64),
        Dtype::Float16 | Dtype::Bfloat16 | Dtype::Complex64 | Dtype::Complex128 => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "tolist not yet supported for dtype {}",
                dtype
            )))
        }
    }
}

/// 将单个元素的原始字节按 dtype 解释为 Python 标量。
#[allow(deprecated)]
fn bytes_to_scalar(py: Python<'_>, bytes: &[u8], dtype: Dtype) -> PyResult<PyObject> {
    macro_rules! to_scalar {
        ($t:ty) => {{
            let v: $t = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const $t) };
            return Ok(v.into_py(py));
        }};
    }

    match dtype {
        Dtype::Bool => to_scalar!(bool),
        Dtype::Int8 => to_scalar!(i8),
        Dtype::Int16 => to_scalar!(i16),
        Dtype::Int32 => to_scalar!(i32),
        Dtype::Int64 => to_scalar!(i64),
        Dtype::Uint8 => to_scalar!(u8),
        Dtype::Uint16 => to_scalar!(u16),
        Dtype::Uint32 => to_scalar!(u32),
        Dtype::Uint64 => to_scalar!(u64),
        Dtype::Float32 => to_scalar!(f32),
        Dtype::Float64 => to_scalar!(f64),
        Dtype::Float16 | Dtype::Bfloat16 | Dtype::Complex64 | Dtype::Complex128 => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "item not yet supported for dtype {}",
                dtype
            )))
        }
    }
}
