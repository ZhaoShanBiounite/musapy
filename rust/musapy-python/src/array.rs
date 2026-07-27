//! PyArray — Python 绑定的 Array 类（ADR L1-11, L3-27, L0-8）。
//!
//! 从 Buffer + Layout + resolution 构造，提供只读属性和 name 管理。
//! `device` getter 返回带 resolution source 的 PyDevice（L0-8 反馈原则）。

use crate::device::PyDevice;
use crate::dtype::PyDtype;
use crate::stream::PyStream;
use musapy_core::Array;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

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
