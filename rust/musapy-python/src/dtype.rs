//! PyDtype — Python 绑定的 Dtype 类（ADR L1-4）。
//!
//! 15 种 dtype（bool, int8/16/32/64, uint8/16/32/64, float16/32/64, bfloat16,
//! complex64/128）。通过模块常量访问：`ms.float32`, `ms.int32` 等。

use musapy_core::Dtype;
use pyo3::prelude::*;

/// Python Dtype 类。
///
/// 构造：`ms.Dtype("float32")` 或直接用常量 `ms.float32`。
#[pyclass(name = "Dtype", module = "musapy", eq, frozen, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyDtype(pub Dtype);

impl PyDtype {
    /// 从 musapy-core Dtype 构造（内部用）。
    pub fn from_dtype(dtype: Dtype) -> Self {
        Self(dtype)
    }

    /// 返回内部 Dtype（内部用）。
    pub fn inner(&self) -> Dtype {
        self.0
    }
}

/// 将 dtype 名称字符串解析为 Dtype。
fn parse_dtype(name: &str) -> Option<Dtype> {
    match name.to_lowercase().as_str() {
        "bool" => Some(Dtype::Bool),
        "int8" => Some(Dtype::Int8),
        "int16" => Some(Dtype::Int16),
        "int32" => Some(Dtype::Int32),
        "int64" => Some(Dtype::Int64),
        "uint8" => Some(Dtype::Uint8),
        "uint16" => Some(Dtype::Uint16),
        "uint32" => Some(Dtype::Uint32),
        "uint64" => Some(Dtype::Uint64),
        "float16" | "half" => Some(Dtype::Float16),
        "float32" | "single" => Some(Dtype::Float32),
        "float64" | "double" => Some(Dtype::Float64),
        "bfloat16" | "bf16" => Some(Dtype::Bfloat16),
        "complex64" => Some(Dtype::Complex64),
        "complex128" => Some(Dtype::Complex128),
        _ => None,
    }
}

#[pymethods]
impl PyDtype {
    /// 从字符串构造：`Dtype("float32")`。
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        parse_dtype(name)
            .map(Self)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("unknown dtype: {}", name))
            })
    }

    /// `"float32"` 等。
    fn __repr__(&self) -> String {
        self.0.to_string()
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __format__(&self, _fmt: &str) -> String {
        self.0.to_string()
    }

    /// dtype 名称（如 `"float32"`）。
    #[getter]
    fn name(&self) -> String {
        self.0.to_string()
    }

    /// 元素大小（字节）。
    #[getter]
    fn element_size(&self) -> usize {
        self.0.element_size()
    }

    /// 是否为浮点类型。
    #[getter]
    fn is_floating(&self) -> bool {
        self.0.is_floating()
    }

    /// 是否为整数类型（bool 也算）。
    #[getter]
    fn is_integer(&self) -> bool {
        self.0.is_integer()
    }

    /// 是否为复数类型。
    #[getter]
    fn is_complex(&self) -> bool {
        self.0.is_complex()
    }
}

// ============================================================
// 常量注册
// ============================================================

/// 在 #[pymodule] 中注册 15 个 dtype 常量。
pub fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("bool", PyDtype(Dtype::Bool))?;
    m.add("int8", PyDtype(Dtype::Int8))?;
    m.add("int16", PyDtype(Dtype::Int16))?;
    m.add("int32", PyDtype(Dtype::Int32))?;
    m.add("int64", PyDtype(Dtype::Int64))?;
    m.add("uint8", PyDtype(Dtype::Uint8))?;
    m.add("uint16", PyDtype(Dtype::Uint16))?;
    m.add("uint32", PyDtype(Dtype::Uint32))?;
    m.add("uint64", PyDtype(Dtype::Uint64))?;
    m.add("float16", PyDtype(Dtype::Float16))?;
    m.add("float32", PyDtype(Dtype::Float32))?;
    m.add("float64", PyDtype(Dtype::Float64))?;
    m.add("bfloat16", PyDtype(Dtype::Bfloat16))?;
    m.add("complex64", PyDtype(Dtype::Complex64))?;
    m.add("complex128", PyDtype(Dtype::Complex128))?;
    Ok(())
}
