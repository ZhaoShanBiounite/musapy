//! PyDtype — Python 绑定的 Dtype 类（ADR L1-4）。
//!
//! 15 种 dtype（bool, int8/16/32/64, uint8/16/32/64, float16/32/64, bfloat16,
//! complex64/128）。通过模块常量访问：`ms.float32`, `ms.int32` 等。
//!
//! dtype 参数接受三种形式（v0.3：`dtype='f32'` 字符串语法为主）：
//!   - 字符串短别名：`'f32'` / `'i64'` / `'c64'` / `'b1'` 等
//!   - 字符串全名：`'float32'` / `'int64'` 等（兼容）
//!   - Dtype 实例：`ms.float32` 常量或 `Dtype('float32')`（向后兼容）

use musapy_core::Dtype;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;

/// Python Dtype 类。
///
/// 构造：`ms.Dtype("float32")` 或直接用常量 `ms.float32`。
///
/// 注意：不实现 Clone（pyo3 的 blanket FromPyObject 要求 `PyClass + Clone`，
/// 会与下方手写的字符串提取 impl 冲突）；参数提取/返回均按值构造。
#[pyclass(name = "Dtype", module = "musapy", frozen)]
#[derive(PartialEq, Eq, Hash)]
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
///
/// 支持：
///   - 短别名：`'f32'`/`'f64'`/`'i8'`-`'i64'`/`'u8'`-`'u64'`/`'f16'`/`'bf16'`/
///     `'c64'`/`'c128'`/`'b1'`
///   - 全名：`'float32'`/`'complex128'` 等
///   - 兼容名：`'half'`/`'single'`/`'double'`/`'bf16'`
pub(crate) fn parse_dtype(name: &str) -> Option<Dtype> {
    match name.to_lowercase().as_str() {
        "bool" | "b1" | "bool8" => Some(Dtype::Bool),
        "int8" | "i8" => Some(Dtype::Int8),
        "int16" | "i16" => Some(Dtype::Int16),
        "int32" | "i32" => Some(Dtype::Int32),
        "int64" | "i64" => Some(Dtype::Int64),
        "uint8" | "u8" => Some(Dtype::Uint8),
        "uint16" | "u16" => Some(Dtype::Uint16),
        "uint32" | "u32" => Some(Dtype::Uint32),
        "uint64" | "u64" => Some(Dtype::Uint64),
        "float16" | "half" | "f16" => Some(Dtype::Float16),
        "float32" | "single" | "f32" => Some(Dtype::Float32),
        "float64" | "double" | "f64" => Some(Dtype::Float64),
        "bfloat16" | "bf16" => Some(Dtype::Bfloat16),
        "complex64" | "c64" => Some(Dtype::Complex64),
        "complex128" | "c128" => Some(Dtype::Complex128),
        _ => None,
    }
}

/// dtype 参数提取：接受 Dtype 实例或 dtype 字符串。
///
/// ```python
/// ms.array([1, 2], dtype='f32')          # 短别名
/// ms.array([1, 2], dtype='float32')      # 全名
/// ms.array([1, 2], dtype=ms.float32)     # 常量（向后兼容）
/// ```
impl<'py> FromPyObject<'py> for PyDtype {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        // 1. Dtype 实例（常量或 Dtype("float32")）
        if let Ok(d) = obj.downcast::<PyDtype>() {
            return Ok(PyDtype(d.borrow().0));
        }
        // 2. dtype 字符串
        if let Ok(s) = obj.extract::<String>() {
            if let Some(dtype) = parse_dtype(&s) {
                return Ok(PyDtype(dtype));
            }
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown dtype: {}",
                s
            )));
        }
        // 3. 其他类型 → TypeError
        Err(pyo3::exceptions::PyTypeError::new_err(
            "dtype must be a Dtype instance or a dtype string (e.g. 'f32', 'float32')",
        ))
    }
}

#[pymethods]
impl PyDtype {
    /// 从字符串构造：`Dtype("float32")`。
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        parse_dtype(name).map(Self).ok_or_else(|| {
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

    /// 相等比较：`a.dtype == ms.float32` 或 `a.dtype == 'float32'` / `'f32'`。
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(d) = other.downcast::<PyDtype>() {
            return Ok(self.0 == d.borrow().0);
        }
        if let Ok(s) = other.extract::<String>() {
            return Ok(parse_dtype(&s).is_some_and(|d| d == self.0));
        }
        Ok(false)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
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
