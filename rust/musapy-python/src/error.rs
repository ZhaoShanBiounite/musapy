//! Rust → Python 错误映射（ADR L3-5, L3-6, L3-7）。
//!
//! Python 异常层次：
//! ```text
//! Exception
//!   └── MusapyError              # 顶层 musapy 错误
//!        ├── DeviceError
//!        │    └── DeviceNotConfiguredError
//!        ├── DtypeError
//!        ├── ShapeError
//!        ├── MemoryError
//!        │    └── OutOfMemoryError
//!        ├── StreamError
//!        ├── KernelError
//!        ├── InteropError
//!        └── LinAlgError          # v0.3 Phase 2（ADR-003 003-D3）
//! ```
//!
//! ADR L3-6 理想状态要求部分继承 Python builtins
//! （如 DeviceError(MusapyError, RuntimeError)），
//! 但 PyO3 create_exception! 宏仅支持单继承。当前实现为单继承层次，
//! 后续可按需用 #[pyclass(extends=...)] 扩展为多重继承。
//!
//! ADR L3-7：OutOfMemoryError 不继承 Python's MemoryError。

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// ============================================================
// Python 异常类定义
// ============================================================

create_exception!(musapy, MusapyError, PyException);
create_exception!(musapy, DeviceError, MusapyError);
create_exception!(musapy, DeviceNotConfiguredError, DeviceError);
create_exception!(musapy, DtypeError, MusapyError);
create_exception!(musapy, ShapeError, MusapyError);
create_exception!(musapy, MemoryError, MusapyError);
create_exception!(musapy, OutOfMemoryError, MemoryError);
create_exception!(musapy, StreamError, MusapyError);
create_exception!(musapy, KernelError, MusapyError);
create_exception!(musapy, InteropError, MusapyError);
create_exception!(musapy, LinAlgError, MusapyError);

// ============================================================
// MusapyError → PyErr 转换辅助函数
// ============================================================

/// 将 musapy-core 的 MusapyError 转换为对应的 Python 异常。
///
/// 用法：`.map_err(error::to_pyerr)?`
pub fn to_pyerr(e: musapy_core::MusapyError) -> PyErr {
    let msg = e.to_string();
    match e {
        musapy_core::MusapyError::Device(de) => match de {
            musapy_core::DeviceError::NotConfigured => DeviceNotConfiguredError::new_err(msg),
            _ => DeviceError::new_err(msg),
        },
        musapy_core::MusapyError::Dtype(_) => DtypeError::new_err(msg),
        musapy_core::MusapyError::Shape(_) => ShapeError::new_err(msg),
        musapy_core::MusapyError::Memory(me) => match me {
            musapy_core::MemoryError::OutOfMemory(_) => OutOfMemoryError::new_err(msg),
            _ => MemoryError::new_err(msg),
        },
        musapy_core::MusapyError::Stream(_) => StreamError::new_err(msg),
        musapy_core::MusapyError::Kernel(_) => KernelError::new_err(msg),
        musapy_core::MusapyError::Interop(_) => InteropError::new_err(msg),
        musapy_core::MusapyError::LinAlg(_) => LinAlgError::new_err(msg),
        // 高级索引越界（Phase 8，ADR-003 003-D3 扩展）：抛 Python 内置
        // IndexError（NumPy 兼容，L3-6 单继承限制下不继承 MusapyError）
        musapy_core::MusapyError::Index(_) => pyo3::exceptions::PyIndexError::new_err(msg),
    }
}

// ============================================================
// 模块注册辅助
// ============================================================

/// 在 #[pymodule] 中注册所有异常类。
///
/// 用法：`error::register_exceptions(m)?;`
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("MusapyError", m.py().get_type::<MusapyError>())?;
    m.add("DeviceError", m.py().get_type::<DeviceError>())?;
    m.add(
        "DeviceNotConfiguredError",
        m.py().get_type::<DeviceNotConfiguredError>(),
    )?;
    m.add("DtypeError", m.py().get_type::<DtypeError>())?;
    m.add("ShapeError", m.py().get_type::<ShapeError>())?;
    m.add("MemoryError", m.py().get_type::<MemoryError>())?;
    m.add("OutOfMemoryError", m.py().get_type::<OutOfMemoryError>())?;
    m.add("StreamError", m.py().get_type::<StreamError>())?;
    m.add("KernelError", m.py().get_type::<KernelError>())?;
    m.add("InteropError", m.py().get_type::<InteropError>())?;
    m.add("LinAlgError", m.py().get_type::<LinAlgError>())?;
    Ok(())
}
