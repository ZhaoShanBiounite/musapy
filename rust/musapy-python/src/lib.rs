//! musapy-python: PyO3 绑定层（ADR L2-6, L2-7）
//!
//! Phase 1: 最小 PyO3 模块，暴露 __version__ 和 startup_report()
//! Phase 5+: 加 PyDevice/PyDtype/PyStream/PyArray + ms.array() 等

use pyo3::prelude::*;

/// musapy Python 扩展模块入口。
///
/// pyproject.toml 里 module-name = "musapy._core"，
/// 所以本函数名必须是 `_core`，对应 C 符号 `PyInit__core`。
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", musapy_core::VERSION)?;
    m.add_function(wrap_pyfunction!(startup_report, m)?)?;
    Ok(())
}

/// 返回 musapy 启动期 ABI 校验报告字符串。
///
/// 报告内容示例（真实环境）：
///   musapy ABI v1, MUSA Runtime 3.1.0, mcc/clang clang version 14.0.0 (...)
///
/// 报告内容示例（mock 模式）：
///   musapy ABI v1, MUSA Runtime unknown, mcc/clang unknown [MOCK]
#[pyfunction]
fn startup_report() -> PyResult<String> {
    match musapy_core::abi::run_startup_checks() {
        Ok(r) => Ok(r.to_string()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}