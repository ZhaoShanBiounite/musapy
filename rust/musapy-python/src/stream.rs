//! PyStream — Python 绑定的 Stream 类（ADR L1-7, L1-9）。
//!
//! 每个设备有一个默认 stream，用户也可创建自定义 stream。

use crate::device::PyDevice;
use crate::error;
use musapy_core::{Device, Stream};
use pyo3::prelude::*;
use std::sync::Arc;

/// Python Stream 类。
///
/// 构造：`ms.Stream(device="musa:0", priority=0)`
#[pyclass(name = "Stream", module = "musapy")]
pub struct PyStream {
    pub(crate) inner: Arc<Stream>,
}

impl PyStream {
    /// 从 musapy-core Arc<Stream> 构造（内部用）。
    pub fn from_stream(stream: Arc<Stream>) -> Self {
        Self { inner: stream }
    }

    /// 返回内部 Arc<Stream> 的克隆（内部用）。
    pub fn inner(&self) -> Arc<Stream> {
        Arc::clone(&self.inner)
    }
}

#[pymethods]
impl PyStream {
    /// 创建流。
    ///
    /// 参数：
    ///   device: str 或 Device，如 "musa:0" 或 "cpu"
    ///   priority: int，流优先级（默认 0）
    #[new]
    #[pyo3(signature = (device = "cpu".to_string(), priority = 0))]
    fn new(device: String, priority: i32) -> PyResult<Self> {
        let dev = Device::parse(&device).map_err(error::to_pyerr)?;
        let stream = Stream::new(dev, priority).map_err(error::to_pyerr)?;
        Ok(Self::from_stream(Arc::new(stream)))
    }

    /// 等待流上所有操作完成。
    fn synchronize(&self) -> PyResult<()> {
        self.inner.synchronize().map_err(error::to_pyerr)
    }

    /// 流优先级。
    #[getter]
    fn priority(&self) -> i32 {
        self.inner.priority()
    }

    /// 流所属设备。
    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_device(self.inner.device().clone())
    }

    /// 流的唯一 ID。
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id()
    }

    /// 是否已中毒（内部错误标记）。
    #[getter]
    fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }

    /// 待处理操作数。
    #[getter]
    fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Stream(device={}, priority={}, id={})",
            self.inner.device(),
            self.inner.priority(),
            self.inner.id()
        )
    }
}
