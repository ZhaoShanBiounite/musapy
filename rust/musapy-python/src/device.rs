//! PyDevice — Python 绑定的 Device 类（ADR L1-1, L0-8）。
//!
//! 支持字符串构造（`"cpu"` / `"musa:0"`）和从 Array 解析出的 device。
//! `__repr__` 显示 resolution source（ADR L0-8 反馈原则）。

use musapy_core::{Device, DeviceResolution};
use pyo3::prelude::*;

use crate::error;

/// Python Device 类。
///
/// 构造：`ms.Device("musa:0")` 或 `ms.Device("cpu")`
///
/// 从 Array 解析出的 device 会附加 resolution 信息：
///   `Device(musa:0)  # resolved from: global_default`
#[pyclass(name = "Device", module = "musapy")]
pub struct PyDevice {
    pub(crate) inner: Device,
    pub(crate) resolution: Option<DeviceResolution>,
}

impl PyDevice {
    /// 从 musapy-core Device 构造（内部，供 ops.rs 用）。
    pub fn from_device(device: Device) -> Self {
        Self {
            inner: device,
            resolution: None,
        }
    }

    /// 从解析结果构造（供 PyArray.device getter 用）。
    pub fn from_resolution(res: &DeviceResolution) -> Self {
        Self {
            inner: res.device.clone(),
            resolution: Some(res.clone()),
        }
    }
}

#[pymethods]
impl PyDevice {
    /// 从字符串构造：`Device("musa:0")` / `Device("cpu")`。
    #[new]
    fn new(spec: &str) -> PyResult<Self> {
        let device = Device::parse(spec).map_err(error::to_pyerr)?;
        Ok(Self::from_device(device))
    }

    /// `Device(musa:0)` 或带 resolution：`Device(musa:0)  # resolved from: global_default`
    fn __repr__(&self) -> String {
        match &self.resolution {
            None => format!("Device({})", self.inner),
            Some(res) => format!("Device({})  # resolved from: {}", self.inner, res.source),
        }
    }

    /// `"musa:0"` / `"cpu"`
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    /// 字符串形式，与 `__str__` 相同。
    fn __format__(&self, _fmt: &str) -> String {
        self.inner.to_string()
    }

    /// 是否为 MUSA GPU 设备。
    #[getter]
    fn is_musa(&self) -> bool {
        self.inner.is_musa()
    }

    /// MUSA 设备 ID（CPU 返回 None）。
    #[getter]
    fn musa_id(&self) -> Option<u32> {
        self.inner.musa_id()
    }

    /// 解析来源（如 "global_default"），无 resolution 时为 None。
    #[getter]
    fn resolution_source(&self) -> Option<String> {
        self.resolution.as_ref().map(|r| r.source.to_string())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
        let s = self.inner.to_string();
        hasher.write(s.as_bytes());
        hasher.finish()
    }
}
