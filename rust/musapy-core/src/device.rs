//! Device 标识与解析元数据（ADR L1-1, L1-2, L0-6, L0-8）
//!
//! 职责：
//!   1. Device 枚举：Cpu / Musa(u32)，支持字符串解析 "cpu" / "musa:0"
//!   2. ResolutionSource：5 级优先级链（Arg > Context > InputArray > GlobalDefault > AutoProbe）
//!   3. DeviceResolution：每次设备解析的可追溯记录，附在 Array 上
//!   4. SourceLocation：debug 模式下的源代码位置（Phase 4 的 P4.10 实现真实捕获）

use crate::error::{DeviceError, Result};
use std::fmt;

// ============================================================
// 1. Device 枚举
// ============================================================

/// 设备标识符（ADR L1-1）。
///
/// 支持两种设备：
/// - `Cpu`：主机内存
/// - `Musa(u32)`：摩尔线程 GPU，u32 是设备 ID（0, 1, 2, ...）
///
/// 字符串格式：`"cpu"` / `"musa:0"` / `"musa:1"`，大小写不敏感。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Musa(u32),
}

impl Device {
    /// 从字符串解析设备标识。
    ///
    /// 支持格式（大小写不敏感，容忍前后空格）：
    /// - `"cpu"` → `Device::Cpu`
    /// - `"musa:0"`, `"musa:1"` → `Device::Musa(n)`
    ///
    /// 非法格式返回 `DeviceError::Unavailable`。
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "cpu" => Ok(Device::Cpu),
            s if s.starts_with("musa:") => {
                let id_str = &s["musa:".len()..];
                let id: u32 = id_str.parse().map_err(|_| {
                    DeviceError::Unavailable(format!(
                        "invalid device id in '{}': expected 'musa:<number>'",
                        s
                    ))
                })?;
                Ok(Device::Musa(id))
            }
            _ => Err(DeviceError::Unavailable(format!(
                "cannot parse device '{}': expected 'cpu' or 'musa:<id>'",
                s
            ))
            .into()),
        }
    }

    /// 是否为 MUSA GPU 设备。
    pub fn is_musa(&self) -> bool {
        matches!(self, Device::Musa(_))
    }

    /// 如果是 MUSA 设备，返回设备 ID；否则返回 None。
    pub fn musa_id(&self) -> Option<u32> {
        match self {
            Device::Musa(id) => Some(*id),
            Device::Cpu => None,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Musa(id) => write!(f, "musa:{}", id),
        }
    }
}

// ============================================================
// 2. ResolutionSource — 5 级优先级链（ADR L0-6）
// ============================================================

/// 设备/dtype 解析来源优先级（ADR L0-6, L0-7）。
///
/// 5 级优先级链，数字越小优先级越高：
/// 1. `Arg` — 函数调用时显式传入的 `device=` 参数
/// 2. `Context` — `with ms.device(...)` 上下文
/// 3. `InputArray` — 输入 Array 的 device（ufunc 风格，`a + b` 跟随 `a`）
/// 4. `GlobalDefault` — `ms.set_default_device()` 设置的全局默认（线程局部）
/// 5. `AutoProbe` — 启动期自动探测（优先 MUSA over CPU）
///
/// 每级可被更高优先级覆盖，每次解析都生成可追溯记录（L0-8）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionSource {
    Arg,
    Context,
    InputArray,
    GlobalDefault,
    AutoProbe,
}

impl ResolutionSource {
    /// 优先级数字（1=最高，5=最低）。
    pub fn priority(&self) -> u8 {
        match self {
            ResolutionSource::Arg => 1,
            ResolutionSource::Context => 2,
            ResolutionSource::InputArray => 3,
            ResolutionSource::GlobalDefault => 4,
            ResolutionSource::AutoProbe => 5,
        }
    }
}

impl fmt::Display for ResolutionSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolutionSource::Arg => write!(f, "arg"),
            ResolutionSource::Context => write!(f, "context"),
            ResolutionSource::InputArray => write!(f, "input_array"),
            ResolutionSource::GlobalDefault => write!(f, "global_default"),
            ResolutionSource::AutoProbe => write!(f, "auto_probe"),
        }
    }
}

// ============================================================
// 3. SourceLocation — 源代码位置（debug 模式，ADR L0-8, L3-26）
// ============================================================

/// 源代码位置（debug 模式下捕获）。
///
/// Phase 2 只定义结构；Phase 4 的 P4.10 在 `#[cfg(debug)]` 下实现真实捕获
/// （从 PyO3 调用栈提取 file/line/function）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

// ============================================================
// 4. DeviceResolution — 解析记录（ADR L0-8）
// ============================================================

/// 设备解析记录（ADR L0-8）。
///
/// 每次数组创建时的设备解析都生成此记录，附在 Array 上。
/// 用于分布式调试："为什么数据跑到了错误的设备"。
///
/// `Display` 输出示例：
/// ```text
/// musa:0  # resolved from: global_default at <stdin>:3
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceResolution {
    pub device: Device,
    pub source: ResolutionSource,
    pub source_location: Option<SourceLocation>,
}

impl DeviceResolution {
    pub fn new(device: Device, source: ResolutionSource) -> Self {
        Self {
            device,
            source,
            source_location: None,
        }
    }

    pub fn with_location(mut self, loc: SourceLocation) -> Self {
        self.source_location = Some(loc);
        self
    }
}

impl fmt::Display for DeviceResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.device)?;
        write!(f, "  # resolved from: {}", self.source)?;
        if let Some(ref loc) = self.source_location {
            write!(f, " at {}", loc)?;
        }
        Ok(())
    }
}

// ============================================================
// 5. 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Device::parse ---

    #[test]
    fn parse_cpu() {
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
    }

    #[test]
    fn parse_cpu_case_insensitive() {
        assert_eq!(Device::parse("CPU").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("Cpu").unwrap(), Device::Cpu);
    }

    #[test]
    fn parse_cpu_with_spaces() {
        assert_eq!(Device::parse("  cpu  ").unwrap(), Device::Cpu);
    }

    #[test]
    fn parse_musa() {
        assert_eq!(Device::parse("musa:0").unwrap(), Device::Musa(0));
        assert_eq!(Device::parse("musa:1").unwrap(), Device::Musa(1));
        assert_eq!(Device::parse("musa:42").unwrap(), Device::Musa(42));
    }

    #[test]
    fn parse_musa_case_insensitive() {
        assert_eq!(Device::parse("MUSA:0").unwrap(), Device::Musa(0));
        assert_eq!(Device::parse("Musa:1").unwrap(), Device::Musa(1));
    }

    #[test]
    fn parse_musa_with_spaces() {
        assert_eq!(Device::parse("  musa:0  ").unwrap(), Device::Musa(0));
    }

    #[test]
    fn parse_invalid() {
        assert!(Device::parse("gpu:0").is_err());
        assert!(Device::parse("cuda:0").is_err());
        assert!(Device::parse("musa").is_err());
        assert!(Device::parse("musa:").is_err());
        assert!(Device::parse("musa:abc").is_err());
        assert!(Device::parse("").is_err());
    }

    // --- Device 方法 ---

    #[test]
    fn is_musa() {
        assert!(!Device::Cpu.is_musa());
        assert!(Device::Musa(0).is_musa());
        assert!(Device::Musa(5).is_musa());
    }

    #[test]
    fn musa_id() {
        assert_eq!(Device::Cpu.musa_id(), None);
        assert_eq!(Device::Musa(0).musa_id(), Some(0));
        assert_eq!(Device::Musa(7).musa_id(), Some(7));
    }

    // --- Device::Display ---

    #[test]
    fn display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Musa(0).to_string(), "musa:0");
        assert_eq!(Device::Musa(3).to_string(), "musa:3");
    }

    // --- ResolutionSource ---

    #[test]
    fn resolution_source_priority() {
        assert_eq!(ResolutionSource::Arg.priority(), 1);
        assert_eq!(ResolutionSource::Context.priority(), 2);
        assert_eq!(ResolutionSource::InputArray.priority(), 3);
        assert_eq!(ResolutionSource::GlobalDefault.priority(), 4);
        assert_eq!(ResolutionSource::AutoProbe.priority(), 5);
    }

    #[test]
    fn resolution_source_display() {
        assert_eq!(ResolutionSource::Arg.to_string(), "arg");
        assert_eq!(
            ResolutionSource::GlobalDefault.to_string(),
            "global_default"
        );
    }

    // --- DeviceResolution ---

    #[test]
    fn device_resolution_display_without_location() {
        let r = DeviceResolution::new(Device::Musa(0), ResolutionSource::GlobalDefault);
        assert_eq!(r.to_string(), "musa:0  # resolved from: global_default");
    }

    #[test]
    fn device_resolution_display_with_location() {
        let r = DeviceResolution::new(Device::Musa(0), ResolutionSource::GlobalDefault)
            .with_location(SourceLocation {
                file: "<stdin>".to_string(),
                line: 3,
                column: 0,
            });
        assert_eq!(
            r.to_string(),
            "musa:0  # resolved from: global_default at <stdin>:3"
        );
    }

    #[test]
    fn device_resolution_equality() {
        let r1 = DeviceResolution::new(Device::Cpu, ResolutionSource::Arg);
        let r2 = DeviceResolution::new(Device::Cpu, ResolutionSource::Arg);
        assert_eq!(r1, r2);
    }
}
