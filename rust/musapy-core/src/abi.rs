// musapy ABI 版本管理与兼容性校验 (P1.8)
//
// 职责：
//   1. 暴露编译期嵌入的 ABI 版本常量（由 build.rs 注入）
//   2. 生成 kernel 符号名：musapy_<op>_<dtype>_v<ABI>
//   3. 运行时校验：kernel 期望的 ABI 版本 vs 运行时 ABI 版本
//   4. 运行时校验：mcc 版本 vs 运行时 ABI 版本的兼容性矩阵
//
// 设计依据：ADR L2-1（Build System）
//   - ABI 版本嵌入符号名：musapy_mul_f32_v1
//   - 运行时检查 kernel ABI
//   - mcc 版本与 runtime 版本兼容性矩阵检查

use std::fmt;

// ============================================================
// 1. ABI 版本常量
// ============================================================

/// 编译期嵌入的 ABI 版本号。
///
/// 由 build.rs 通过 `cargo:rustc-env=MUSAPY_ABI_VERSION` 注入。
/// 每次 kernel ABI 破坏性变更时在 build.rs 的 `ABI_VERSION` 常量处 +1。
///
/// 用 const fn 手写解析，因为 `str::parse` 不是 const fn。
pub const ABI_VERSION: u32 = {
    let raw = env!("MUSAPY_ABI_VERSION");
    let bytes = raw.as_bytes();
    let mut n: u32 = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        n = n * 10 + (b - b'0') as u32;
        i += 1;
    }
    n
};

/// musapy 运行时支持的最低 ABI 版本。
///
/// 低于此版本的 kernel 被视为过旧，拒绝加载。
/// 当前 ABI_VERSION 即唯一支持版本；未来引入 v2 时，
/// 若 v1 仍可回退兼容，则把此值保持为 1。
pub const MIN_SUPPORTED_ABI_VERSION: u32 = 1;

/// mcc 版本号原始字符串（编译期由 build.rs 探测注入）。
///
/// mock 模式或探测失败时为空字符串。
/// 用 `env!("VAR", "")` 形式提供默认值，避免变量不存在时编译失败。
pub const MCC_VERSION_RAW: &str = match option_env!("MUSAPY_MCC_VERSION") {
    Some(v) => v,
    None => "",
};

// ============================================================
// 2. Error 类型
// ============================================================

/// ABI 校验相关的错误。
///
/// Phase 2（P2.1）定义完整 Error 枚举时，可通过 `#[from]` 将本类型合并进去。
#[derive(Debug)]
pub enum AbiError {
    /// kernel 期望的 ABI 版本低于运行时最低支持版本
    KernelAbiTooOld {
        kernel_abi: u32,
        min_supported: u32,
    },
    /// kernel 期望的 ABI 版本高于运行时当前版本（运行时需升级）
    KernelAbiTooNew {
        kernel_abi: u32,
        runtime_abi: u32,
    },
    /// 无法从 kernel 符号名中解析出版本后缀
    InvalidKernelSymbol(String),
    /// 无法解析 mcc 版本号字符串
    InvalidMccVersion(String),
    /// mcc 版本不支持当前运行时 ABI 版本
    MccIncompatible {
        mcc: SemVer,
        runtime_abi: u32,
        max_abi_supported: u32,
    },
}

impl fmt::Display for AbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbiError::KernelAbiTooOld {
                kernel_abi,
                min_supported,
            } => write!(
                f,
                "kernel ABI version {} is older than minimum supported {}",
                kernel_abi, min_supported
            ),
            AbiError::KernelAbiTooNew {
                kernel_abi,
                runtime_abi,
            } => write!(
                f,
                "kernel ABI version {} is newer than runtime ABI {} (upgrade musapy)",
                kernel_abi, runtime_abi
            ),
            AbiError::InvalidKernelSymbol(s) => {
                write!(f, "invalid kernel symbol, cannot parse ABI version: {}", s)
            }
            AbiError::InvalidMccVersion(s) => {
                write!(f, "cannot parse mcc version string: {:?}", s)
            }
            AbiError::MccIncompatible {
                mcc,
                runtime_abi,
                max_abi_supported,
            } => write!(
                f,
                "mcc {} supports ABI up to v{}, but runtime is v{} (upgrade mcc)",
                mcc, max_abi_supported, runtime_abi
            ),
        }
    }
}

impl std::error::Error for AbiError {}

// ============================================================
// 3. 语义化版本号
// ============================================================

/// 解析后的 mcc 语义化版本号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SemVer {
    /// 从字符串中提取第一个 `数字.数字.数字` 序列。
    ///
    /// 容忍前缀，如 `"mcc 1.2.0"` → `SemVer{1,2,0}`。
    /// 容忍缺失的 minor/patch，如 `"1"` → `SemVer{1,0,0}`。
    pub fn parse(s: &str) -> Result<Self, AbiError> {
        let bytes = s.as_bytes();

        // 跳过前导非数字字符
        let mut start = 0;
        while start < bytes.len() && !bytes[start].is_ascii_digit() {
            start += 1;
        }
        if start >= bytes.len() {
            return Err(AbiError::InvalidMccVersion(s.to_string()));
        }

        // 按 '.' 分段，每段只取连续数字
        let rest = &s[start..];
        let mut parts = rest.split('.');
        let parse_part = |p: Option<&str>| -> u32 {
            p.and_then(|seg| {
                let digits: String = seg.chars().take_while(|c| c.is_ascii_digit()).collect();
                if digits.is_empty() {
                    None
                } else {
                    digits.parse().ok()
                }
            })
            .unwrap_or(0)
        };

        let major = parse_part(parts.next());
        let minor = parse_part(parts.next());
        let patch = parse_part(parts.next());

        if major == 0 && minor == 0 && patch == 0 {
            return Err(AbiError::InvalidMccVersion(s.to_string()));
        }

        Ok(SemVer {
            major,
            minor,
            patch,
        })
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ============================================================
// 4. 公开 API
// ============================================================

/// 返回当前运行时编译期嵌入的 ABI 版本。
///
/// 对应 P1.8 要求导出的 `musapy_abi_version()`。
pub fn musapy_abi_version() -> u32 {
    ABI_VERSION
}

/// 生成 kernel 符号名：`musapy_<op>_<dtype>_v<ABI>`。
///
/// 例如 `kernel_symbol("add", "f32")` → `"musapy_add_f32_v1"`。
///
/// 所有 kernel 符号必须经此函数生成，确保版本后缀统一。
/// Phase 6 的 musapy-ops 会用它拼 `extern "C"` 符号名。
pub fn kernel_symbol(op: &str, dtype: &str) -> String {
    format!("musapy_{}_{}_v{}", op, dtype, ABI_VERSION)
}

/// 从 kernel 符号名中解析 ABI 版本后缀。
///
/// 输入 `"musapy_add_f32_v1"` → `Ok(1)`。
/// 用 `rfind("_v")` 定位最后一个 `_v`，避免与 op/dtype 名中的 `_v` 冲突。
pub fn parse_kernel_symbol_abi(symbol: &str) -> Result<u32, AbiError> {
    let idx = symbol
        .rfind("_v")
        .ok_or_else(|| AbiError::InvalidKernelSymbol(symbol.to_string()))?;
    let ver_str = &symbol[idx + 2..];
    ver_str
        .parse::<u32>()
        .map_err(|_| AbiError::InvalidKernelSymbol(symbol.to_string()))
}

/// 校验某个 kernel 期望的 ABI 版本是否与运行时兼容。
///
/// 兼容区间：`[MIN_SUPPORTED_ABI_VERSION, ABI_VERSION]`。
/// Phase 6 加载 kernel 前会调用此函数。
pub fn check_kernel_abi(kernel_abi: u32) -> Result<(), AbiError> {
    if kernel_abi < MIN_SUPPORTED_ABI_VERSION {
        return Err(AbiError::KernelAbiTooOld {
            kernel_abi,
            min_supported: MIN_SUPPORTED_ABI_VERSION,
        });
    }
    if kernel_abi > ABI_VERSION {
        return Err(AbiError::KernelAbiTooNew {
            kernel_abi,
            runtime_abi: ABI_VERSION,
        });
    }
    Ok(())
}

/// mcc 版本兼容性矩阵：返回某个 mcc 版本所支持的最高 musapy ABI 版本。
///
/// 当前矩阵（随 mcc 版本演进更新）：
///   - mcc >= 2.0 → ABI v2
///   - mcc >= 1.0 → ABI v1
///   - mcc  < 1.0 → 不支持（返回 0）
///
/// 未来 mcc 破坏性变更时在此函数追加分支。
pub fn mcc_max_supported_abi(mcc: &SemVer) -> u32 {
    if mcc.major >= 2 {
        2
    } else if mcc.major >= 1 {
        1
    } else {
        0
    }
}

/// 启动期校验：mcc 版本与运行时 ABI 的兼容性。
///
/// - mock 模式：跳过校验（无真实 mcc）
/// - mcc 版本为空（探测失败）：跳过，留给链接期暴露问题
/// - mcc 版本可用：检查其支持的最高 ABI 是否 >= 运行时 ABI
pub fn check_mcc_compatibility() -> Result<(), AbiError> {
    // mock 模式下没有真实 mcc，直接跳过
    if cfg!(musapy_mock_musa) {
        return Ok(());
    }

    let raw = MCC_VERSION_RAW.trim();
    if raw.is_empty() {
        // build.rs 探测失败时为空，这里不 fatal，
        // 真正的链接错误会在 Phase 6 加载 kernel 时暴露。
        return Ok(());
    }

    let mcc = SemVer::parse(raw)?;
    let max_abi = mcc_max_supported_abi(&mcc);
    if max_abi < ABI_VERSION {
        return Err(AbiError::MccIncompatible {
            mcc,
            runtime_abi: ABI_VERSION,
            max_abi_supported: max_abi,
        });
    }
    Ok(())
}

// ============================================================
// 5. 启动期一次性校验
// ============================================================

/// 启动期 ABI 校验报告。
///
/// 由 `run_startup_checks()` 返回，供上层（PyO3）实现 `ms.device_summary()` 等调试输出。
#[derive(Debug, Clone)]
pub struct StartupReport {
    /// 运行时 ABI 版本
    pub abi_version: u32,
    /// 解析后的 mcc 版本（mock 模式或探测失败时为 None）
    pub mcc_version: Option<SemVer>,
    /// 是否处于 mock 模式
    pub mock_mode: bool,
}

impl fmt::Display for StartupReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "musapy ABI v{}", self.abi_version)?;
        match &self.mcc_version {
            Some(v) => write!(f, ", mcc {}", v)?,
            None => write!(f, ", mcc unknown")?,
        }
        if self.mock_mode {
            write!(f, " [MOCK]")?;
        }
        Ok(())
    }
}

/// 执行所有启动期 ABI 校验，返回报告。
///
/// 应在 musapy 首次被 Python import 时调用一次（Phase 5 的 PyO3 module init 里）。
pub fn run_startup_checks() -> Result<StartupReport, AbiError> {
    check_mcc_compatibility()?;

    let mcc_version = if MCC_VERSION_RAW.trim().is_empty() {
        None
    } else {
        SemVer::parse(MCC_VERSION_RAW.trim()).ok()
    };

    Ok(StartupReport {
        abi_version: ABI_VERSION,
        mcc_version,
        mock_mode: cfg!(musapy_mock_musa),
    })
}

// ============================================================
// 6. 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_positive() {
        assert!(ABI_VERSION >= 1, "ABI_VERSION should be >= 1");
    }

    #[test]
    fn musapy_abi_version_matches_const() {
        assert_eq!(musapy_abi_version(), ABI_VERSION);
    }

    #[test]
    fn kernel_symbol_format() {
        let sym = kernel_symbol("add", "f32");
        assert_eq!(sym, format!("musapy_add_f32_v{}", ABI_VERSION));
    }

    #[test]
    fn parse_kernel_symbol_abi_roundtrip() {
        let sym = kernel_symbol("mul", "f64");
        let parsed = parse_kernel_symbol_abi(&sym).unwrap();
        assert_eq!(parsed, ABI_VERSION);
    }

    #[test]
    fn parse_kernel_symbol_abi_with_v_in_op_name() {
        // op 名含 _v 不应干扰解析（rfind 定位最后一个 _v）
        let sym = "musapy_vadd_f32_v1";
        assert_eq!(parse_kernel_symbol_abi(sym).unwrap(), 1);
    }

    #[test]
    fn parse_kernel_symbol_abi_invalid() {
        assert!(parse_kernel_symbol_abi("musapy_add_f32").is_err());
        assert!(parse_kernel_symbol_abi("musapy_add_f32_vx").is_err());
    }

    #[test]
    fn check_kernel_abi_current_version_ok() {
        assert!(check_kernel_abi(ABI_VERSION).is_ok());
    }

    #[test]
    fn check_kernel_abi_too_new() {
        let err = check_kernel_abi(ABI_VERSION + 1).unwrap_err();
        assert!(matches!(
            err,
            AbiError::KernelAbiTooNew {
                runtime_abi: ABI_VERSION,
                ..
            }
        ));
    }

    #[test]
    fn semver_parse_with_prefix() {
        let v = SemVer::parse("mcc 1.2.3").unwrap();
        assert_eq!(v, SemVer { major: 1, minor: 2, patch: 3 });
    }

    #[test]
    fn semver_parse_partial() {
        assert_eq!(
            SemVer::parse("1").unwrap(),
            SemVer { major: 1, minor: 0, patch: 0 }
        );
        assert_eq!(
            SemVer::parse("2.0").unwrap(),
            SemVer { major: 2, minor: 0, patch: 0 }
        );
    }

    #[test]
    fn semver_parse_empty_fails() {
        assert!(SemVer::parse("no version here").is_err());
    }

    #[test]
    fn mcc_matrix() {
        assert_eq!(
            mcc_max_supported_abi(&SemVer { major: 1, minor: 0, patch: 0 }),
            1
        );
        assert_eq!(
            mcc_max_supported_abi(&SemVer { major: 1, minor: 9, patch: 9 }),
            1
        );
        assert_eq!(
            mcc_max_supported_abi(&SemVer { major: 2, minor: 0, patch: 0 }),
            2
        );
        assert_eq!(
            mcc_max_supported_abi(&SemVer { major: 0, minor: 9, patch: 0 }),
            0
        );
    }

    #[test]
    fn startup_checks_runs() {
        // 不论 mock 与否，都应返回 report（mock 模式跳过 mcc 校验）
        let report = run_startup_checks().unwrap();
        assert_eq!(report.abi_version, ABI_VERSION);
        println!("startup report: {}", report);
    }
}