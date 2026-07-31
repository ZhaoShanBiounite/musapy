// musapy ABI 版本管理与兼容性校验 (P1.8)
//
// 职责：
//   1. 暴露编译期嵌入的 ABI 版本常量（由 build.rs 注入）
//   2. 生成 kernel 符号名：musapy_<op>_<dtype>_v<ABI>
//   3. 运行时校验：kernel 期望的 ABI 版本 vs 运行时 ABI 版本
//   4. 运行时校验：MUSA Runtime 版本 vs 运行时 ABI 版本的兼容性矩阵
//
// 设计依据：ADR L2-1（Build System）
//   - ABI 版本嵌入符号名：musapy_mul_f32_v1
//   - 运行时检查 kernel ABI
//   - MUSA Runtime 版本与 musapy ABI 版本兼容性矩阵检查
//
// 注意：兼容性矩阵基于 MUSA Runtime API 版本（MUSART_VERSION，来自
// musart_version.h），而非 mcc/clang 版本。mcc 基于 clang，其版本号
// 不直接反映 MUSA SDK 版本，仅作调试显示用途。

use std::fmt;

// ============================================================
// 1. 版本常量
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

/// MUSA Runtime API 版本（编码整数，来自 musart_version.h）。
///
/// 编码规则：MAJOR*10000 + MINOR*100 + PATCH（对标 CUDART_VERSION）。
/// 例如 musart 3.1.0 → 30100。
///
/// 由 build.rs 从 include/musart_version.h 解析后注入。
/// 探测失败或 mock 模式时为 0（表示"未知"，启动期跳过兼容性校验）。
pub const MUSART_VERSION: u32 = {
    let raw = match option_env!("MUSAPY_MUSART_VERSION") {
        Some(v) => v,
        None => "",
    };
    let bytes = raw.as_bytes();
    let mut n: u32 = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        // 防御：若注入了非数字字符（不应发生），编译期 panic 而非静默回绕
        if !b.is_ascii_digit() {
            panic!("MUSAPY_MUSART_VERSION contains non-digit character");
        }
        n = n * 10 + (b - b'0') as u32;
        i += 1;
    }
    n
};

/// MUSA Runtime API 版本字符串（如 "3.1.0"），由 build.rs 注入，供调试显示。
///
/// mock 模式或探测失败时为空字符串。
pub const MUSA_API_VERSION_RAW: &str = match option_env!("MUSAPY_MUSA_API_VERSION") {
    Some(v) => v,
    None => "",
};

/// mcc/clang 版本号原始字符串（编译期由 build.rs 探测注入）。
///
/// 注意：mcc 基于 clang，这是 clang 的版本号，不是 MUSA SDK 版本。
/// 仅作调试显示，不参与 ABI 兼容性判断。ABI 判断使用 MUSART_VERSION。
///
/// mock 模式或探测失败时为空字符串。
pub const MCC_VERSION_RAW: &str = match option_env!("MUSAPY_MCC_VERSION") {
    Some(v) => v,
    None => "",
};

// ============================================================
// 2. Error 类型
// ============================================================

/// ABI 校验相关的错误。
#[derive(Debug)]
pub enum AbiError {
    /// kernel 期望的 ABI 版本低于运行时最低支持版本
    KernelAbiTooOld { kernel_abi: u32, min_supported: u32 },
    /// kernel 期望的 ABI 版本高于运行时当前版本（运行时需升级）
    KernelAbiTooNew { kernel_abi: u32, runtime_abi: u32 },
    /// 无法从 kernel 符号名中解析出版本后缀
    InvalidKernelSymbol(String),
    /// MUSA Runtime 版本不支持当前运行时 ABI 版本
    MusartIncompatible {
        musart_version: u32,
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
            AbiError::MusartIncompatible {
                musart_version,
                runtime_abi,
                max_abi_supported,
            } => write!(
                f,
                "MUSA Runtime {} supports ABI up to v{}, but musapy runtime is v{} (upgrade MUSA SDK)",
                format_musart_version(*musart_version),
                max_abi_supported,
                runtime_abi
            ),
        }
    }
}

impl std::error::Error for AbiError {}

// ============================================================
// 3. 公开 API
// ============================================================

/// 返回当前运行时编译期嵌入的 ABI 版本。
pub fn musapy_abi_version() -> u32 {
    ABI_VERSION
}

/// 生成 kernel 符号名：`musapy_<op>_<dtype>_v<ABI>`。
///
/// 例如 `kernel_symbol("add", "f32")` → `"musapy_add_f32_v1"`。
///
/// 所有 kernel 符号必须经此函数生成，确保版本后缀统一。
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

/// MUSA Runtime 版本兼容性矩阵：返回某个 musart 版本所支持的最高 musapy ABI 版本。
///
/// 输入是编码后的 MUSART_VERSION（MAJOR*10000 + MINOR*100 + PATCH）。
///
/// 当前矩阵（随 MUSA SDK 演进更新）：
///   - musart >= 1.0 (>= 10000) → ABI v1（当前唯一版本，已在 musart 3.1.0 上验证）
///   - musart <  1.0            → 不支持（返回 0）
///
/// 未来引入 ABI v2 时，在此函数追加分支，gate 在支持 v2 特性的最低 musart 版本上
/// （该版本需在真实硬件上测试确认）。
pub fn musart_max_supported_abi(musart: u32) -> u32 {
    if musart >= 10000 { 1 } else { 0 }
}

/// 启动期校验：MUSA Runtime 版本与运行时 ABI 的兼容性。
///
/// - mock 模式：跳过校验（无真实 MUSA）
/// - musart 版本未知（探测失败，MUSART_VERSION == 0）：跳过，留给链接期暴露问题
/// - musart 版本可用：检查其支持的最高 ABI 是否 >= 运行时 ABI
pub fn check_musart_compatibility() -> Result<(), AbiError> {
    if cfg!(musapy_mock_musa) {
        return Ok(());
    }

    if MUSART_VERSION == 0 {
        // build.rs 探测失败时为 0，这里不 fatal，
        // 真正的链接错误会在 Phase 6 加载 kernel 时暴露。
        return Ok(());
    }

    let max_abi = musart_max_supported_abi(MUSART_VERSION);
    if max_abi < ABI_VERSION {
        return Err(AbiError::MusartIncompatible {
            musart_version: MUSART_VERSION,
            runtime_abi: ABI_VERSION,
            max_abi_supported: max_abi,
        });
    }
    Ok(())
}

/// 把编码的 musart 版本整数格式化为 "MAJOR.MINOR.PATCH" 字符串。
fn format_musart_version(encoded: u32) -> String {
    let major = encoded / 10000;
    let minor = (encoded / 100) % 100;
    let patch = encoded % 100;
    format!("{}.{}.{}", major, minor, patch)
}

// ============================================================
// 4. 启动期一次性校验
// ============================================================

/// 启动期 ABI 校验报告。
///
/// 由 `run_startup_checks()` 返回，供上层（PyO3）实现 `ms.device_summary()` 等调试输出。
#[derive(Debug, Clone)]
pub struct StartupReport {
    /// 运行时 ABI 版本
    pub abi_version: u32,
    /// 解析后的 MUSA Runtime 版本（编码整数；mock 或探测失败时为 None）
    pub musart_version: Option<u32>,
    /// mcc/clang 版本原始字符串（仅调试显示；mock 或探测失败时为 None）
    pub mcc_version: Option<String>,
    /// 是否处于 mock 模式
    pub mock_mode: bool,
}

impl fmt::Display for StartupReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "musapy ABI v{}", self.abi_version)?;
        match self.musart_version {
            Some(v) => write!(f, ", MUSA Runtime {}", format_musart_version(v))?,
            None => write!(f, ", MUSA Runtime unknown")?,
        }
        match &self.mcc_version {
            Some(v) => write!(f, ", mcc/clang {}", v)?,
            None => write!(f, ", mcc/clang unknown")?,
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
    check_musart_compatibility()?;

    let musart_version = if MUSART_VERSION == 0 {
        None
    } else {
        Some(MUSART_VERSION)
    };

    let mcc_version = {
        let raw = MCC_VERSION_RAW.trim();
        if raw.is_empty() {
            None
        } else {
            // 只取第一行，避免多行版本信息污染 Display 输出
            Some(raw.lines().next().unwrap_or("").to_string())
        }
    };

    Ok(StartupReport {
        abi_version: ABI_VERSION,
        musart_version,
        mcc_version,
        mock_mode: cfg!(musapy_mock_musa),
    })
}

// ============================================================
// 5. 单元测试
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
    fn musart_matrix() {
        // musart < 1.0 不支持
        assert_eq!(musart_max_supported_abi(0), 0);
        assert_eq!(musart_max_supported_abi(9999), 0);
        // musart >= 1.0 支持 ABI v1
        assert_eq!(musart_max_supported_abi(10000), 1); // 1.0.0
        assert_eq!(musart_max_supported_abi(10300), 1); // 1.3.0（SDK 头文件引用的最低版本）
        assert_eq!(musart_max_supported_abi(30100), 1); // 3.1.0（当前测试环境）
    }

    #[test]
    fn format_musart_version_decodes() {
        assert_eq!(format_musart_version(30100), "3.1.0");
        assert_eq!(format_musart_version(10300), "1.3.0");
        assert_eq!(format_musart_version(10000), "1.0.0");
        assert_eq!(format_musart_version(0), "0.0.0");
    }

    #[test]
    fn startup_checks_runs() {
        // 不论 mock 与否，都应返回 report（mock 模式跳过 musart 校验）
        let report = run_startup_checks().unwrap();
        assert_eq!(report.abi_version, ABI_VERSION);
        println!("startup report: {}", report);
    }
}
