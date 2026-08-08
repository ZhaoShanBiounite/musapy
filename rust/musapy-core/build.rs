// musapy-core build script (P1.7)
//
// 职责：编译期探测 MUSA SDK 位置，向 rustc 注入链接信息。
// 约束（L2-1）：禁止硬编码路径、禁止运行时逻辑、双探针。
//
// 探测策略（按优先级）：
//   1. 环境变量（MUSA_INSTALL_PATH 优先，MUSA_HOME 兼容）
//   2. pkg-config musa-runtime（MUSA SDK 若提供 .pc 文件时生效）
//   3. 降级 mock 模式（CI/无 GPU 开发机）
//
// 版本采集（P1.8）：
//   - MUSA Runtime 版本：从 include/musart_version.h 读取 __MUSA_API_VER_* 宏。
//     这是 kernel ABI 兼容性矩阵的正确数据源（对标 cudart_version.h）。
//   - mcc/clang 版本：从 mcc --version 读取，仅作调试显示，不参与 ABI 判断。
//     mcc 基于 clang，其版本号不直接反映 MUSA SDK 版本。
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// SDK 根目录必须包含的子目录，用于校验探测结果是否"长得像"MUSA SDK
const REQUIRED_SUBDIRS: &[&str] = &["include", "lib"];

/// ABI 版本（P1.8）。每次 kernel ABI 破坏性变更时 +1。
/// build.rs 只负责把它注入编译期环境，运行期校验在 abi.rs 里做。
const ABI_VERSION: u32 = 1;

fn main() {
    // 声明自定义 cfg，让 rustc 的 check-cfg 机制认识它，消除 unexpected_cfgs 警告
    println!("cargo::rustc-check-cfg=cfg(musapy_mock_musa)");
    // -------- 1. 双探针找 MUSA SDK --------
    let musa_home = probe_musa_home();

    match musa_home {
        Some(home) => {
            emit_link_config(&home);
            // 采集 MUSA Runtime 版本（ABI 兼容性矩阵数据源）
            probe_musart_version(&home);
            // 采集 mcc/clang 版本（仅调试显示）
            probe_mcc_version(&home);
            println!("cargo:rustc-env=MUSAPY_MUSA_HOME={}", home.display());
        }
        None => {
            handle_missing_sdk();
        }
    }

    // -------- 2. 注入 ABI 版本 --------
    println!("cargo:rustc-env=MUSAPY_ABI_VERSION={}", ABI_VERSION);

    // -------- 3. 声明 rerun 触发条件 --------
    println!("cargo:rerun-if-env-changed=MUSA_INSTALL_PATH");
    println!("cargo:rerun-if-env-changed=MUSA_HOME");
    println!("cargo:rerun-if-env-changed=MUSAPY_MOCK_MUSA");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
}

/// 双探针主入口
fn probe_musa_home() -> Option<PathBuf> {
    if let Some(home) = probe_via_env() {
        return Some(home);
    }
    if let Some(home) = probe_via_pkgconfig() {
        return Some(home);
    }
    None
}

/// 探针 1：环境变量（MUSA_INSTALL_PATH > MUSA_HOME）
fn probe_via_env() -> Option<PathBuf> {
    const CANDIDATES: &[&str] = &["MUSA_INSTALL_PATH", "MUSA_HOME"];

    for var in CANDIDATES {
        if let Ok(raw) = env::var(var) {
            let path = PathBuf::from(&raw);
            if validate(&path) {
                println!("cargo:warning=MUSAPY: MUSA SDK found via {}={}", var, raw);
                return Some(path);
            } else {
                println!(
                    "cargo:warning=MUSAPY: {}={} set but invalid \
                     (missing include/ or lib/), trying next candidate",
                    var, raw
                );
            }
        }
    }
    None
}

/// 探针 2：pkg-config 查询 musa-runtime（MUSA SDK 目前不发 .pc，预期失败）
fn probe_via_pkgconfig() -> Option<PathBuf> {
    let probe = pkg_config::Config::new()
        .cargo_metadata(false)
        .statik(false)
        .probe("musa-runtime");

    match probe {
        Ok(lib) => {
            // 合并嵌套 if（clippy collapsible_if 门禁）
            let home = lib
                .include_paths
                .first()
                .and_then(|inc| inc.parent())
                .map(|p| p.to_path_buf());
            if let Some(home) = home.filter(|h| validate(h)) {
                println!(
                    "cargo:warning=MUSAPY: MUSA SDK found via pkg-config, home={}",
                    home.display()
                );
                return Some(home);
            }
            None
        }
        Err(_) => None,
    }
}

/// 校验路径是否"长得像"MUSA SDK（同时有 include/ 和 lib/）
fn validate(p: &Path) -> bool {
    REQUIRED_SUBDIRS.iter().all(|sub| p.join(sub).is_dir())
}

/// 把 include / lib 路径和链接库告诉 rustc
/// 把 include / lib 路径和链接库告诉 rustc
fn emit_link_config(home: &Path) {
    let include_dir = home.join("include");
    let lib_dir = home.join("lib");

    println!("cargo:include={}", include_dir.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // 链接 libmusart.so（MUSA Runtime，对标 libcudart.so）
    // 注意 L2-3：musapy-core 只允许依赖 MUSA runtime API，
    // 不链接 mublas/mudnn/mccl（那些是 v0.3+ 的事）
    println!("cargo:rustc-link-lib=dylib=musart");

    // 运行期也需要知道 lib 路径（dlopen 或调试时用）
    println!("cargo:rustc-env=MUSAPY_MUSA_LIB_DIR={}", lib_dir.display());
}

/// 从 include/musart_version.h 读取 MUSA Runtime API 版本。
///
/// 头文件格式（对标 cudart_version.h）：
///   #define __MUSA_API_VER_MAJOR__ 3
///   #define __MUSA_API_VER_MINOR__ 1
///   #define __MUSA_API_VER_PATCH__ 0
///   #define MUSART_VERSION ((MAJOR*10000) + (MINOR*100) + PATCH)
///
/// 注入：
///   MUSAPY_MUSART_VERSION   = 编码整数（如 "30100"），供 abi.rs 兼容性矩阵
///   MUSAPY_MUSA_API_VERSION = "3.1.0" 格式字符串，供调试显示
///
/// 读取失败时不注入，运行期按"未知"处理（跳过校验）。
fn probe_musart_version(home: &Path) {
    let header = home.join("include").join("musart_version.h");
    if !header.exists() {
        println!(
            "cargo:warning=MUSAPY: {} not found, skip musart version probe",
            header.display()
        );
        return;
    }

    let src = match std::fs::read_to_string(&header) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "cargo:warning=MUSAPY: read {} failed: {}, skip musart version probe",
                header.display(),
                e
            );
            return;
        }
    };

    let major = parse_define(&src, "__MUSA_API_VER_MAJOR__");
    let minor = parse_define(&src, "__MUSA_API_VER_MINOR__");
    let patch = parse_define(&src, "__MUSA_API_VER_PATCH__");

    match (major, minor, patch) {
        (Some(ma), Some(mi), Some(pa)) => {
            let encoded = ma * 10000 + mi * 100 + pa;
            let pretty = format!("{}.{}.{}", ma, mi, pa);
            println!("cargo:rustc-env=MUSAPY_MUSART_VERSION={}", encoded);
            println!("cargo:rustc-env=MUSAPY_MUSA_API_VERSION={}", pretty);
            println!(
                "cargo:warning=MUSAPY: detected MUSA Runtime API version: {} ({})",
                pretty, encoded
            );
        }
        _ => {
            println!(
                "cargo:warning=MUSAPY: musart_version.h found but __MUSA_API_VER_* macros missing, version unknown"
            );
        }
    }
}

/// 从 C 头文件文本中提取 `#define NAME 数字` 的值。
///
/// 简单行扫描，不依赖 regex crate。
fn parse_define(src: &str, name: &str) -> Option<u32> {
    for line in src.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("#define") {
            continue;
        }
        let rest = trimmed["#define".len()..].trim_start();
        if !rest.starts_with(name) {
            continue;
        }
        let after_name = &rest[name.len()..];
        let digits: String = after_name
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if digits.is_empty() {
            return None;
        }
        return digits.parse().ok();
    }
    None
}

/// 采集 mcc/clang 版本号（仅调试显示，不参与 ABI 兼容性判断）。
///
/// mcc 基于 clang，--version 输出 clang 版本，不反映 MUSA SDK 版本。
/// ABI 判断使用 MUSA Runtime 版本（见 probe_musart_version）。
fn probe_mcc_version(home: &Path) {
    let mcc = home.join("bin").join("mcc");
    if !mcc.exists() {
        println!(
            "cargo:warning=MUSAPY: mcc not found at {}, skip version probe",
            mcc.display()
        );
        return;
    }
    let output = Command::new(&mcc).arg("--version").output();
    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let ver = if stdout.trim().is_empty() {
                stderr.trim()
            } else {
                stdout.trim()
            };
            if !ver.is_empty() {
                println!("cargo:rustc-env=MUSAPY_MCC_VERSION={}", ver);
                println!("cargo:warning=MUSAPY: detected mcc/clang: {}", ver);
            } else {
                println!("cargo:warning=MUSAPY: `mcc --version` returned empty, version unknown");
            }
        }
        _ => {
            println!("cargo:warning=MUSAPY: `mcc --version` failed, version unknown");
        }
    }
}

/// SDK 没找到时的降级处理
fn handle_missing_sdk() {
    if env::var("MUSAPY_MOCK_MUSA").is_ok() {
        println!("cargo:warning=MUSAPY: MUSA SDK not found, building in MOCK mode");
        println!("cargo:warning=MUSAPY: all MUSA calls will hit mock stubs (dev/CI only)");
        println!("cargo:rustc-cfg=musapy_mock_musa");
    } else {
        panic!(
            "\n\
             ╔════════════════════════════════════════════════════════════╗\n\
             ║  MUSAPY: MUSA SDK not found                                ║\n\
             ╠════════════════════════════════════════════════════════════╣\n\
             ║  To fix this, do ONE of the following:                     ║\n\
             ║                                                            ║\n\
             ║  1. Set MUSA_INSTALL_PATH (MUSA SDK official):             ║\n\
             ║       export MUSA_INSTALL_PATH=/path/to/musa-sdk           ║\n\
             ║     or use MUSA_HOME (CUDA-style alias):                   ║\n\
             ║       export MUSA_HOME=/path/to/musa-sdk                   ║\n\
             ║                                                            ║\n\
             ║  2. Ensure pkg-config can find musa-runtime:               ║\n\
             ║       pkg-config --modversion musa-runtime                 ║\n\
             ║     (Note: MUSA SDK currently does not ship .pc files,     ║\n\
             ║      so option 1 is the recommended approach.)             ║\n\
             ║                                                            ║\n\
             ║  3. Build without MUSA (CI/dev only):                      ║\n\
             ║       export MUSAPY_MOCK_MUSA=1                            ║\n\
             ╚════════════════════════════════════════════════════════════╝\n"
        );
    }
}
