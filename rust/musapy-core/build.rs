// musapy-core build script (P1.7)
//
// 职责：编译期探测 MUSA SDK 位置，向 rustc 注入链接信息。
// 约束（L2-1）：禁止硬编码路径、禁止运行时逻辑、双探针。
//
// 探测策略（按优先级）：
//   1. 环境变量（MUSA_INSTALL_PATH 优先，MUSA_HOME 兼容）
//   2. pkg-config musa-runtime（MUSA SDK 若提供 .pc 文件时生效）
//   3. 降级 mock 模式（CI/无 GPU 开发机）
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
            // 找到了，emit 链接配置
            emit_link_config(&home);
            // 采集 mcc 版本（供启动期兼容性矩阵校验）
            probe_mcc_version(&home);
            // 把路径注入环境，供运行期 FFI 模块读取
            println!("cargo:rustc-env=MUSAPY_MUSA_HOME={}", home.display());
          
        }
        None => {
            // 没找到 —— 走 mock 降级或直接失败
            handle_missing_sdk();
        }
    }

    // -------- 2. 注入 ABI 版本（P1.8 的一部分） --------
    println!("cargo:rustc-env=MUSAPY_ABI_VERSION={}", ABI_VERSION);

    // -------- 3. 声明 rerun 触发条件 --------
    // 否则 cargo 只在 build.rs 自身变化时才重跑，环境变量改了不会重新探测
    println!("cargo:rerun-if-env-changed=MUSA_INSTALL_PATH");
    println!("cargo:rerun-if-env-changed=MUSA_HOME");
    println!("cargo:rerun-if-env-changed=MUSAPY_MOCK_MUSA");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
}

/// 双探针主入口
fn probe_musa_home() -> Option<PathBuf> {
    // 探针 1：环境变量（优先级最高，开发者显式指定）
    if let Some(home) = probe_via_env() {
        return Some(home);
    }
    // 探针 2：pkg-config（系统级注册，适合包管理器安装的 SDK）
    if let Some(home) = probe_via_pkgconfig() {
        return Some(home);
    }
    None
}

/// 探针 1：从环境变量读取 MUSA SDK 路径。
///
/// MUSA SDK 的官方约定变量名是 `MUSA_INSTALL_PATH`（摩尔线程文档标准），
/// 但部分文档/教程也会用 `MUSA_HOME`（沿用 CUDA 习惯），所以两个都试。
/// 优先级：MUSA_INSTALL_PATH > MUSA_HOME
fn probe_via_env() -> Option<PathBuf> {
    // 候选变量名，按优先级排列
    const CANDIDATES: &[&str] = &[
        "MUSA_INSTALL_PATH", // MUSA SDK 官方约定（摩尔线程文档标准）
        "MUSA_HOME",         // 兼容 CUDA 习惯命名的用户
    ];

    for var in CANDIDATES {
        if let Ok(raw) = env::var(var) {
            let path = PathBuf::from(&raw);
            if validate(&path) {
                println!(
                    "cargo:warning=MUSAPY: MUSA SDK found via {}={}",
                    var, raw
                );
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

/// 探针 2：pkg-config 查询 musa-runtime。
///
/// 注意：MUSA SDK 目前实际不发布 `musa-runtime.pc` 文件，
/// 所以此探针在真实环境下大概率失败 —— 这是预期行为。
/// 只有当两个探针都失败时，才在最终错误信息里提示 pkg-config 方案。
/// 此处失败时只静默返回 None，不打冗长 warning 避免误导用户。
fn probe_via_pkgconfig() -> Option<PathBuf> {
    // cargo_metadata(false) —— 我们要自己控制 emit 哪些 cargo 指令，
    // 避免 pkg-config crate 自动 emit 的 link 指令和我们手写的冲突
    let probe = pkg_config::Config::new()
        .cargo_metadata(false)
        .statik(false)
        .probe("musa-runtime");

    match probe {
        Ok(lib) => {
            // .pc 文件里的 includedir 通常是 <MUSA_HOME>/include
            // 用它的 parent 反推 MUSA_HOME
            if let Some(inc) = lib.include_paths.first() {
                if let Some(home) = inc.parent() {
                    let home = home.to_path_buf();
                    if validate(&home) {
                        println!(
                            "cargo:warning=MUSAPY: MUSA SDK found via pkg-config, home={}",
                            home.display()
                        );
                        return Some(home);
                    }
                }
            }
            // include_paths 为空时退而求其次：直接用 pkg-config emit 的 link 信息
            // 但这种情况无法反推 home，musapy 不推荐
            None
        }
        Err(_) => {
            // MUSA SDK 目前不发布 .pc 文件，pkg-config 探针失败是预期行为。
            // 不打 warning，避免长串 pkg-config 错误输出误导用户。
            // 只有当环境变量探针也失败时，handle_missing_sdk() 才会提示 pkg-config 方案。
            None
        }
    }
}

/// 校验路径是否"长得像"MUSA SDK（同时有 include/ 和 lib/）
fn validate(p: &Path) -> bool {
    REQUIRED_SUBDIRS.iter().all(|sub| p.join(sub).is_dir())
}

/// 把 include / lib 路径和链接库告诉 rustc
fn emit_link_config(home: &Path) {
    let include_dir = home.join("include");
    let lib_dir = home.join("lib");

    // 如果将来用 bindgen 生成 FFI binding，需要 include 路径
    println!("cargo:include={}", include_dir.display());

    // 告诉 rustc 去哪里找 .so
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // 链接 libmusa_runtime.so
    // 注意 L2-3：musapy-core 只允许依赖 MUSA runtime API，
    // 不链接 mublas/mudnn/mccl（那些是 v0.3+ 的事）
    println!("cargo:rustc-link-lib=dylib=musa_runtime");

    // 运行期也需要知道 lib 路径（dlopen 或调试时用）
    println!("cargo:rustc-env=MUSAPY_MUSA_LIB_DIR={}", lib_dir.display());
}

/// 采集 mcc 版本号（编译期只采集，不校验 —— 矩阵校验在运行期 abi.rs 做）
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
            // mcc 可能输出到 stdout 或 stderr，取非空的那个
            let ver = if stdout.trim().is_empty() {
                stderr.trim()
            } else {
                stdout.trim()
            };
            if !ver.is_empty() {
                println!("cargo:rustc-env=MUSAPY_MCC_VERSION={}", ver);
                println!("cargo:warning=MUSAPY: detected mcc: {}", ver);
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
        // 降级：mock 模式，CI/无 GPU 开发机用
        println!("cargo:warning=MUSAPY: MUSA SDK not found, building in MOCK mode");
        println!("cargo:warning=MUSAPY: all MUSA calls will hit mock stubs (dev/CI only)");
        println!("cargo:rustc-cfg=musapy_mock_musa");
    } else {
        // 真正失败：给出可操作的修复提示
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