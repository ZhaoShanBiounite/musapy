// musapy-ops build script (P6.2)
//
// 职责：用 mcc 编译 MUSA kernel（.mu → .o → libmusapy_kernels.a），
// 向 rustc 注入静态库链接信息。
//
// 约束（ADR L2-1）：禁止硬编码路径、禁止运行时逻辑、双探针。
//
// 探测策略（与 musapy-core 一致）：
//   1. 环境变量（MUSA_INSTALL_PATH 优先，MUSA_HOME 兼容）
//   2. pkg-config musa-runtime（MUSA SDK 若提供 .pc 文件时生效）
//   3. 降级 mock 模式（CI/无 GPU 开发机）
//
// 真实模式：mcc 编译 kernels/*.mu → libmusapy_kernels.a
// Mock 模式：跳过编译，kernels.rs 提供 Rust stub

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// SDK 根目录必须包含的子目录，用于校验探测结果
const REQUIRED_SUBDIRS: &[&str] = &["include", "lib"];

/// Block size（与 kernel 中的 256 一致，仅用于文档一致性校验）
const BLOCK_SIZE: usize = 256;

fn main() {
    // 声明自定义 cfg，消除 unexpected_cfgs 警告
    println!("cargo::rustc-check-cfg=cfg(musapy_mock_musa)");

    // kernels 源文件路径（相对 workspace root）
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_dir = Path::new(&manifest_dir)
        .join("..")
        .join("..")
        .join("kernels");
    let kernels_dir = kernels_dir.canonicalize().unwrap_or(kernels_dir);

    let elementwise_mu = kernels_dir.join("elementwise.mu");
    let reduction_mu = kernels_dir.join("reduction.mu");

    // -------- 1. 双探针找 MUSA SDK --------
    let musa_home = probe_musa_home();

    match &musa_home {
        Some(home) => {
            // -------- 2. 编译 kernels --------
            compile_kernels(&kernels_dir, &elementwise_mu, &reduction_mu, home);

            // 声明 rerun 触发条件
            println!("cargo:rerun-if-changed={}", elementwise_mu.display());
            println!("cargo:rerun-if-changed={}", reduction_mu.display());
            let common_h = kernels_dir.join("include").join("common.h");
            println!("cargo:rerun-if-changed={}", common_h.display());
        }
        None => {
            handle_missing_sdk();
        }
    }

    // -------- 3. 环境变量 rerun --------
    println!("cargo:rerun-if-env-changed=MUSA_INSTALL_PATH");
    println!("cargo:rerun-if-env-changed=MUSA_HOME");
    println!("cargo:rerun-if-env-changed=MUSAPY_MOCK_MUSA");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    // 消除未使用警告
    let _ = BLOCK_SIZE;
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
                println!(
                    "cargo:warning=MUSAPY-OPS: MUSA SDK found via {}={}",
                    var, raw
                );
                return Some(path);
            } else {
                println!(
                    "cargo:warning=MUSAPY-OPS: {}={} set but invalid \
                     (missing include/ or lib/), trying next candidate",
                    var, raw
                );
            }
        }
    }
    None
}

/// 探针 2：pkg-config 查询 musa-runtime
fn probe_via_pkgconfig() -> Option<PathBuf> {
    let probe = pkg_config::Config::new()
        .cargo_metadata(false)
        .statik(false)
        .probe("musa-runtime");

    match probe {
        Ok(lib) => {
            if let Some(inc) = lib.include_paths.first() {
                if let Some(home) = inc.parent() {
                    let home = home.to_path_buf();
                    if validate(&home) {
                        println!(
                            "cargo:warning=MUSAPY-OPS: MUSA SDK found via pkg-config, home={}",
                            home.display()
                        );
                        return Some(home);
                    }
                }
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

/// 用 mcc 编译 kernel 源文件 → 静态库
fn compile_kernels(kernels_dir: &Path, elementwise_mu: &Path, reduction_mu: &Path, home: &Path) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    let mcc = home.join("bin").join("mcc");
    let include_dir = home.join("include");
    let elementwise_obj = out_dir.join("elementwise.o");
    let reduction_obj = out_dir.join("reduction.o");
    let lib = out_dir.join("libmusapy_kernels.a");

    // -------- 1. mcc 编译 .mu → .o --------
    if !mcc.exists() {
        panic!(
            "MUSAPY-OPS: mcc not found at {}. \
             Cannot compile MUSA kernels. \
             Set MUSAPY_MOCK_MUSA=1 for mock mode.",
            mcc.display()
        );
    }

    // 编译 elementwise.mu
    println!(
        "cargo:warning=MUSAPY-OPS: compiling {} with mcc",
        elementwise_mu.display()
    );

    let status = Command::new(&mcc)
        .arg("-c")
        .arg("-fPIC") // 共享库（.so）需要位置无关代码
        .arg(elementwise_mu)
        .arg("-I")
        .arg(kernels_dir) // 让 #include "include/common.h" 能找到
        .arg("-I")
        .arg(&include_dir) // musa_runtime.h 等 SDK 头文件
        .arg("-o")
        .arg(&elementwise_obj)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!(
                "cargo:warning=MUSAPY-OPS: mcc compiled elementwise.mu → {}",
                elementwise_obj.display()
            );
        }
        Ok(s) => {
            panic!(
                "MUSAPY-OPS: mcc compilation of elementwise.mu failed with exit code {:?}",
                s.code()
            );
        }
        Err(e) => {
            panic!("MUSAPY-OPS: failed to execute mcc: {}", e);
        }
    }

    // 编译 reduction.mu
    println!(
        "cargo:warning=MUSAPY-OPS: compiling {} with mcc",
        reduction_mu.display()
    );

    let status = Command::new(&mcc)
        .arg("-c")
        .arg("-fPIC")
        .arg(reduction_mu)
        .arg("-I")
        .arg(kernels_dir)
        .arg("-I")
        .arg(&include_dir)
        .arg("-o")
        .arg(&reduction_obj)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!(
                "cargo:warning=MUSAPY-OPS: mcc compiled reduction.mu → {}",
                reduction_obj.display()
            );
        }
        Ok(s) => {
            panic!(
                "MUSAPY-OPS: mcc compilation of reduction.mu failed with exit code {:?}",
                s.code()
            );
        }
        Err(e) => {
            panic!("MUSAPY-OPS: failed to execute mcc: {}", e);
        }
    }

    // -------- 2. ar 打包 .o → .a --------
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .arg(&elementwise_obj)
        .arg(&reduction_obj)
        .status();

    match ar_status {
        Ok(s) if s.success() => {
            println!(
                "cargo:warning=MUSAPY-OPS: created static lib {}",
                lib.display()
            );
        }
        Ok(s) => {
            panic!("MUSAPY-OPS: ar failed with exit code {:?}", s.code());
        }
        Err(e) => {
            panic!("MUSAPY-OPS: failed to execute ar: {}", e);
        }
    }

    // -------- 3. 链接配置 --------
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=musapy_kernels");
    // kernel 代码引用 musart 符号（musaStream_t 等），也需要链接
    let lib_dir = home.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=musart");
}

/// SDK 没找到时的降级处理
fn handle_missing_sdk() {
    if env::var("MUSAPY_MOCK_MUSA").is_ok() {
        println!("cargo:warning=MUSAPY-OPS: MUSA SDK not found, building in MOCK mode");
        println!("cargo:warning=MUSAPY-OPS: kernel FFI will use Rust stubs (dev/CI only)");
        println!("cargo:rustc-cfg=musapy_mock_musa");
    } else {
        panic!(
            "\n\
             ╔════════════════════════════════════════════════════════════╗\n\
             ║  MUSAPY-OPS: MUSA SDK not found                           ║\n\
             ╠════════════════════════════════════════════════════════════╣\n\
             ║  Cannot compile MUSA kernels without mcc.                 ║\n\
             ║                                                            ║\n\
             ║  To fix this, do ONE of the following:                     ║\n\
             ║                                                            ║\n\
             ║  1. Set MUSA_INSTALL_PATH (MUSA SDK official):             ║\n\
             ║       export MUSA_INSTALL_PATH=/path/to/musa-sdk           ║\n\
             ║     or use MUSA_HOME (CUDA-style alias):                   ║\n\
             ║       export MUSA_HOME=/path/to/musa-sdk                   ║\n\
             ║                                                            ║\n\
             ║  2. Build without MUSA (CI/dev only):                      ║\n\
             ║       export MUSAPY_MOCK_MUSA=1                            ║\n\
             ╚════════════════════════════════════════════════════════════╝\n"
        );
    }
}
