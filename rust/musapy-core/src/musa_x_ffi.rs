//! MUSA-X 数学库 FFI 绑定(v0.3,ADR-003 003-D1/003-D2)
//!
//! 覆盖 muBLAS / muSOLVER / muRAND / muFFT / muSPARSE 五个数学库,沿用
//! `musa_ffi.rs` 的手写 FFI + real/mock 编译期分支模式。
//!
//! **架构边界(003-D1)**:本模块只做**声明**与类型定义,计算调用(gemm/getrf/
//! fft/spmv 等)只在 musapy-ops 发起;MUSA-X 链接指令由 musapy-ops/build.rs
//! 发出,musapy-core 保持 musart-only(L2-3)。extern 声明在 musapy-python
//! cdylib 最终链接期解析。
//!
//! **签名验证**:基于 SDK 3.1.0 头文件级核对(2026-08-06)+ 运行期符号审计
//! (`tools/check_musax_symbols.sh`,83/83 PASS)。关键事实:
//!   - musolver **无独立句柄**(无 musolverDnCreate),例程接收 `mublasHandle_t`、
//!     返回 `mublasStatus_t`(003-D2);
//!   - murand/mufft 的 GetVersion **无句柄参数**(只收 `int*`);
//!   - musparse 的 `MUstream` 与 `musaStream_t` 同为 `struct MUstream_st*`,无需转换。
//!
//! Phase 1 只声明生命周期符号(Create/Destroy/SetStream/GetVersion/Plan*);
//! 计算例程(Sgemm/getrf/ExecC2C/SpMV 等)由 Phase 2-6 按 v0.3 计划附录 A
//! 步骤 1 逐个追加。
//!
//! mock 模式(musapy_mock_musa):提供同签名 Rust stub,返回成功码 + dummy 句柄。

#![allow(non_camel_case_types)]
// mock 分支的 Rust stub 与 extern 声明同名(mublasCreate 等),保留 C 命名。
#![allow(non_snake_case)]
use crate::error::{DeviceError, MusapyError, Result};
use crate::musa_ffi::musaStream_t;
use std::ffi::{c_int, c_void};

// ============================================================
// 句柄类型(4 类;musolver 复用 mublas 句柄,003-D2)
// ============================================================

/// muBLAS 句柄(opaque 指针)。**muSOLVER 例程共用此句柄**(SDK 3.1.0 无独立句柄)。
pub type mublasHandle_t = *mut c_void;

/// muRAND 生成器句柄(opaque 指针)。
pub type murandGenerator_t = *mut c_void;

/// muFFT plan 句柄(opaque 指针)。
pub type mufftHandle = *mut c_void;

/// muSPARSE 句柄(opaque 指针)。
pub type musparseHandle_t = *mut c_void;

// ============================================================
// 状态码(4 类;musolver 共用 mublasStatus)
// ============================================================

/// muBLAS 状态码(muSOLVER 共用)。
pub type mublasStatus_t = c_int;
pub const MUBLAS_STATUS_SUCCESS: mublasStatus_t = 0;

/// muRAND 状态码。
pub type murandStatus_t = c_int;
pub const MURAND_STATUS_SUCCESS: murandStatus_t = 0;

/// muFFT 状态码(头文件 `typedef enum mufftResult_t {...} mufftResult`)。
pub type mufftResult = c_int;
pub const MUFFT_SUCCESS: mufftResult = 0;

/// muSPARSE 状态码。
pub type musparseStatus_t = c_int;
pub const MUSPARSE_STATUS_SUCCESS: musparseStatus_t = 0;

// ============================================================
// 枚举常量(Phase 1 最小集;计算期枚举 Phase 2-6 追加)
// ============================================================

/// mublasOperation_t(mublas-types.h:115-117)。
pub const MUBLAS_OP_N: c_int = 111;
pub const MUBLAS_OP_T: c_int = 112;
pub const MUBLAS_OP_C: c_int = 113;

/// murandRngType_t(murand.h)。
pub type murandRngType_t = c_int;
pub const MURAND_RNG_PSEUDO_DEFAULT: murandRngType_t = 400;
pub const MURAND_RNG_PSEUDO_XORWOW: murandRngType_t = 401;
pub const MURAND_RNG_PSEUDO_MRG32K3A: murandRngType_t = 402;
pub const MURAND_RNG_PSEUDO_MTGP32: murandRngType_t = 403;
pub const MURAND_RNG_PSEUDO_PHILOX4_32_10: murandRngType_t = 404;

/// mufftType(mufft.h:注意非连续值)。
pub type mufftType = c_int;
pub const MUFFT_R2C: mufftType = 0x2a;
pub const MUFFT_C2R: mufftType = 0x2c;
pub const MUFFT_C2C: mufftType = 0x29;
pub const MUFFT_D2Z: mufftType = 0x6a;
pub const MUFFT_Z2D: mufftType = 0x6c;
pub const MUFFT_Z2Z: mufftType = 0x69;

/// FFT 方向(mufft.h:105/108,#define 常量)。
pub const MUFFT_FORWARD: c_int = -1;
pub const MUFFT_INVERSE: c_int = 1;

// ============================================================
// FFI 声明(真实模式)
// ============================================================

#[cfg(not(musapy_mock_musa))]
mod real {
    use super::*;

    unsafe extern "C" {
        // ── muBLAS(mublas-auxiliary.h;生命周期)──
        pub fn mublasCreate(handle: *mut mublasHandle_t) -> mublasStatus_t;
        pub fn mublasDestroy(handle: mublasHandle_t) -> mublasStatus_t;
        pub fn mublasSetStream(handle: mublasHandle_t, stream: musaStream_t) -> mublasStatus_t;
        pub fn mublasGetVersion(handle: mublasHandle_t, version: *mut c_int) -> mublasStatus_t;

        // ── muSOLVER:无生命周期符号(无独立句柄,复用 mublasHandle_t)──
        // 计算例程(getrf/getrs/geqrf/orgqr/cungqr/gesvd + *_bufferSize)
        // 由 Phase 2/3 追加。

        // ── muRAND(murand.h;生命周期)──
        pub fn murandCreateGenerator(
            generator: *mut murandGenerator_t,
            rng_type: murandRngType_t,
        ) -> murandStatus_t;
        pub fn murandDestroyGenerator(generator: murandGenerator_t) -> murandStatus_t;
        pub fn murandSetStream(
            generator: murandGenerator_t,
            stream: musaStream_t,
        ) -> murandStatus_t;
        // 注意:无句柄参数(murand.h:527)
        pub fn murandGetVersion(version: *mut c_int) -> murandStatus_t;

        // ── muFFT(mufft.h;生命周期 + plan 创建)──
        pub fn mufftCreate(plan: *mut mufftHandle) -> mufftResult;
        pub fn mufftDestroy(plan: mufftHandle) -> mufftResult;
        pub fn mufftSetStream(plan: mufftHandle, stream: musaStream_t) -> mufftResult;
        // 注意:无句柄参数(mufft.h:746)
        pub fn mufftGetVersion(version: *mut c_int) -> mufftResult;
        // batch 参数已弃用(建议 PlanMany),但签名保留(mufft.h:148)
        pub fn mufftPlan1d(
            plan: *mut mufftHandle,
            nx: c_int,
            ftype: mufftType,
            batch: c_int,
        ) -> mufftResult;
        pub fn mufftPlan2d(
            plan: *mut mufftHandle,
            nx: c_int,
            ny: c_int,
            ftype: mufftType,
        ) -> mufftResult;
        pub fn mufftPlan3d(
            plan: *mut mufftHandle,
            nx: c_int,
            ny: c_int,
            nz: c_int,
            ftype: mufftType,
        ) -> mufftResult;
        pub fn mufftPlanMany(
            plan: *mut mufftHandle,
            rank: c_int,
            n: *mut c_int,
            inembed: *mut c_int,
            istride: c_int,
            idist: c_int,
            onembed: *mut c_int,
            ostride: c_int,
            odist: c_int,
            ftype: mufftType,
            batch: c_int,
        ) -> mufftResult;

        // ── muSPARSE(musparse-auxiliary.h;生命周期)──
        pub fn musparseCreate(handle: *mut musparseHandle_t) -> musparseStatus_t;
        pub fn musparseDestroy(handle: musparseHandle_t) -> musparseStatus_t;
        pub fn musparseSetStream(
            handle: musparseHandle_t,
            stream: musaStream_t, // MUstream ≡ musaStream_t,无需转换
        ) -> musparseStatus_t;
        pub fn musparseGetVersion(
            handle: musparseHandle_t,
            version: *mut c_int,
        ) -> musparseStatus_t;
    }
}

// ============================================================
// Mock stub(同签名,musapy_mock_musa)
// ============================================================

#[cfg(musapy_mock_musa)]
mod mock {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// dummy 句柄计数器(mock 下句柄只需非空且可区分)。
    static MOCK_MATH_HANDLE_COUNTER: AtomicUsize = AtomicUsize::new(0x1000_0000);

    fn next_handle() -> *mut c_void {
        MOCK_MATH_HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed) as *mut c_void
    }

    // ── muBLAS ──

    pub unsafe fn mublasCreate(handle: *mut mublasHandle_t) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        unsafe { *handle = next_handle() };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasDestroy(handle: mublasHandle_t) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasSetStream(handle: mublasHandle_t, _stream: musaStream_t) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasGetVersion(handle: mublasHandle_t, version: *mut c_int) -> mublasStatus_t {
        if handle.is_null() || version.is_null() {
            return 1;
        }
        unsafe { *version = 30100 }; // SDK 3.1.0
        MUBLAS_STATUS_SUCCESS
    }

    // ── muRAND ──

    pub unsafe fn murandCreateGenerator(
        generator: *mut murandGenerator_t,
        _rng_type: murandRngType_t,
    ) -> murandStatus_t {
        if generator.is_null() {
            return 1;
        }
        unsafe { *generator = next_handle() };
        MURAND_STATUS_SUCCESS
    }

    pub unsafe fn murandDestroyGenerator(generator: murandGenerator_t) -> murandStatus_t {
        if generator.is_null() {
            return 1;
        }
        MURAND_STATUS_SUCCESS
    }

    pub unsafe fn murandSetStream(generator: murandGenerator_t, _stream: musaStream_t) -> murandStatus_t {
        if generator.is_null() {
            return 1;
        }
        MURAND_STATUS_SUCCESS
    }

    pub unsafe fn murandGetVersion(version: *mut c_int) -> murandStatus_t {
        if version.is_null() {
            return 1;
        }
        unsafe { *version = 30100 };
        MURAND_STATUS_SUCCESS
    }

    // ── muFFT ──

    pub unsafe fn mufftCreate(plan: *mut mufftHandle) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        unsafe { *plan = next_handle() };
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftDestroy(plan: mufftHandle) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftSetStream(plan: mufftHandle, _stream: musaStream_t) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftGetVersion(version: *mut c_int) -> mufftResult {
        if version.is_null() {
            return 1;
        }
        unsafe { *version = 30100 };
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftPlan1d(
        plan: *mut mufftHandle,
        _nx: c_int,
        _ftype: mufftType,
        _batch: c_int,
    ) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        unsafe { *plan = next_handle() };
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftPlan2d(
        plan: *mut mufftHandle,
        _nx: c_int,
        _ny: c_int,
        _ftype: mufftType,
    ) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        unsafe { *plan = next_handle() };
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftPlan3d(
        plan: *mut mufftHandle,
        _nx: c_int,
        _ny: c_int,
        _nz: c_int,
        _ftype: mufftType,
    ) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        unsafe { *plan = next_handle() };
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftPlanMany(
        plan: *mut mufftHandle,
        _rank: c_int,
        _n: *mut c_int,
        _inembed: *mut c_int,
        _istride: c_int,
        _idist: c_int,
        _onembed: *mut c_int,
        _ostride: c_int,
        _odist: c_int,
        _ftype: mufftType,
        _batch: c_int,
    ) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        unsafe { *plan = next_handle() };
        MUFFT_SUCCESS
    }

    // ── muSPARSE ──

    pub unsafe fn musparseCreate(handle: *mut musparseHandle_t) -> musparseStatus_t {
        if handle.is_null() {
            return 1;
        }
        unsafe { *handle = next_handle() };
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseDestroy(handle: musparseHandle_t) -> musparseStatus_t {
        if handle.is_null() {
            return 1;
        }
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseSetStream(handle: musparseHandle_t, _stream: musaStream_t) -> musparseStatus_t {
        if handle.is_null() {
            return 1;
        }
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseGetVersion(handle: musparseHandle_t, version: *mut c_int) -> musparseStatus_t {
        if handle.is_null() || version.is_null() {
            return 1;
        }
        unsafe { *version = 30100 };
        MUSPARSE_STATUS_SUCCESS
    }
}

#[cfg(musapy_mock_musa)]
pub use mock::*;
#[cfg(not(musapy_mock_musa))]
pub use real::*;

// ============================================================
// 高层辅助:状态码 → MusapyError(仿 musa_ffi::check_musa)
// ============================================================

/// muBLAS/muSOLVER 状态检查(003-D2:创建/生命周期失败 → DeviceError)。
pub fn check_mublas(status: mublasStatus_t, context: &str) -> Result<()> {
    if status == MUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(MusapyError::Device(DeviceError::MathLibCallFailed(
            format!("{}: mublas call failed (status {})", context, status),
        )))
    }
}

/// muRAND 状态检查。
pub fn check_murand(status: murandStatus_t, context: &str) -> Result<()> {
    if status == MURAND_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(MusapyError::Device(DeviceError::MathLibCallFailed(
            format!("{}: murand call failed (status {})", context, status),
        )))
    }
}

/// muFFT 状态检查。
pub fn check_mufft(status: mufftResult, context: &str) -> Result<()> {
    if status == MUFFT_SUCCESS {
        Ok(())
    } else {
        Err(MusapyError::Device(DeviceError::MathLibCallFailed(
            format!("{}: mufft call failed (status {})", context, status),
        )))
    }
}

/// muSPARSE 状态检查。
pub fn check_musparse(status: musparseStatus_t, context: &str) -> Result<()> {
    if status == MUSPARSE_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(MusapyError::Device(DeviceError::MathLibCallFailed(
            format!("{}: musparse call failed (status {})", context, status),
        )))
    }
}

// ============================================================
// 测试(仅 mock 模式:真实模式下 musapy-core 不链接 MUSA-X,003-D1)
// ============================================================

#[cfg(all(test, musapy_mock_musa))]
mod tests {
    use super::*;

    #[test]
    fn test_mublas_lifecycle() {
        let mut h: mublasHandle_t = std::ptr::null_mut();
        assert_eq!(unsafe { mublasCreate(&mut h) }, MUBLAS_STATUS_SUCCESS);
        assert!(!h.is_null());
        let mut v: c_int = 0;
        assert_eq!(unsafe { mublasGetVersion(h, &mut v) }, MUBLAS_STATUS_SUCCESS);
        assert_eq!(v, 30100);
        assert_eq!(
            unsafe { mublasSetStream(h, std::ptr::null_mut()) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(unsafe { mublasDestroy(h) }, MUBLAS_STATUS_SUCCESS);
    }

    #[test]
    fn test_mublas_null_rejected() {
        assert_ne!(unsafe { mublasCreate(std::ptr::null_mut()) }, MUBLAS_STATUS_SUCCESS);
        assert_ne!(unsafe { mublasDestroy(std::ptr::null_mut()) }, MUBLAS_STATUS_SUCCESS);
    }

    #[test]
    fn test_murand_lifecycle() {
        let mut g: murandGenerator_t = std::ptr::null_mut();
        assert_eq!(
            unsafe { murandCreateGenerator(&mut g, MURAND_RNG_PSEUDO_DEFAULT) },
            MURAND_STATUS_SUCCESS
        );
        assert!(!g.is_null());
        let mut v: c_int = 0;
        assert_eq!(unsafe { murandGetVersion(&mut v) }, MURAND_STATUS_SUCCESS); // 无句柄参数
        assert_eq!(
            unsafe { murandSetStream(g, std::ptr::null_mut()) },
            MURAND_STATUS_SUCCESS
        );
        assert_eq!(unsafe { murandDestroyGenerator(g) }, MURAND_STATUS_SUCCESS);
    }

    #[test]
    fn test_mufft_lifecycle_and_plans() {
        let mut v: c_int = 0;
        assert_eq!(unsafe { mufftGetVersion(&mut v) }, MUFFT_SUCCESS); // 无句柄参数

        let mut p1: mufftHandle = std::ptr::null_mut();
        assert_eq!(unsafe { mufftPlan1d(&mut p1, 64, MUFFT_C2C, 1) }, MUFFT_SUCCESS);
        assert!(!p1.is_null());
        assert_eq!(
            unsafe { mufftSetStream(p1, std::ptr::null_mut()) },
            MUFFT_SUCCESS
        );
        assert_eq!(unsafe { mufftDestroy(p1) }, MUFFT_SUCCESS);

        let mut p2: mufftHandle = std::ptr::null_mut();
        assert_eq!(unsafe { mufftPlan2d(&mut p2, 8, 8, MUFFT_Z2Z) }, MUFFT_SUCCESS);
        assert_eq!(unsafe { mufftDestroy(p2) }, MUFFT_SUCCESS);

        let mut p3: mufftHandle = std::ptr::null_mut();
        assert_eq!(
            unsafe { mufftPlan3d(&mut p3, 4, 4, 4, MUFFT_C2C) },
            MUFFT_SUCCESS
        );
        assert_eq!(unsafe { mufftDestroy(p3) }, MUFFT_SUCCESS);

        let mut pm: mufftHandle = std::ptr::null_mut();
        let mut n: [c_int; 2] = [8, 8];
        assert_eq!(
            unsafe {
                mufftPlanMany(
                    &mut pm,
                    2,
                    n.as_mut_ptr(),
                    std::ptr::null_mut(),
                    1,
                    64,
                    std::ptr::null_mut(),
                    1,
                    64,
                    MUFFT_C2C,
                    3,
                )
            },
            MUFFT_SUCCESS
        );
        assert_eq!(unsafe { mufftDestroy(pm) }, MUFFT_SUCCESS);
    }

    #[test]
    fn test_musparse_lifecycle() {
        let mut h: musparseHandle_t = std::ptr::null_mut();
        assert_eq!(unsafe { musparseCreate(&mut h) }, MUSPARSE_STATUS_SUCCESS);
        assert!(!h.is_null());
        let mut v: c_int = 0;
        assert_eq!(
            unsafe { musparseGetVersion(h, &mut v) },
            MUSPARSE_STATUS_SUCCESS
        );
        assert_eq!(
            unsafe { musparseSetStream(h, std::ptr::null_mut()) },
            MUSPARSE_STATUS_SUCCESS
        );
        assert_eq!(unsafe { musparseDestroy(h) }, MUSPARSE_STATUS_SUCCESS);
    }

    #[test]
    fn test_check_helpers() {
        assert!(check_mublas(MUBLAS_STATUS_SUCCESS, "t").is_ok());
        assert!(check_mublas(6, "t").is_err());
        assert!(check_murand(MURAND_STATUS_SUCCESS, "t").is_ok());
        assert!(check_murand(102, "t").is_err());
        assert!(check_mufft(MUFFT_SUCCESS, "t").is_ok());
        assert!(check_mufft(2, "t").is_err());
        assert!(check_musparse(MUSPARSE_STATUS_SUCCESS, "t").is_ok());
        assert!(check_musparse(1, "t").is_err());

        // 错误映射到 DeviceError::MathLibCallFailed(003-D2)
        let e = check_mublas(6, "mublasCreate").unwrap_err();
        assert!(matches!(
            e,
            MusapyError::Device(DeviceError::MathLibCallFailed(_))
        ));
    }
}
