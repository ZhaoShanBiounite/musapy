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
//! (`tools/check_musax_symbols.sh`,88/88 PASS)。关键事实:
//!   - musolver **无独立句柄**(无 musolverDnCreate),例程接收 `mublasHandle_t`、
//!     返回 `mublasStatus_t`(003-D2);
//!   - musolver `*_bufferSize` **无句柄参数**;getrf/getrs 的 ipiv/info 为设备指针;
//!   - murand/mufft 的 GetVersion **无句柄参数**(只收 `int*`);
//!   - musparse 的 `MUstream` 与 `musaStream_t` 同为 `struct MUstream_st*`,无需转换;
//!   - `mublas_int` = int32(LP64 默认,未启用 `mublas_ILP64`);
//!   - `muComplex`/`muDoubleComplex` = float2/double2(mublas-types.h ≡ muComplex.h)。
//!
//! Phase 1 声明生命周期符号(Create/Destroy/SetStream/GetVersion/Plan*);
//! Phase 2 追加 gemm/dot/getrf/getrs 计算例程;其余例程(ExecC2C/SpMV 等)
//! 由 Phase 3-6 按 v0.3 计划附录 A 步骤 1 逐个追加。
//!
//! mock 模式(musapy_mock_musa):提供同签名 Rust stub,返回成功码 + dummy 句柄。

#![allow(non_camel_case_types)]
// mock 分支的 Rust stub 与 extern 声明同名(mublasCreate 等),保留 C 命名。
#![allow(non_snake_case)]
// mock stub 是与 extern 同名的测试替身,安全性由调用侧 unsafe 块承担,
// 与 Phase 1 既有 stub 风格一致,不逐函数写 Safety 文档。
#![allow(clippy::missing_safety_doc)]
// FFI 参数个数由 SDK 头文件签名决定(gemm 14 参/getrs 10 参等),不可拆分。
#![allow(clippy::too_many_arguments)]
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
// 标量/复数类型(Phase 2:gemm/dot 的 alpha/beta 与复数矩阵)
// ============================================================

/// muBLAS 整数类型(mublas-types.h:`typedef int32_t mublas_int`,LP64 默认;
/// 仅 `mublas_ILP64` 宏开启时为 int64,本项目不启用)。
pub type mublas_int = c_int;

/// 单精度复数(muComplex.h:`typedef float2 muFloatComplex; typedef muFloatComplex muComplex`)。
/// ABI 与 C 的 `float2 { x, y }` 一致。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct muComplex {
    pub re: f32,
    pub im: f32,
}

/// 双精度复数(muComplex.h:`typedef double2 muDoubleComplex`)。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct muDoubleComplex {
    pub re: f64,
    pub im: f64,
}

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

/// mublasPointerMode_t(mublas-types.h:175-181)。musapy 统一使用 HOST 模式
/// (alpha/beta 为 host 标量,见 math_handle.rs 句柄初始化)。
pub type mublasPointerMode_t = c_int;
pub const MUBLAS_POINTER_MODE_HOST: mublasPointerMode_t = 0;
pub const MUBLAS_POINTER_MODE_DEVICE: mublasPointerMode_t = 1;

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
        pub fn mublasSetPointerMode(
            handle: mublasHandle_t,
            pointer_mode: mublasPointerMode_t,
        ) -> mublasStatus_t;

        // ── muBLAS 计算例程(mublas-functions.h;Phase 2:gemm/dot)──
        // 语义与 cuBLAS 一致:列主序,m/n/k 为列主序矩阵维度;HOST pointer mode
        // 下 alpha/beta 为 host 标量指针。
        pub fn mublasSgemm(
            handle: mublasHandle_t,
            transa: c_int,
            transb: c_int,
            m: mublas_int,
            n: mublas_int,
            k: mublas_int,
            alpha: *const f32,
            a: *const f32,
            lda: mublas_int,
            b: *const f32,
            ldb: mublas_int,
            beta: *const f32,
            c: *mut f32,
            ldc: mublas_int,
        ) -> mublasStatus_t;
        pub fn mublasDgemm(
            handle: mublasHandle_t,
            transa: c_int,
            transb: c_int,
            m: mublas_int,
            n: mublas_int,
            k: mublas_int,
            alpha: *const f64,
            a: *const f64,
            lda: mublas_int,
            b: *const f64,
            ldb: mublas_int,
            beta: *const f64,
            c: *mut f64,
            ldc: mublas_int,
        ) -> mublasStatus_t;
        pub fn mublasCgemm(
            handle: mublasHandle_t,
            transa: c_int,
            transb: c_int,
            m: mublas_int,
            n: mublas_int,
            k: mublas_int,
            alpha: *const muComplex,
            a: *const muComplex,
            lda: mublas_int,
            b: *const muComplex,
            ldb: mublas_int,
            beta: *const muComplex,
            c: *mut muComplex,
            ldc: mublas_int,
        ) -> mublasStatus_t;
        pub fn mublasZgemm(
            handle: mublasHandle_t,
            transa: c_int,
            transb: c_int,
            m: mublas_int,
            n: mublas_int,
            k: mublas_int,
            alpha: *const muDoubleComplex,
            a: *const muDoubleComplex,
            lda: mublas_int,
            b: *const muDoubleComplex,
            ldb: mublas_int,
            beta: *const muDoubleComplex,
            c: *mut muDoubleComplex,
            ldc: mublas_int,
        ) -> mublasStatus_t;
        pub fn mublasSdot(
            handle: mublasHandle_t,
            n: mublas_int,
            x: *const f32,
            incx: mublas_int,
            y: *const f32,
            incy: mublas_int,
            result: *mut f32,
        ) -> mublasStatus_t;
        pub fn mublasDdot(
            handle: mublasHandle_t,
            n: mublas_int,
            x: *const f64,
            incx: mublas_int,
            y: *const f64,
            incy: mublas_int,
            result: *mut f64,
        ) -> mublasStatus_t;
        pub fn mublasCdotu(
            handle: mublasHandle_t,
            n: mublas_int,
            x: *const muComplex,
            incx: mublas_int,
            y: *const muComplex,
            incy: mublas_int,
            result: *mut muComplex,
        ) -> mublasStatus_t;
        pub fn mublasZdotu(
            handle: mublasHandle_t,
            n: mublas_int,
            x: *const muDoubleComplex,
            incx: mublas_int,
            y: *const muDoubleComplex,
            incy: mublas_int,
            result: *mut muDoubleComplex,
        ) -> mublasStatus_t;

        // ── muSOLVER:无生命周期符号(无独立句柄,复用 mublasHandle_t,003-D2)──
        // 返回值类型为 mublasStatus_t(musolver_functions.h)。
        // Phase 2:getrf/getrs(LU 分解 + 回代,solve 的两个阶段)。
        // 注意:*_bufferSize 无句柄参数;getrf/getrs 的 info/ipiv 为设备指针。
        pub fn musolverSgetrf_bufferSize(
            m: c_int,
            n: c_int,
            pivot: bool,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverDgetrf_bufferSize(
            m: c_int,
            n: c_int,
            pivot: bool,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverCgetrf_bufferSize(
            m: c_int,
            n: c_int,
            pivot: bool,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverZgetrf_bufferSize(
            m: c_int,
            n: c_int,
            pivot: bool,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverSgetrf(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            a: *mut f32,
            lda: c_int,
            ipiv: *mut c_int,
            info: *mut c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverDgetrf(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            a: *mut f64,
            lda: c_int,
            ipiv: *mut c_int,
            info: *mut c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverCgetrf(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            a: *mut muComplex,
            lda: c_int,
            ipiv: *mut c_int,
            info: *mut c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverZgetrf(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            a: *mut muDoubleComplex,
            lda: c_int,
            ipiv: *mut c_int,
            info: *mut c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverSgetrs_bufferSize(
            trans: c_int,
            n: c_int,
            nrhs: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverDgetrs_bufferSize(
            trans: c_int,
            n: c_int,
            nrhs: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverCgetrs_bufferSize(
            trans: c_int,
            n: c_int,
            nrhs: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverZgetrs_bufferSize(
            trans: c_int,
            n: c_int,
            nrhs: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverSgetrs(
            handle: mublasHandle_t,
            trans: c_int,
            n: c_int,
            nrhs: c_int,
            a: *const f32,
            lda: c_int,
            ipiv: *const c_int,
            b: *mut f32,
            ldb: c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverDgetrs(
            handle: mublasHandle_t,
            trans: c_int,
            n: c_int,
            nrhs: c_int,
            a: *const f64,
            lda: c_int,
            ipiv: *const c_int,
            b: *mut f64,
            ldb: c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverCgetrs(
            handle: mublasHandle_t,
            trans: c_int,
            n: c_int,
            nrhs: c_int,
            a: *const muComplex,
            lda: c_int,
            ipiv: *const c_int,
            b: *mut muComplex,
            ldb: c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverZgetrs(
            handle: mublasHandle_t,
            trans: c_int,
            n: c_int,
            nrhs: c_int,
            a: *const muDoubleComplex,
            lda: c_int,
            ipiv: *const c_int,
            b: *mut muDoubleComplex,
            ldb: c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        // geqrf/orgqr/cungqr/gesvd + *_bufferSize 由 Phase 3 追加。

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

    pub unsafe fn mublasSetPointerMode(
        handle: mublasHandle_t,
        _pointer_mode: mublasPointerMode_t,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        MUBLAS_STATUS_SUCCESS
    }

    // ── muBLAS 计算例程(mock:确定性数值,供 pytest 验证 shape/布局逻辑)──
    // 约定:Sgemm/Dgemm/Cgemm/Zgemm 把 C 全填 1.0;Sdot/Ddot 返回 n;
    // Cdotu/Zdotu 返回 (n, 0);getrf 保持 A 不变 + ipiv 恒等 + info=0;
    // getrs 保持 B 不变(输入即输出);*_bufferSize 返回 4096。

    pub unsafe fn mublasSgemm(
        handle: mublasHandle_t,
        _transa: c_int,
        _transb: c_int,
        m: mublas_int,
        n: mublas_int,
        _k: mublas_int,
        _alpha: *const f32,
        _a: *const f32,
        _lda: mublas_int,
        _b: *const f32,
        _ldb: mublas_int,
        _beta: *const f32,
        c: *mut f32,
        _ldc: mublas_int,
    ) -> mublasStatus_t {
        if handle.is_null() || c.is_null() {
            return 1;
        }
        unsafe { std::slice::from_raw_parts_mut(c, (m * n) as usize).fill(1.0) };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasDgemm(
        handle: mublasHandle_t,
        _transa: c_int,
        _transb: c_int,
        m: mublas_int,
        n: mublas_int,
        _k: mublas_int,
        _alpha: *const f64,
        _a: *const f64,
        _lda: mublas_int,
        _b: *const f64,
        _ldb: mublas_int,
        _beta: *const f64,
        c: *mut f64,
        _ldc: mublas_int,
    ) -> mublasStatus_t {
        if handle.is_null() || c.is_null() {
            return 1;
        }
        unsafe { std::slice::from_raw_parts_mut(c, (m * n) as usize).fill(1.0) };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasCgemm(
        handle: mublasHandle_t,
        _transa: c_int,
        _transb: c_int,
        m: mublas_int,
        n: mublas_int,
        _k: mublas_int,
        _alpha: *const muComplex,
        _a: *const muComplex,
        _lda: mublas_int,
        _b: *const muComplex,
        _ldb: mublas_int,
        _beta: *const muComplex,
        c: *mut muComplex,
        _ldc: mublas_int,
    ) -> mublasStatus_t {
        if handle.is_null() || c.is_null() {
            return 1;
        }
        unsafe {
            std::slice::from_raw_parts_mut(c, (m * n) as usize).fill(muComplex { re: 1.0, im: 0.0 })
        };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasZgemm(
        handle: mublasHandle_t,
        _transa: c_int,
        _transb: c_int,
        m: mublas_int,
        n: mublas_int,
        _k: mublas_int,
        _alpha: *const muDoubleComplex,
        _a: *const muDoubleComplex,
        _lda: mublas_int,
        _b: *const muDoubleComplex,
        _ldb: mublas_int,
        _beta: *const muDoubleComplex,
        c: *mut muDoubleComplex,
        _ldc: mublas_int,
    ) -> mublasStatus_t {
        if handle.is_null() || c.is_null() {
            return 1;
        }
        unsafe {
            std::slice::from_raw_parts_mut(c, (m * n) as usize)
                .fill(muDoubleComplex { re: 1.0, im: 0.0 })
        };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasSdot(
        handle: mublasHandle_t,
        n: mublas_int,
        _x: *const f32,
        _incx: mublas_int,
        _y: *const f32,
        _incy: mublas_int,
        result: *mut f32,
    ) -> mublasStatus_t {
        if handle.is_null() || result.is_null() {
            return 1;
        }
        unsafe { *result = n as f32 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasDdot(
        handle: mublasHandle_t,
        n: mublas_int,
        _x: *const f64,
        _incx: mublas_int,
        _y: *const f64,
        _incy: mublas_int,
        result: *mut f64,
    ) -> mublasStatus_t {
        if handle.is_null() || result.is_null() {
            return 1;
        }
        unsafe { *result = n as f64 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasCdotu(
        handle: mublasHandle_t,
        n: mublas_int,
        _x: *const muComplex,
        _incx: mublas_int,
        _y: *const muComplex,
        _incy: mublas_int,
        result: *mut muComplex,
    ) -> mublasStatus_t {
        if handle.is_null() || result.is_null() {
            return 1;
        }
        unsafe {
            *result = muComplex {
                re: n as f32,
                im: 0.0,
            }
        };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn mublasZdotu(
        handle: mublasHandle_t,
        n: mublas_int,
        _x: *const muDoubleComplex,
        _incx: mublas_int,
        _y: *const muDoubleComplex,
        _incy: mublas_int,
        result: *mut muDoubleComplex,
    ) -> mublasStatus_t {
        if handle.is_null() || result.is_null() {
            return 1;
        }
        unsafe {
            *result = muDoubleComplex {
                re: n as f64,
                im: 0.0,
            }
        };
        MUBLAS_STATUS_SUCCESS
    }

    // ── muSOLVER(mock:确定性数值)──

    pub unsafe fn musolverSgetrf_bufferSize(
        _m: c_int,
        _n: c_int,
        _pivot: bool,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDgetrf_bufferSize(
        _m: c_int,
        _n: c_int,
        _pivot: bool,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCgetrf_bufferSize(
        _m: c_int,
        _n: c_int,
        _pivot: bool,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZgetrf_bufferSize(
        _m: c_int,
        _n: c_int,
        _pivot: bool,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverSgetrf(
        handle: mublasHandle_t,
        _m: c_int,
        n: c_int,
        _a: *mut f32,
        _lda: c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || ipiv.is_null() || info.is_null() {
            return 1;
        }
        // ipiv 恒等置换(LAPACK 1-based:ipiv[i] = i+1)+ info=0(非奇异)
        unsafe {
            for i in 0..n as usize {
                *ipiv.add(i) = (i + 1) as c_int;
            }
            *info = 0;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDgetrf(
        handle: mublasHandle_t,
        _m: c_int,
        n: c_int,
        _a: *mut f64,
        _lda: c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || ipiv.is_null() || info.is_null() {
            return 1;
        }
        unsafe {
            for i in 0..n as usize {
                *ipiv.add(i) = (i + 1) as c_int;
            }
            *info = 0;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCgetrf(
        handle: mublasHandle_t,
        _m: c_int,
        n: c_int,
        _a: *mut muComplex,
        _lda: c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || ipiv.is_null() || info.is_null() {
            return 1;
        }
        unsafe {
            for i in 0..n as usize {
                *ipiv.add(i) = (i + 1) as c_int;
            }
            *info = 0;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZgetrf(
        handle: mublasHandle_t,
        _m: c_int,
        n: c_int,
        _a: *mut muDoubleComplex,
        _lda: c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || ipiv.is_null() || info.is_null() {
            return 1;
        }
        unsafe {
            for i in 0..n as usize {
                *ipiv.add(i) = (i + 1) as c_int;
            }
            *info = 0;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverSgetrs_bufferSize(
        _trans: c_int,
        _n: c_int,
        _nrhs: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDgetrs_bufferSize(
        _trans: c_int,
        _n: c_int,
        _nrhs: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCgetrs_bufferSize(
        _trans: c_int,
        _n: c_int,
        _nrhs: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZgetrs_bufferSize(
        _trans: c_int,
        _n: c_int,
        _nrhs: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverSgetrs(
        handle: mublasHandle_t,
        _trans: c_int,
        _n: c_int,
        _nrhs: c_int,
        _a: *const f32,
        _lda: c_int,
        _ipiv: *const c_int,
        _b: *mut f32,
        _ldb: c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        // B 原地保持不动(mock 下输入即输出)
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDgetrs(
        handle: mublasHandle_t,
        _trans: c_int,
        _n: c_int,
        _nrhs: c_int,
        _a: *const f64,
        _lda: c_int,
        _ipiv: *const c_int,
        _b: *mut f64,
        _ldb: c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCgetrs(
        handle: mublasHandle_t,
        _trans: c_int,
        _n: c_int,
        _nrhs: c_int,
        _a: *const muComplex,
        _lda: c_int,
        _ipiv: *const c_int,
        _b: *mut muComplex,
        _ldb: c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZgetrs(
        handle: mublasHandle_t,
        _trans: c_int,
        _n: c_int,
        _nrhs: c_int,
        _a: *const muDoubleComplex,
        _lda: c_int,
        _ipiv: *const c_int,
        _b: *mut muDoubleComplex,
        _ldb: c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
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

    pub unsafe fn murandSetStream(
        generator: murandGenerator_t,
        _stream: musaStream_t,
    ) -> murandStatus_t {
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

    pub unsafe fn musparseSetStream(
        handle: musparseHandle_t,
        _stream: musaStream_t,
    ) -> musparseStatus_t {
        if handle.is_null() {
            return 1;
        }
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseGetVersion(
        handle: musparseHandle_t,
        version: *mut c_int,
    ) -> musparseStatus_t {
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
        assert_eq!(
            unsafe { mublasGetVersion(h, &mut v) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(v, 30100);
        assert_eq!(
            unsafe { mublasSetStream(h, std::ptr::null_mut()) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(unsafe { mublasDestroy(h) }, MUBLAS_STATUS_SUCCESS);
    }

    #[test]
    fn test_mublas_null_rejected() {
        assert_ne!(
            unsafe { mublasCreate(std::ptr::null_mut()) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_ne!(
            unsafe { mublasDestroy(std::ptr::null_mut()) },
            MUBLAS_STATUS_SUCCESS
        );
    }

    #[test]
    fn test_mublas_pointer_mode() {
        let mut h: mublasHandle_t = std::ptr::null_mut();
        assert_eq!(unsafe { mublasCreate(&mut h) }, MUBLAS_STATUS_SUCCESS);
        assert_eq!(
            unsafe { mublasSetPointerMode(h, MUBLAS_POINTER_MODE_HOST) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_ne!(
            unsafe { mublasSetPointerMode(std::ptr::null_mut(), MUBLAS_POINTER_MODE_HOST) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(unsafe { mublasDestroy(h) }, MUBLAS_STATUS_SUCCESS);
    }

    #[test]
    fn test_mublas_gemm_mock_fills_c() {
        let mut h: mublasHandle_t = std::ptr::null_mut();
        assert_eq!(unsafe { mublasCreate(&mut h) }, MUBLAS_STATUS_SUCCESS);

        let mut c = [0.0f32; 6]; // 2×3
        assert_eq!(
            unsafe {
                mublasSgemm(
                    h,
                    MUBLAS_OP_N,
                    MUBLAS_OP_N,
                    2,
                    3,
                    4,
                    &1.0,
                    std::ptr::null(),
                    4,
                    std::ptr::null(),
                    3,
                    &0.0,
                    c.as_mut_ptr(),
                    3,
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert!(c.iter().all(|&v| v == 1.0));

        let mut cd = [0.0f64; 6];
        assert_eq!(
            unsafe {
                mublasDgemm(
                    h,
                    MUBLAS_OP_N,
                    MUBLAS_OP_N,
                    2,
                    3,
                    4,
                    &1.0,
                    std::ptr::null(),
                    4,
                    std::ptr::null(),
                    3,
                    &0.0,
                    cd.as_mut_ptr(),
                    3,
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert!(cd.iter().all(|&v| v == 1.0));

        let mut cc = [muComplex { re: 0.0, im: 0.0 }; 4];
        assert_eq!(
            unsafe {
                mublasCgemm(
                    h,
                    MUBLAS_OP_N,
                    MUBLAS_OP_N,
                    2,
                    2,
                    2,
                    &muComplex { re: 1.0, im: 0.0 },
                    std::ptr::null(),
                    2,
                    std::ptr::null(),
                    2,
                    &muComplex { re: 0.0, im: 0.0 },
                    cc.as_mut_ptr(),
                    2,
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert!(cc.iter().all(|v| v.re == 1.0 && v.im == 0.0));

        assert_eq!(unsafe { mublasDestroy(h) }, MUBLAS_STATUS_SUCCESS);
    }

    #[test]
    fn test_mublas_dot_mock_returns_n() {
        let mut h: mublasHandle_t = std::ptr::null_mut();
        assert_eq!(unsafe { mublasCreate(&mut h) }, MUBLAS_STATUS_SUCCESS);

        let mut r = 0.0f32;
        assert_eq!(
            unsafe { mublasSdot(h, 7, std::ptr::null(), 1, std::ptr::null(), 1, &mut r) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(r, 7.0);

        let mut rd = 0.0f64;
        assert_eq!(
            unsafe { mublasDdot(h, 5, std::ptr::null(), 1, std::ptr::null(), 1, &mut rd) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(rd, 5.0);

        let mut rc = muComplex { re: 0.0, im: 0.0 };
        assert_eq!(
            unsafe { mublasCdotu(h, 3, std::ptr::null(), 1, std::ptr::null(), 1, &mut rc) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(rc.re, 3.0);
        assert_eq!(rc.im, 0.0);

        assert_eq!(unsafe { mublasDestroy(h) }, MUBLAS_STATUS_SUCCESS);
    }

    #[test]
    fn test_musolver_getrf_getrs_mock() {
        let mut h: mublasHandle_t = std::ptr::null_mut();
        assert_eq!(unsafe { mublasCreate(&mut h) }, MUBLAS_STATUS_SUCCESS);

        // bufferSize:无句柄参数,返回 4096
        let mut bs: c_int = 0;
        assert_eq!(
            unsafe { musolverSgetrf_bufferSize(4, 4, true, &mut bs) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(bs, 4096);
        assert_eq!(
            unsafe { musolverDgetrs_bufferSize(MUBLAS_OP_N, 4, 1, &mut bs) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(bs, 4096);

        // getrf:ipiv 恒等置换 + info=0
        let mut a = [0.0f64; 9];
        let mut ipiv = [0i32; 3];
        let mut info: c_int = -1;
        assert_eq!(
            unsafe {
                musolverDgetrf(
                    h,
                    3,
                    3,
                    a.as_mut_ptr(),
                    3,
                    ipiv.as_mut_ptr(),
                    &mut info,
                    std::ptr::null_mut(),
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(info, 0);
        assert_eq!(ipiv, [1, 2, 3]);

        // getrs:B 原地不动(mock),仅校验返回码
        let mut b = [1.0f64; 3];
        assert_eq!(
            unsafe {
                musolverDgetrs(
                    h,
                    MUBLAS_OP_N,
                    3,
                    1,
                    a.as_ptr(),
                    3,
                    ipiv.as_ptr(),
                    b.as_mut_ptr(),
                    3,
                    std::ptr::null_mut(),
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(b, [1.0, 1.0, 1.0]);

        assert_eq!(unsafe { mublasDestroy(h) }, MUBLAS_STATUS_SUCCESS);
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
        assert_eq!(
            unsafe { mufftPlan1d(&mut p1, 64, MUFFT_C2C, 1) },
            MUFFT_SUCCESS
        );
        assert!(!p1.is_null());
        assert_eq!(
            unsafe { mufftSetStream(p1, std::ptr::null_mut()) },
            MUFFT_SUCCESS
        );
        assert_eq!(unsafe { mufftDestroy(p1) }, MUFFT_SUCCESS);

        let mut p2: mufftHandle = std::ptr::null_mut();
        assert_eq!(
            unsafe { mufftPlan2d(&mut p2, 8, 8, MUFFT_Z2Z) },
            MUFFT_SUCCESS
        );
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
