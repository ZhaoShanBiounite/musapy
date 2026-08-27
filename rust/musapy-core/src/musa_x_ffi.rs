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

/// muSPARSE 稀疏矩阵描述符(opaque 指针;musparse-types.h:113)。
pub type musparseSpMatDescr_t = *mut c_void;

/// muSPARSE 稠密向量描述符(opaque 指针;musparse-types.h:114)。
pub type musparseDnVecDescr_t = *mut c_void;

/// muSPARSE 稠密矩阵描述符(opaque 指针;musparse-types.h:115)。
pub type musparseDnMatDescr_t = *mut c_void;

// ============================================================
// 标量/复数类型(Phase 2:gemm/dot 的 alpha/beta 与复数矩阵)
// ============================================================

/// muBLAS 整数类型(mublas-types.h:`typedef int32_t mublas_int`,LP64 默认;
/// 仅 `mublas_ILP64` 宏开启时为 int64,本项目不启用)。
pub type mublas_int = c_int;

/// 单精度复数(muComplex.h:`typedef float2 muFloatComplex; typedef muFloatComplex muComplex`)。
/// ABI 与 C 的 `float2 { x, y }` 一致。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct muComplex {
    pub re: f32,
    pub im: f32,
}

/// 双精度复数(muComplex.h:`typedef double2 muDoubleComplex`)。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
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

/// mublasSvect_t(musolver_extra_types.h:43-51)。gesvd 的左/右奇异向量模式
/// (替代 LAPACK 的 jobu/jobvt 字符参数;Phase 3 svd)。
pub type mublasSvect_t = c_int;
pub const MUBLAS_SVECT_ALL: mublasSvect_t = 191; // 计算整个正交矩阵(满)
pub const MUBLAS_SVECT_SINGULAR: mublasSvect_t = 192; // 仅奇异向量(薄)
pub const MUBLAS_SVECT_OVERWRITE: mublasSvect_t = 193; // 覆写输入矩阵
pub const MUBLAS_SVECT_NONE: mublasSvect_t = 194; // 不计算

/// mublasWorkmode_t(musolver_extra_types.h:56-62)。gesvd 的快速算法模式。
pub type mublasWorkmode_t = c_int;
pub const MUBLAS_OUTOFPLACE: mublasWorkmode_t = 201;
pub const MUBLAS_INPLACE: mublasWorkmode_t = 202;

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

/// muSPARSE 常量(musparse-types.h / library_types.h 实测,Phase 6)。
pub type musparseIndexType_t = c_int;
pub const MUSPARSE_INDEX_32I: musparseIndexType_t = 2;
pub const MUSPARSE_INDEX_64I: musparseIndexType_t = 3;

pub type musparseIndexBase_t = c_int;
pub const MUSPARSE_INDEX_BASE_ZERO: musparseIndexBase_t = 0;

pub type musparseMatrixType_t = c_int;
pub const MUSPARSE_MATRIX_TYPE_GENERAL: musparseMatrixType_t = 0;

/// 稠密数据类型(musparseDataType_t ≡ musaDataType_t,library_types.h:7)。
pub type musparseDataType_t = c_int;
pub const MUSA_R_32F: musparseDataType_t = 0;
pub const MUSA_R_64F: musparseDataType_t = 1;

pub type musparseOperation_t = c_int;
pub const MUSPARSE_OPERATION_NON_TRANSPOSE: musparseOperation_t = 111;
pub const MUSPARSE_OPERATION_TRANSPOSE: musparseOperation_t = 112;
pub const MUSPARSE_OPERATION_CONJUGATE_TRANSPOSE: musparseOperation_t = 113;

pub type musparseOrder_t = c_int;
pub const MUSPARSE_ORDER_ROW: musparseOrder_t = 0;
pub const MUSPARSE_ORDER_COLUMN: musparseOrder_t = 1;

pub type musparseSpMVAlg_t = c_int;
pub const MUSPARSE_SPMV_ALG_DEFAULT: musparseSpMVAlg_t = 0;

pub type musparseSpMMAlg_t = c_int;
pub const MUSPARSE_SPMM_ALG_DEFAULT: musparseSpMMAlg_t = 0;

pub type musparseSpMMStage_t = c_int;
pub const MUSPARSE_SPMM_STAGE_AUTO: musparseSpMMStage_t = 0;

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

        // ── muSOLVER Phase 3（musolver_functions.h;lu/qr/svd 分解）──
        // 注意：geqrf 的 ipiv 参数实为 tau（Householder 反射系数，设备指针）；
        // gesvd 为 muSOLVER 私有签名（mublasSvect 枚举 + E 输出 + fast_alg）。
        pub fn musolverSgeqrf_bufferSize(
            m: c_int,
            n: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverDgeqrf_bufferSize(
            m: c_int,
            n: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverCgeqrf_bufferSize(
            m: c_int,
            n: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverZgeqrf_bufferSize(
            m: c_int,
            n: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverSgeqrf(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            a: *mut f32,
            lda: c_int,
            tau: *mut f32,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverDgeqrf(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            a: *mut f64,
            lda: c_int,
            tau: *mut f64,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverCgeqrf(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            a: *mut muComplex,
            lda: c_int,
            tau: *mut muComplex,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverZgeqrf(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            a: *mut muDoubleComplex,
            lda: c_int,
            tau: *mut muDoubleComplex,
            buffer: *mut c_void,
        ) -> mublasStatus_t;

        pub fn musolverSorgqr_bufferSize(
            m: c_int,
            n: c_int,
            k: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverDorgqr_bufferSize(
            m: c_int,
            n: c_int,
            k: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverCungqr_bufferSize(
            m: c_int,
            n: c_int,
            k: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverZungqr_bufferSize(
            m: c_int,
            n: c_int,
            k: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverSorgqr(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *mut f32,
            lda: c_int,
            tau: *const f32,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverDorgqr(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *mut f64,
            lda: c_int,
            tau: *const f64,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverCungqr(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *mut muComplex,
            lda: c_int,
            tau: *const muComplex,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverZungqr(
            handle: mublasHandle_t,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *mut muDoubleComplex,
            lda: c_int,
            tau: *const muDoubleComplex,
            buffer: *mut c_void,
        ) -> mublasStatus_t;

        pub fn musolverSgesvd_bufferSize(
            left_svect: c_int,
            right_svect: c_int,
            m: c_int,
            n: c_int,
            batch_count: c_int,
            fast_alg: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverDgesvd_bufferSize(
            left_svect: c_int,
            right_svect: c_int,
            m: c_int,
            n: c_int,
            batch_count: c_int,
            fast_alg: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverCgesvd_bufferSize(
            left_svect: c_int,
            right_svect: c_int,
            m: c_int,
            n: c_int,
            batch_count: c_int,
            fast_alg: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverZgesvd_bufferSize(
            left_svect: c_int,
            right_svect: c_int,
            m: c_int,
            n: c_int,
            batch_count: c_int,
            fast_alg: c_int,
            buffersize: *mut c_int,
        ) -> mublasStatus_t;
        pub fn musolverSgesvd(
            handle: mublasHandle_t,
            left_svect: c_int,
            right_svect: c_int,
            m: c_int,
            n: c_int,
            a: *mut f32,
            lda: c_int,
            s: *mut f32,
            u: *mut f32,
            ldu: c_int,
            v: *mut f32,
            ldv: c_int,
            e: *mut f32,
            fast_alg: c_int,
            info: *mut c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverDgesvd(
            handle: mublasHandle_t,
            left_svect: c_int,
            right_svect: c_int,
            m: c_int,
            n: c_int,
            a: *mut f64,
            lda: c_int,
            s: *mut f64,
            u: *mut f64,
            ldu: c_int,
            v: *mut f64,
            ldv: c_int,
            e: *mut f64,
            fast_alg: c_int,
            info: *mut c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverCgesvd(
            handle: mublasHandle_t,
            left_svect: c_int,
            right_svect: c_int,
            m: c_int,
            n: c_int,
            a: *mut muComplex,
            lda: c_int,
            s: *mut f32,
            u: *mut muComplex,
            ldu: c_int,
            v: *mut muComplex,
            ldv: c_int,
            e: *mut muComplex,
            fast_alg: c_int,
            info: *mut c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        pub fn musolverZgesvd(
            handle: mublasHandle_t,
            left_svect: c_int,
            right_svect: c_int,
            m: c_int,
            n: c_int,
            a: *mut muDoubleComplex,
            lda: c_int,
            s: *mut f64,
            u: *mut muDoubleComplex,
            ldu: c_int,
            v: *mut muDoubleComplex,
            ldv: c_int,
            e: *mut muDoubleComplex,
            fast_alg: c_int,
            info: *mut c_int,
            buffer: *mut c_void,
        ) -> mublasStatus_t;
        // syevd（对称特征值）由 linalg C（eigh）阶段追加。

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
        // Phase 4:计算/配置（murand.h;2026-08-07 头文件核对）
        // 注意:Generate* 的 n 为元素个数(size_t);Normal 原生带 mean/stddev
        // 参数——normal(loc,scale) 可一步生成,uniform(low,high) 仍需仿射。
        pub fn murandSetPseudoRandomGeneratorSeed(
            generator: murandGenerator_t,
            seed: u64,
        ) -> murandStatus_t;
        pub fn murandSetGeneratorOffset(
            generator: murandGenerator_t,
            offset: u64,
        ) -> murandStatus_t;
        pub fn murandGenerateUniform(
            generator: murandGenerator_t,
            output_data: *mut f32,
            n: usize,
        ) -> murandStatus_t;
        pub fn murandGenerateUniformDouble(
            generator: murandGenerator_t,
            output_data: *mut f64,
            n: usize,
        ) -> murandStatus_t;
        pub fn murandGenerateNormal(
            generator: murandGenerator_t,
            output_data: *mut f32,
            n: usize,
            mean: f32,
            stddev: f32,
        ) -> murandStatus_t;
        pub fn murandGenerateNormalDouble(
            generator: murandGenerator_t,
            output_data: *mut f64,
            n: usize,
            mean: f64,
            stddev: f64,
        ) -> murandStatus_t;

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
        // ── muFFT 执行例程（Phase 5，mufft.h 实测签名）──
        // mufftComplex ≡ muComplex、mufftDoubleComplex ≡ muDoubleComplex
        // （mufft.h:123/127 typedef），interleaved re/im buffer 直接透传指针。
        // C2C/Z2Z：direction = MUFFT_FORWARD(-1) / MUFFT_INVERSE(1)
        pub fn mufftExecC2C(
            plan: mufftHandle,
            idata: *mut muComplex,
            odata: *mut muComplex,
            direction: c_int,
        ) -> mufftResult;
        pub fn mufftExecR2C(
            plan: mufftHandle,
            idata: *mut f32,
            odata: *mut muComplex,
        ) -> mufftResult;
        pub fn mufftExecZ2Z(
            plan: mufftHandle,
            idata: *mut muDoubleComplex,
            odata: *mut muDoubleComplex,
            direction: c_int,
        ) -> mufftResult;
        pub fn mufftExecD2Z(
            plan: mufftHandle,
            idata: *mut f64,
            odata: *mut muDoubleComplex,
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
        // ── muSPARSE 泛型 API(Phase 6,musparse-auxiliary.h/functions.h 实测)──
        // 所有 data/indices/values 均为 device 指针;alpha/beta 为 host 标量
        // (默认 PointerMode HOST)。描述符销毁统一走 DestroySpMat/DestroyDnVec/DestroyDnMat。
        pub fn musparseCreateCsr(
            descr: *mut musparseSpMatDescr_t,
            rows: i64,
            cols: i64,
            nnz: i64,
            csr_row_ptr: *mut c_void,
            csr_col_ind: *mut c_void,
            csr_val: *mut c_void,
            row_ptr_type: musparseIndexType_t,
            col_ind_type: musparseIndexType_t,
            idx_base: musparseIndexBase_t,
            data_type: musparseDataType_t,
        ) -> musparseStatus_t;
        pub fn musparseDestroySpMat(descr: musparseSpMatDescr_t) -> musparseStatus_t;
        pub fn musparseCreateDnVec(
            descr: *mut musparseDnVecDescr_t,
            size: i64,
            values: *mut c_void,
            data_type: musparseDataType_t,
        ) -> musparseStatus_t;
        pub fn musparseDestroyDnVec(descr: musparseDnVecDescr_t) -> musparseStatus_t;
        pub fn musparseCreateDnMat(
            descr: *mut musparseDnMatDescr_t,
            rows: i64,
            cols: i64,
            ld: i64,
            values: *mut c_void,
            data_type: musparseDataType_t,
            order: musparseOrder_t,
        ) -> musparseStatus_t;
        pub fn musparseDestroyDnMat(descr: musparseDnMatDescr_t) -> musparseStatus_t;
        // 两段式在同一调用:temp_buffer=NULL 时写所需字节数到 buffer_size 并返回
        // (无独立 _bufferSize 符号,SDK 限制 2.5);再用 buffer 调计算。
        pub fn musparseSpMV(
            handle: musparseHandle_t,
            trans: musparseOperation_t,
            alpha: *const c_void,
            mat: musparseSpMatDescr_t,
            x: musparseDnVecDescr_t,
            beta: *const c_void,
            y: musparseDnVecDescr_t,
            compute_type: musparseDataType_t,
            alg: musparseSpMVAlg_t,
            buffer_size: *mut usize,
            temp_buffer: *mut c_void,
        ) -> musparseStatus_t;
        pub fn musparseSpMM(
            handle: musparseHandle_t,
            trans_A: musparseOperation_t,
            trans_B: musparseOperation_t,
            alpha: *const c_void,
            mat_A: musparseSpMatDescr_t,
            mat_B: musparseDnMatDescr_t,
            beta: *const c_void,
            mat_C: musparseDnMatDescr_t,
            compute_type: musparseDataType_t,
            alg: musparseSpMMAlg_t,
            stage: musparseSpMMStage_t,
            buffer_size: *mut usize,
            temp_buffer: *mut c_void,
        ) -> musparseStatus_t;
    }
}

// ============================================================
// Mock stub(同签名,musapy_mock_musa)
// ============================================================

#[cfg(musapy_mock_musa)]
mod mock {
    // mock stub 在 unsafe fn 体内直接操作裸指针（host 内存 + dummy 句柄，语义安全）。
    // edition 2024 的 unsafe_op_in_unsafe_fn 在此为纯噪音，allow 掉。
    #![allow(unsafe_op_in_unsafe_fn)]
    #![allow(dead_code)] // mock 仿真记录字段（rows/nnz/data_type/ftype 等）仅写入未读，保留供对照
    use super::*;
    use std::collections::HashMap;
    use std::sync::LazyLock;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// dummy 句柄计数器(mock 下句柄只需非空且可区分)。
    static MOCK_MATH_HANDLE_COUNTER: AtomicUsize = AtomicUsize::new(0x1000_0000);

    /// mock 句柄全局表。
    ///
    /// mock 下句柄是 `next_handle()` 生成的 dummy 裸指针值，仅作身份标识、从不解引用；
    /// 跨线程访问由 `Mutex` 串行化。裸指针 key 本身非 `Send`/`Sync`（`HashMap` 值无此约束），
    /// 无法直接放入 `static`，这里按 mock 语义手动标注（真值则不可，见 L2-3 注释）。
    struct MockHandleTable<T> {
        inner: Mutex<HashMap<*mut c_void, T>>,
    }

    impl<T> MockHandleTable<T> {
        fn new() -> Self {
            Self {
                inner: Mutex::new(HashMap::new()),
            }
        }

        fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<*mut c_void, T>> {
            self.inner.lock().unwrap()
        }
    }

    unsafe impl<T> Send for MockHandleTable<T> {}
    unsafe impl<T> Sync for MockHandleTable<T> {}

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

    // ── muSOLVER Phase 3（mock：确定性数值，供 pytest 验证形状/视图逻辑）──

    /// mock 确定性填充的数值类型辅助（f32/f64/complex）。
    trait MockNum {
        fn zero() -> Self;
        fn one() -> Self;
        fn from_usize(n: usize) -> Self;
    }

    impl MockNum for f32 {
        fn zero() -> Self {
            0.0
        }
        fn one() -> Self {
            1.0
        }
        fn from_usize(n: usize) -> Self {
            n as f32
        }
    }

    impl MockNum for f64 {
        fn zero() -> Self {
            0.0
        }
        fn one() -> Self {
            1.0
        }
        fn from_usize(n: usize) -> Self {
            n as f64
        }
    }

    impl MockNum for muComplex {
        fn zero() -> Self {
            muComplex { re: 0.0, im: 0.0 }
        }
        fn one() -> Self {
            muComplex { re: 1.0, im: 0.0 }
        }
        fn from_usize(n: usize) -> Self {
            muComplex {
                re: n as f32,
                im: 0.0,
            }
        }
    }

    impl MockNum for muDoubleComplex {
        fn zero() -> Self {
            muDoubleComplex { re: 0.0, im: 0.0 }
        }
        fn one() -> Self {
            muDoubleComplex { re: 1.0, im: 0.0 }
        }
        fn from_usize(n: usize) -> Self {
            muDoubleComplex {
                re: n as f64,
                im: 0.0,
            }
        }
    }

    pub unsafe fn musolverSgeqrf_bufferSize(
        _m: c_int,
        _n: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDgeqrf_bufferSize(
        _m: c_int,
        _n: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCgeqrf_bufferSize(
        _m: c_int,
        _n: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZgeqrf_bufferSize(
        _m: c_int,
        _n: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverSorgqr_bufferSize(
        _m: c_int,
        _n: c_int,
        _k: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDorgqr_bufferSize(
        _m: c_int,
        _n: c_int,
        _k: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCungqr_bufferSize(
        _m: c_int,
        _n: c_int,
        _k: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZungqr_bufferSize(
        _m: c_int,
        _n: c_int,
        _k: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverSgesvd_bufferSize(
        _left_svect: c_int,
        _right_svect: c_int,
        _m: c_int,
        _n: c_int,
        _batch_count: c_int,
        _fast_alg: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDgesvd_bufferSize(
        _left_svect: c_int,
        _right_svect: c_int,
        _m: c_int,
        _n: c_int,
        _batch_count: c_int,
        _fast_alg: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCgesvd_bufferSize(
        _left_svect: c_int,
        _right_svect: c_int,
        _m: c_int,
        _n: c_int,
        _batch_count: c_int,
        _fast_alg: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZgesvd_bufferSize(
        _left_svect: c_int,
        _right_svect: c_int,
        _m: c_int,
        _n: c_int,
        _batch_count: c_int,
        _fast_alg: c_int,
        buffersize: *mut c_int,
    ) -> mublasStatus_t {
        if buffersize.is_null() {
            return 1;
        }
        unsafe { *buffersize = 4096 };
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverSgeqrf(
        handle: mublasHandle_t,
        m: c_int,
        n: c_int,
        _a: *mut f32,
        _lda: c_int,
        tau: *mut f32,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || tau.is_null() {
            return 1;
        }
        // mock：A 保持原样，tau 全 0（不产生实际反射变换）
        let k = (m.min(n)) as usize;
        unsafe {
            for i in 0..k {
                *tau.add(i) = MockNum::zero();
            }
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDgeqrf(
        handle: mublasHandle_t,
        m: c_int,
        n: c_int,
        _a: *mut f64,
        _lda: c_int,
        tau: *mut f64,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || tau.is_null() {
            return 1;
        }
        // mock：A 保持原样，tau 全 0（不产生实际反射变换）
        let k = (m.min(n)) as usize;
        unsafe {
            for i in 0..k {
                *tau.add(i) = MockNum::zero();
            }
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCgeqrf(
        handle: mublasHandle_t,
        m: c_int,
        n: c_int,
        _a: *mut muComplex,
        _lda: c_int,
        tau: *mut muComplex,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || tau.is_null() {
            return 1;
        }
        // mock：A 保持原样，tau 全 0（不产生实际反射变换）
        let k = (m.min(n)) as usize;
        unsafe {
            for i in 0..k {
                *tau.add(i) = MockNum::zero();
            }
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZgeqrf(
        handle: mublasHandle_t,
        m: c_int,
        n: c_int,
        _a: *mut muDoubleComplex,
        _lda: c_int,
        tau: *mut muDoubleComplex,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || tau.is_null() {
            return 1;
        }
        // mock：A 保持原样，tau 全 0（不产生实际反射变换）
        let k = (m.min(n)) as usize;
        unsafe {
            for i in 0..k {
                *tau.add(i) = MockNum::zero();
            }
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverSorgqr(
        handle: mublasHandle_t,
        _m: c_int,
        _n: c_int,
        _k: c_int,
        _a: *mut f32,
        _lda: c_int,
        _tau: *const f32,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        // mock：缓冲原样返回（Q = geqrf 后的 A 内容，仅验证形状/视图）
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDorgqr(
        handle: mublasHandle_t,
        _m: c_int,
        _n: c_int,
        _k: c_int,
        _a: *mut f64,
        _lda: c_int,
        _tau: *const f64,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        // mock：缓冲原样返回（Q = geqrf 后的 A 内容，仅验证形状/视图）
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCungqr(
        handle: mublasHandle_t,
        _m: c_int,
        _n: c_int,
        _k: c_int,
        _a: *mut muComplex,
        _lda: c_int,
        _tau: *const muComplex,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        // mock：缓冲原样返回（Q = geqrf 后的 A 内容，仅验证形状/视图）
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZungqr(
        handle: mublasHandle_t,
        _m: c_int,
        _n: c_int,
        _k: c_int,
        _a: *mut muDoubleComplex,
        _lda: c_int,
        _tau: *const muDoubleComplex,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() {
            return 1;
        }
        // mock：缓冲原样返回（Q = geqrf 后的 A 内容，仅验证形状/视图）
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverSgesvd(
        handle: mublasHandle_t,
        left_svect: c_int,
        right_svect: c_int,
        m: c_int,
        n: c_int,
        _a: *mut f32,
        _lda: c_int,
        s: *mut f32,
        u: *mut f32,
        ldu: c_int,
        v: *mut f32,
        ldv: c_int,
        _e: *mut f32,
        _fast_alg: c_int,
        info: *mut c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || s.is_null() || info.is_null() {
            return 1;
        }
        let k = m.min(n) as usize;
        unsafe {
            // S 降序 k..1（mock 确定性数值；真实值由真机 pytest 对照）
            for i in 0..k {
                *s.add(i) = MockNum::from_usize(k - i);
            }
            // U/V 单位阵（left/right_svect 决定列数；NONE 时指针可为 null）
            if !u.is_null() {
                let u_cols = if left_svect == super::MUBLAS_SVECT_ALL {
                    m as usize
                } else {
                    k
                };
                for j in 0..u_cols {
                    for i in 0..m as usize {
                        *u.add(i + j * ldu as usize) = if i == j {
                            MockNum::one()
                        } else {
                            MockNum::zero()
                        };
                    }
                }
            }
            if !v.is_null() {
                let v_cols = if right_svect == super::MUBLAS_SVECT_ALL {
                    n as usize
                } else {
                    k
                };
                for j in 0..v_cols {
                    for i in 0..n as usize {
                        *v.add(i + j * ldv as usize) = if i == j {
                            MockNum::one()
                        } else {
                            MockNum::zero()
                        };
                    }
                }
            }
            *info = 0;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverDgesvd(
        handle: mublasHandle_t,
        left_svect: c_int,
        right_svect: c_int,
        m: c_int,
        n: c_int,
        _a: *mut f64,
        _lda: c_int,
        s: *mut f64,
        u: *mut f64,
        ldu: c_int,
        v: *mut f64,
        ldv: c_int,
        _e: *mut f64,
        _fast_alg: c_int,
        info: *mut c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || s.is_null() || info.is_null() {
            return 1;
        }
        let k = m.min(n) as usize;
        unsafe {
            // S 降序 k..1（mock 确定性数值；真实值由真机 pytest 对照）
            for i in 0..k {
                *s.add(i) = MockNum::from_usize(k - i);
            }
            // U/V 单位阵（left/right_svect 决定列数；NONE 时指针可为 null）
            if !u.is_null() {
                let u_cols = if left_svect == super::MUBLAS_SVECT_ALL {
                    m as usize
                } else {
                    k
                };
                for j in 0..u_cols {
                    for i in 0..m as usize {
                        *u.add(i + j * ldu as usize) = if i == j {
                            MockNum::one()
                        } else {
                            MockNum::zero()
                        };
                    }
                }
            }
            if !v.is_null() {
                let v_cols = if right_svect == super::MUBLAS_SVECT_ALL {
                    n as usize
                } else {
                    k
                };
                for j in 0..v_cols {
                    for i in 0..n as usize {
                        *v.add(i + j * ldv as usize) = if i == j {
                            MockNum::one()
                        } else {
                            MockNum::zero()
                        };
                    }
                }
            }
            *info = 0;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverCgesvd(
        handle: mublasHandle_t,
        left_svect: c_int,
        right_svect: c_int,
        m: c_int,
        n: c_int,
        _a: *mut muComplex,
        _lda: c_int,
        s: *mut muComplex,
        u: *mut muComplex,
        ldu: c_int,
        v: *mut muComplex,
        ldv: c_int,
        _e: *mut muComplex,
        _fast_alg: c_int,
        info: *mut c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || s.is_null() || info.is_null() {
            return 1;
        }
        let k = m.min(n) as usize;
        unsafe {
            // S 降序 k..1（mock 确定性数值；真实值由真机 pytest 对照）
            for i in 0..k {
                *s.add(i) = MockNum::from_usize(k - i);
            }
            // U/V 单位阵（left/right_svect 决定列数；NONE 时指针可为 null）
            if !u.is_null() {
                let u_cols = if left_svect == super::MUBLAS_SVECT_ALL {
                    m as usize
                } else {
                    k
                };
                for j in 0..u_cols {
                    for i in 0..m as usize {
                        *u.add(i + j * ldu as usize) = if i == j {
                            MockNum::one()
                        } else {
                            MockNum::zero()
                        };
                    }
                }
            }
            if !v.is_null() {
                let v_cols = if right_svect == super::MUBLAS_SVECT_ALL {
                    n as usize
                } else {
                    k
                };
                for j in 0..v_cols {
                    for i in 0..n as usize {
                        *v.add(i + j * ldv as usize) = if i == j {
                            MockNum::one()
                        } else {
                            MockNum::zero()
                        };
                    }
                }
            }
            *info = 0;
        }
        MUBLAS_STATUS_SUCCESS
    }

    pub unsafe fn musolverZgesvd(
        handle: mublasHandle_t,
        left_svect: c_int,
        right_svect: c_int,
        m: c_int,
        n: c_int,
        _a: *mut muDoubleComplex,
        _lda: c_int,
        s: *mut muDoubleComplex,
        u: *mut muDoubleComplex,
        ldu: c_int,
        v: *mut muDoubleComplex,
        ldv: c_int,
        _e: *mut muDoubleComplex,
        _fast_alg: c_int,
        info: *mut c_int,
        _buffer: *mut c_void,
    ) -> mublasStatus_t {
        if handle.is_null() || s.is_null() || info.is_null() {
            return 1;
        }
        let k = m.min(n) as usize;
        unsafe {
            // S 降序 k..1（mock 确定性数值；真实值由真机 pytest 对照）
            for i in 0..k {
                *s.add(i) = MockNum::from_usize(k - i);
            }
            // U/V 单位阵（left/right_svect 决定列数；NONE 时指针可为 null）
            if !u.is_null() {
                let u_cols = if left_svect == super::MUBLAS_SVECT_ALL {
                    m as usize
                } else {
                    k
                };
                for j in 0..u_cols {
                    for i in 0..m as usize {
                        *u.add(i + j * ldu as usize) = if i == j {
                            MockNum::one()
                        } else {
                            MockNum::zero()
                        };
                    }
                }
            }
            if !v.is_null() {
                let v_cols = if right_svect == super::MUBLAS_SVECT_ALL {
                    n as usize
                } else {
                    k
                };
                for j in 0..v_cols {
                    for i in 0..n as usize {
                        *v.add(i + j * ldv as usize) = if i == j {
                            MockNum::one()
                        } else {
                            MockNum::zero()
                        };
                    }
                }
            }
            *info = 0;
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

    // Phase 4:计算/配置 stub——状态化确定性伪随机(seed 重置/计数器推进),
    // 统计保真(uniform = splitmix64 归一化;normal = mean + stddev·(Σ12u − 6)),
    // 使 mock 模式下 pytest 的形状/复现性/分布统计用例均可运行(无 GPU CI)。
    static MOCK_RNG_STATE: std::sync::Mutex<MockRngState> = std::sync::Mutex::new(MockRngState {
        seed: 0,
        counter: 0,
    });

    struct MockRngState {
        seed: u64,
        counter: u64,
    }

    /// splitmix64 风格确定性混合 → [0,1)（与 seed/counter/序号 均相关）。
    fn mock_rng_uniform(state: &MockRngState, i: usize) -> f64 {
        let mut x = state
            .seed
            .wrapping_add(state.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        (x >> 11) as f64 / ((1u64 << 53) as f64) // [0, 1)
    }

    /// Σ12 均匀 − 6 近似 N(0,1)（中心极限定理,均值 0 方差 1）。
    fn mock_rng_normal(state: &MockRngState, i: usize) -> f64 {
        let mut s = 0.0;
        for k in 0..12 {
            s += mock_rng_uniform(state, i * 12 + k);
        }
        s - 6.0
    }

    pub unsafe fn murandSetPseudoRandomGeneratorSeed(
        generator: murandGenerator_t,
        seed: u64,
    ) -> murandStatus_t {
        if generator.is_null() {
            return 1;
        }
        let mut st = MOCK_RNG_STATE.lock().unwrap();
        st.seed = seed;
        st.counter = 0;
        MURAND_STATUS_SUCCESS
    }

    pub unsafe fn murandSetGeneratorOffset(
        generator: murandGenerator_t,
        offset: u64,
    ) -> murandStatus_t {
        if generator.is_null() {
            return 1;
        }
        MOCK_RNG_STATE.lock().unwrap().counter = offset;
        MURAND_STATUS_SUCCESS
    }

    pub unsafe fn murandGenerateUniform(
        generator: murandGenerator_t,
        output_data: *mut f32,
        n: usize,
    ) -> murandStatus_t {
        if generator.is_null() || output_data.is_null() {
            return 1;
        }
        let mut st = MOCK_RNG_STATE.lock().unwrap();
        let base = st.counter;
        unsafe {
            for i in 0..n {
                *output_data.add(i) = mock_rng_uniform(&st, i) as f32;
            }
        }
        st.counter = base.wrapping_add(1);
        MURAND_STATUS_SUCCESS
    }

    pub unsafe fn murandGenerateUniformDouble(
        generator: murandGenerator_t,
        output_data: *mut f64,
        n: usize,
    ) -> murandStatus_t {
        if generator.is_null() || output_data.is_null() {
            return 1;
        }
        let mut st = MOCK_RNG_STATE.lock().unwrap();
        let base = st.counter;
        unsafe {
            for i in 0..n {
                *output_data.add(i) = mock_rng_uniform(&st, i);
            }
        }
        st.counter = base.wrapping_add(1);
        MURAND_STATUS_SUCCESS
    }

    pub unsafe fn murandGenerateNormal(
        generator: murandGenerator_t,
        output_data: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> murandStatus_t {
        if generator.is_null() || output_data.is_null() {
            return 1;
        }
        let mut st = MOCK_RNG_STATE.lock().unwrap();
        let base = st.counter;
        unsafe {
            for i in 0..n {
                *output_data.add(i) =
                    (mean as f64 + stddev as f64 * mock_rng_normal(&st, i)) as f32;
            }
        }
        st.counter = base.wrapping_add(1);
        MURAND_STATUS_SUCCESS
    }

    pub unsafe fn murandGenerateNormalDouble(
        generator: murandGenerator_t,
        output_data: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> murandStatus_t {
        if generator.is_null() || output_data.is_null() {
            return 1;
        }
        let mut st = MOCK_RNG_STATE.lock().unwrap();
        let base = st.counter;
        unsafe {
            for i in 0..n {
                *output_data.add(i) = mean + stddev * mock_rng_normal(&st, i);
            }
        }
        st.counter = base.wrapping_add(1);
        MURAND_STATUS_SUCCESS
    }

    // ── muFFT ──

    /// mock mufft plan 记录（Exec 数值仿真需要知道长度/方向）。
    /// Plan1d 记 nx；PlanMany 记 nx + batch + istride/ostride/idist/odist。
    /// （Plan2d/3d 记首维 nx；fft.rs 当前只走 Plan1d/PlanMany。）
    struct MockFftPlan {
        nx: i32,
        ftype: mufftType,
        /// 批次数（Plan1d 固定 1；PlanMany 取 batch 参数）。
        batch: i32,
        /// 行内 stride（元素单位；PlanMany 的 istride/ostride）。
        stride: i32,
        /// 行间 dist（元素单位；PlanMany 的 idist/odist）。
        idist: i32,
        odist: i32,
    }

    impl MockFftPlan {
        fn new(nx: i32, ftype: mufftType) -> Self {
            Self {
                nx,
                ftype,
                batch: 1,
                stride: 1,
                idist: nx,
                odist: nx,
            }
        }
    }

    static MOCK_MUFFT_PLANS: LazyLock<MockHandleTable<MockFftPlan>> =
        LazyLock::new(MockHandleTable::new);

    fn register_plan(plan: *mut mufftHandle, nx: i32, ftype: mufftType) {
        unsafe { *plan = next_handle() };
        MOCK_MUFFT_PLANS
            .lock()
            .insert(unsafe { *plan }, MockFftPlan::new(nx, ftype));
    }

    /// naive O(N²) real DFT（R2C/D2Z 用；forward，输出 N//2+1）。
    fn naive_dft_real(input: &[f64]) -> Vec<muDoubleComplex> {
        let n = input.len();
        let out_len = n / 2 + 1;
        let mut out = vec![muDoubleComplex { re: 0.0, im: 0.0 }; out_len];
        for (k, o) in out.iter_mut().enumerate() {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (j, x) in input.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                re += x * angle.cos();
                im += x * angle.sin();
            }
            *o = muDoubleComplex { re, im };
        }
        out
    }

    /// naive O(N²) complex DFT（C2C/Z2Z 用；direction=-1 forward，+1 inverse）。
    fn naive_dft_cplx(input: &[muDoubleComplex], inverse: bool) -> Vec<muDoubleComplex> {
        let n = input.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut out = vec![muDoubleComplex { re: 0.0, im: 0.0 }; n];
        for (k, o) in out.iter_mut().enumerate() {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (j, x) in input.iter().enumerate() {
                let angle = sign * 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                let (c, s) = (angle.cos(), angle.sin());
                re += x.re * c - x.im * s;
                im += x.re * s + x.im * c;
            }
            *o = muDoubleComplex { re, im };
        }
        out
    }

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
        MOCK_MUFFT_PLANS.lock().remove(&plan);
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
        nx: c_int,
        ftype: mufftType,
        _batch: c_int,
    ) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        register_plan(plan, nx, ftype);
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftPlan2d(
        plan: *mut mufftHandle,
        nx: c_int,
        _ny: c_int,
        ftype: mufftType,
    ) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        register_plan(plan, nx, ftype);
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftPlan3d(
        plan: *mut mufftHandle,
        nx: c_int,
        _ny: c_int,
        _nz: c_int,
        ftype: mufftType,
    ) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        register_plan(plan, nx, ftype);
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftPlanMany(
        plan: *mut mufftHandle,
        rank: c_int,
        n: *mut c_int,
        _inembed: *mut c_int,
        istride: c_int,
        idist: c_int,
        _onembed: *mut c_int,
        ostride: c_int,
        odist: c_int,
        ftype: mufftType,
        batch: c_int,
    ) -> mufftResult {
        if plan.is_null() {
            return 1;
        }
        let nx = if rank > 0 && !n.is_null() {
            unsafe { *n }
        } else {
            0
        };
        unsafe { *plan = next_handle() };
        MOCK_MUFFT_PLANS.lock().insert(
            *plan,
            MockFftPlan {
                nx,
                ftype,
                batch,
                stride: istride.max(ostride),
                idist,
                odist,
            },
        );
        MUFFT_SUCCESS
    }

    // ── mock muFFT 执行（naive DFT 数值仿真；P-FFT-1 起支持 PlanMany batch）──

    /// 取 plan 的批量信息：(nx, batch, stride, idist, odist)。
    fn mock_plan_batch(plan: mufftHandle) -> Option<(usize, usize, usize, usize, usize)> {
        let g = MOCK_MUFFT_PLANS.lock();
        let p = g.get(&plan)?;
        Some((
            p.nx as usize,
            p.batch.max(1) as usize,
            p.stride.max(1) as usize,
            p.idist.max(1) as usize,
            p.odist.max(1) as usize,
        ))
    }

    pub unsafe fn mufftExecC2C(
        plan: mufftHandle,
        idata: *mut muComplex,
        odata: *mut muComplex,
        direction: c_int,
    ) -> mufftResult {
        if plan.is_null() || idata.is_null() || odata.is_null() {
            return 1;
        }
        let Some((n, batch, stride, idist, odist)) = mock_plan_batch(plan) else {
            return 1;
        };
        let inverse = direction == MUFFT_INVERSE;
        for row in 0..batch {
            let in_row = unsafe { idata.add(row * idist) };
            let out_row = unsafe { odata.add(row * odist) };
            let mut input: Vec<muDoubleComplex> = Vec::with_capacity(n);
            for i in 0..n {
                let v = unsafe { *in_row.add(i * stride) };
                input.push(muDoubleComplex {
                    re: v.re as f64,
                    im: v.im as f64,
                });
            }
            let vals = naive_dft_cplx(&input, inverse);
            for (i, v) in vals.iter().enumerate() {
                unsafe {
                    *out_row.add(i) = muComplex {
                        re: v.re as f32,
                        im: v.im as f32,
                    };
                }
            }
        }
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftExecR2C(
        plan: mufftHandle,
        idata: *mut f32,
        odata: *mut muComplex,
    ) -> mufftResult {
        if plan.is_null() || idata.is_null() || odata.is_null() {
            return 1;
        }
        let Some((n, batch, stride, idist, odist)) = mock_plan_batch(plan) else {
            return 1;
        };
        for row in 0..batch {
            let in_row = unsafe { idata.add(row * idist) };
            let out_row = unsafe { odata.add(row * odist) };
            let input: Vec<f64> = (0..n)
                .map(|i| unsafe { *in_row.add(i * stride) } as f64)
                .collect();
            let vals = naive_dft_real(&input);
            for (i, v) in vals.iter().enumerate() {
                unsafe {
                    *out_row.add(i) = muComplex {
                        re: v.re as f32,
                        im: v.im as f32,
                    };
                }
            }
        }
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftExecZ2Z(
        plan: mufftHandle,
        idata: *mut muDoubleComplex,
        odata: *mut muDoubleComplex,
        direction: c_int,
    ) -> mufftResult {
        if plan.is_null() || idata.is_null() || odata.is_null() {
            return 1;
        }
        let Some((n, batch, stride, idist, odist)) = mock_plan_batch(plan) else {
            return 1;
        };
        let inverse = direction == MUFFT_INVERSE;
        for row in 0..batch {
            let in_row = unsafe { idata.add(row * idist) };
            let out_row = unsafe { odata.add(row * odist) };
            let input: Vec<muDoubleComplex> =
                (0..n).map(|i| unsafe { *in_row.add(i * stride) }).collect();
            let vals = naive_dft_cplx(&input, inverse);
            for (i, v) in vals.iter().enumerate() {
                unsafe { *out_row.add(i) = *v };
            }
        }
        MUFFT_SUCCESS
    }

    pub unsafe fn mufftExecD2Z(
        plan: mufftHandle,
        idata: *mut f64,
        odata: *mut muDoubleComplex,
    ) -> mufftResult {
        if plan.is_null() || idata.is_null() || odata.is_null() {
            return 1;
        }
        let Some((n, batch, stride, idist, odist)) = mock_plan_batch(plan) else {
            return 1;
        };
        for row in 0..batch {
            let in_row = unsafe { idata.add(row * idist) };
            let out_row = unsafe { odata.add(row * odist) };
            let input: Vec<f64> = (0..n).map(|i| unsafe { *in_row.add(i * stride) }).collect();
            let vals = naive_dft_real(&input);
            for (i, v) in vals.iter().enumerate() {
                unsafe { *out_row.add(i) = *v };
            }
        }
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

    // ── mock muSPARSE 泛型 API（host 端数值仿真，供无 GPU CI 对照 NumPy）──
    // 描述符记录 device 指针（mock 下 buffer 是 host 内存，直接读指针做 CSR 计算）。
    // 销毁统一走 DestroySpMat/DestroyDnVec/DestroyDnMat。

    #[derive(Clone, Copy)]
    struct MockSpMat {
        rows: i64,
        cols: i64,
        nnz: i64,
        row_ptr: *const c_void,
        col_ind: *const c_void,
        vals: *const c_void,
        data_type: musparseDataType_t,
    }

    #[derive(Clone, Copy)]
    struct MockDnVec {
        size: i64,
        vals: *const c_void,
        data_type: musparseDataType_t,
    }

    #[derive(Clone, Copy)]
    struct MockDnMat {
        rows: i64,
        cols: i64,
        ld: i64,
        vals: *const c_void,
        data_type: musparseDataType_t,
    }

    static MOCK_SPMATS: LazyLock<MockHandleTable<MockSpMat>> = LazyLock::new(MockHandleTable::new);
    static MOCK_DNVECS: LazyLock<MockHandleTable<MockDnVec>> = LazyLock::new(MockHandleTable::new);
    static MOCK_DNMATS: LazyLock<MockHandleTable<MockDnMat>> = LazyLock::new(MockHandleTable::new);

    pub unsafe fn musparseCreateCsr(
        descr: *mut musparseSpMatDescr_t,
        rows: i64,
        cols: i64,
        nnz: i64,
        csr_row_ptr: *mut c_void,
        csr_col_ind: *mut c_void,
        csr_val: *mut c_void,
        _row_ptr_type: musparseIndexType_t,
        _col_ind_type: musparseIndexType_t,
        _idx_base: musparseIndexBase_t,
        data_type: musparseDataType_t,
    ) -> musparseStatus_t {
        if descr.is_null() {
            return 1;
        }
        let h = next_handle();
        unsafe {
            *descr = h;
        }
        MOCK_SPMATS.lock().insert(
            h,
            MockSpMat {
                rows,
                cols,
                nnz,
                row_ptr: csr_row_ptr as *const c_void,
                col_ind: csr_col_ind as *const c_void,
                vals: csr_val as *const c_void,
                data_type,
            },
        );
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseDestroySpMat(descr: musparseSpMatDescr_t) -> musparseStatus_t {
        MOCK_SPMATS.lock().remove(&descr);
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseCreateDnVec(
        descr: *mut musparseDnVecDescr_t,
        size: i64,
        values: *mut c_void,
        data_type: musparseDataType_t,
    ) -> musparseStatus_t {
        if descr.is_null() {
            return 1;
        }
        let h = next_handle();
        unsafe {
            *descr = h;
        }
        MOCK_DNVECS.lock().insert(
            h,
            MockDnVec {
                size,
                vals: values as *const c_void,
                data_type,
            },
        );
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseDestroyDnVec(descr: musparseDnVecDescr_t) -> musparseStatus_t {
        MOCK_DNVECS.lock().remove(&descr);
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseCreateDnMat(
        descr: *mut musparseDnMatDescr_t,
        rows: i64,
        cols: i64,
        ld: i64,
        values: *mut c_void,
        data_type: musparseDataType_t,
        _order: musparseOrder_t,
    ) -> musparseStatus_t {
        if descr.is_null() {
            return 1;
        }
        let h = next_handle();
        unsafe {
            *descr = h;
        }
        MOCK_DNMATS.lock().insert(
            h,
            MockDnMat {
                rows,
                cols,
                ld,
                vals: values as *const c_void,
                data_type,
            },
        );
        MUSPARSE_STATUS_SUCCESS
    }

    pub unsafe fn musparseDestroyDnMat(descr: musparseDnMatDescr_t) -> musparseStatus_t {
        MOCK_DNMATS.lock().remove(&descr);
        MUSPARSE_STATUS_SUCCESS
    }

    /// mock SpMV：y = alpha·A·x + beta·y（host CSR 计算，f32/f64）。
    /// temp_buffer=NULL 时写 0 到 buffer_size 并返回（两段式查询阶段）。
    pub unsafe fn musparseSpMV(
        _handle: musparseHandle_t,
        trans: musparseOperation_t,
        alpha: *const c_void,
        mat: musparseSpMatDescr_t,
        x: musparseDnVecDescr_t,
        beta: *const c_void,
        y: musparseDnVecDescr_t,
        compute_type: musparseDataType_t,
        _alg: musparseSpMVAlg_t,
        buffer_size: *mut usize,
        temp_buffer: *mut c_void,
    ) -> musparseStatus_t {
        if buffer_size.is_null() {
            return 1;
        }
        if temp_buffer.is_null() {
            // 查询阶段：写所需字节数（mock 用 0，真实 SDK 可能非零）
            unsafe { *buffer_size = 0 };
            return MUSPARSE_STATUS_SUCCESS;
        }
        let (m, xv, yv) = {
            let g = MOCK_SPMATS.lock();
            let s = match g.get(&mat) {
                Some(s) => *s,
                None => return 1,
            };
            let xv = match MOCK_DNVECS.lock().get(&x) {
                Some(v) => *v,
                None => return 1,
            };
            let yv = match MOCK_DNVECS.lock().get(&y) {
                Some(v) => *v,
                None => return 1,
            };
            (s, xv, yv)
        };
        if compute_type == MUSA_R_32F {
            mock_spmv_f64::<f32>(&m, &xv, &yv, trans, alpha, beta);
        } else {
            mock_spmv_f64::<f64>(&m, &xv, &yv, trans, alpha, beta);
        }
        MUSPARSE_STATUS_SUCCESS
    }

    /// 泛型 host CSR spmv（T=f32/f64；alpha/beta 按 host f64 传入）。
    fn mock_spmv_f64<T: CsrVal>(
        m: &MockSpMat,
        xv: &MockDnVec,
        yv: &MockDnVec,
        trans: musparseOperation_t,
        alpha: *const c_void,
        beta: *const c_void,
    ) {
        let rows = m.rows as usize;
        let cols = m.cols as usize;
        let row_ptr = m.row_ptr as *const i64;
        let col_ind = m.col_ind as *const i32;
        let vals = m.vals as *const T;
        let x = xv.vals as *const T;
        let y = yv.vals as *mut T;
        let alpha_v = unsafe { *(alpha as *const f64) };
        let beta_v = unsafe { *(beta as *const f64) };
        if trans == MUSPARSE_OPERATION_NON_TRANSPOSE {
            for i in 0..rows {
                let mut acc: f64 = 0.0;
                let start = unsafe { *row_ptr.add(i) } as usize;
                let end = unsafe { *row_ptr.add(i + 1) } as usize;
                for k in start..end {
                    let j = unsafe { *col_ind.add(k) } as usize;
                    let v = unsafe { *vals.add(k) };
                    acc += v.to_f64() * unsafe { *x.add(j) }.to_f64();
                }
                let yi = unsafe { &mut *y.add(i) };
                *yi = T::from_f64(alpha_v * acc + beta_v * yi.to_f64());
            }
        } else {
            // 转置：x 长度 = rows，输出 cols
            for j in 0..cols {
                let mut acc: f64 = 0.0;
                for i in 0..rows {
                    let start = unsafe { *row_ptr.add(i) } as usize;
                    let end = unsafe { *row_ptr.add(i + 1) } as usize;
                    for k in start..end {
                        if unsafe { *col_ind.add(k) } as usize == j {
                            acc += unsafe { *vals.add(k) }.to_f64() * unsafe { *x.add(i) }.to_f64();
                        }
                    }
                }
                let yj = unsafe { &mut *y.add(j) };
                *yj = T::from_f64(alpha_v * acc + beta_v * yj.to_f64());
            }
        }
    }

    /// mock SpMM：C = alpha·A·B + beta·C（host CSR × 稠密行主序）。
    pub unsafe fn musparseSpMM(
        _handle: musparseHandle_t,
        _trans_A: musparseOperation_t,
        _trans_B: musparseOperation_t,
        alpha: *const c_void,
        mat_A: musparseSpMatDescr_t,
        mat_B: musparseDnMatDescr_t,
        beta: *const c_void,
        mat_C: musparseDnMatDescr_t,
        compute_type: musparseDataType_t,
        _alg: musparseSpMMAlg_t,
        _stage: musparseSpMMStage_t,
        buffer_size: *mut usize,
        temp_buffer: *mut c_void,
    ) -> musparseStatus_t {
        if buffer_size.is_null() {
            return 1;
        }
        if temp_buffer.is_null() {
            unsafe { *buffer_size = 0 };
            return MUSPARSE_STATUS_SUCCESS;
        }
        let (m, b, c) = {
            let g = MOCK_SPMATS.lock();
            let s = match g.get(&mat_A) {
                Some(s) => *s,
                None => return 1,
            };
            let b = match MOCK_DNMATS.lock().get(&mat_B) {
                Some(b) => *b,
                None => return 1,
            };
            let c = match MOCK_DNMATS.lock().get(&mat_C) {
                Some(c) => *c,
                None => return 1,
            };
            (s, b, c)
        };
        if compute_type == MUSA_R_32F {
            mock_spmm_host::<f32>(&m, &b, &c, alpha, beta);
        } else {
            mock_spmm_host::<f64>(&m, &b, &c, alpha, beta);
        }
        MUSPARSE_STATUS_SUCCESS
    }

    /// 泛型 host CSR × 稠密（T=f32/f64；B/C 行主序 ld=cols）。
    fn mock_spmm_host<T: CsrVal>(
        m: &MockSpMat,
        b: &MockDnMat,
        c: &MockDnMat,
        alpha: *const c_void,
        beta: *const c_void,
    ) {
        let rows = m.rows as usize;
        let bcols = b.cols as usize;
        let row_ptr = m.row_ptr as *const i64;
        let col_ind = m.col_ind as *const i32;
        let vals = m.vals as *const T;
        let bv = b.vals as *const T;
        let cv = c.vals as *mut T;
        let alpha_v = unsafe { *(alpha as *const f64) };
        let beta_v = unsafe { *(beta as *const f64) };
        for i in 0..rows {
            for jj in 0..bcols {
                let mut acc: f64 = 0.0;
                let start = unsafe { *row_ptr.add(i) } as usize;
                let end = unsafe { *row_ptr.add(i + 1) } as usize;
                for k in start..end {
                    let j = unsafe { *col_ind.add(k) } as usize;
                    acc += unsafe { *vals.add(k) }.to_f64()
                        * unsafe { *bv.add(j * bcols + jj) }.to_f64();
                }
                let ci = unsafe { &mut *cv.add(i * bcols + jj) };
                *ci = T::from_f64(alpha_v * acc + beta_v * ci.to_f64());
            }
        }
    }

    /// mock 标量转换 trait（f32/f64 复用）。
    trait CsrVal: Copy {
        fn to_f64(self) -> f64;
        fn from_f64(x: f64) -> Self;
    }

    impl CsrVal for f32 {
        fn to_f64(self) -> f64 {
            self as f64
        }
        fn from_f64(x: f64) -> f32 {
            x as f32
        }
    }

    impl CsrVal for f64 {
        fn to_f64(self) -> f64 {
            self
        }
        fn from_f64(x: f64) -> f64 {
            x
        }
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
    fn test_murand_generate_and_seed() {
        // Phase 4 stub：seed 重置 → 同 seed 两次生成逐元素相等；
        // uniform 值域 [0,1)；normal 的 mean/stddev 传递生效。
        let mut g: murandGenerator_t = std::ptr::null_mut();
        assert_eq!(
            unsafe { murandCreateGenerator(&mut g, MURAND_RNG_PSEUDO_DEFAULT) },
            MURAND_STATUS_SUCCESS
        );
        assert_eq!(
            unsafe { murandSetPseudoRandomGeneratorSeed(g, 42) },
            MURAND_STATUS_SUCCESS
        );
        assert_eq!(
            unsafe { murandSetGeneratorOffset(g, 7) },
            MURAND_STATUS_SUCCESS
        );

        // seed 重置 + 计数器推进：同 seed 紧邻两次调用逐元素相等
        let mut d1 = [0.0f32; 4];
        let mut d2 = [0.0f32; 4];
        assert_eq!(
            unsafe { murandSetPseudoRandomGeneratorSeed(g, 42) },
            MURAND_STATUS_SUCCESS
        );
        assert_eq!(
            unsafe { murandGenerateUniform(g, d1.as_mut_ptr(), d1.len()) },
            MURAND_STATUS_SUCCESS
        );
        assert_eq!(
            unsafe { murandSetPseudoRandomGeneratorSeed(g, 42) },
            MURAND_STATUS_SUCCESS
        );
        assert_eq!(
            unsafe { murandGenerateUniform(g, d2.as_mut_ptr(), d2.len()) },
            MURAND_STATUS_SUCCESS
        );
        assert_eq!(d1, d2, "same seed must reproduce the same sequence");
        assert!(d1.iter().all(|&v| (0.0..1.0).contains(&v)));

        // 不同 seed → 不同序列
        let mut d3 = [0.0f32; 4];
        assert_eq!(
            unsafe { murandSetPseudoRandomGeneratorSeed(g, 43) },
            MURAND_STATUS_SUCCESS
        );
        assert_eq!(
            unsafe { murandGenerateUniform(g, d3.as_mut_ptr(), d3.len()) },
            MURAND_STATUS_SUCCESS
        );
        assert_ne!(d1, d3, "different seeds must differ");

        let mut d64 = [0.0f64; 3];
        assert_eq!(
            unsafe { murandGenerateUniformDouble(g, d64.as_mut_ptr(), d64.len()) },
            MURAND_STATUS_SUCCESS
        );
        assert!(d64.iter().all(|&v| (0.0..1.0).contains(&v)));

        // normal：mean/stddev 传递（mean=3,stddev=2 的样本应围绕 3 分布）
        let mut norm = [0.0f32; 64];
        assert_eq!(
            unsafe { murandGenerateNormal(g, norm.as_mut_ptr(), norm.len(), 3.0, 2.0) },
            MURAND_STATUS_SUCCESS
        );
        let mean: f32 = norm.iter().sum::<f32>() / norm.len() as f32;
        assert!((mean - 3.0).abs() < 0.5, "normal mean drift: {mean}");
        let mut norm64 = [0.0f64; 64];
        assert_eq!(
            unsafe { murandGenerateNormalDouble(g, norm64.as_mut_ptr(), norm64.len(), -1.0, 0.5) },
            MURAND_STATUS_SUCCESS
        );
        let mean64: f64 = norm64.iter().sum::<f64>() / norm64.len() as f64;
        assert!((mean64 - (-1.0)).abs() < 0.2, "normal mean drift: {mean64}");

        // 空指针守卫
        assert_ne!(
            unsafe { murandGenerateUniform(g, std::ptr::null_mut(), 4) },
            MURAND_STATUS_SUCCESS
        );
        assert_ne!(
            unsafe { murandSetPseudoRandomGeneratorSeed(std::ptr::null_mut(), 1) },
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
    fn test_musolver_phase3_mock() {
        let mut h: mublasHandle_t = std::ptr::null_mut();
        assert_eq!(unsafe { mublasCreate(&mut h) }, MUBLAS_STATUS_SUCCESS);

        // bufferSize：无 handle 参数，返回 4096
        let mut bs: c_int = 0;
        assert_eq!(
            unsafe { musolverDgeqrf_bufferSize(4, 3, &mut bs) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(bs, 4096);
        assert_eq!(
            unsafe { musolverDorgqr_bufferSize(4, 3, 3, &mut bs) },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(
            unsafe {
                musolverDgesvd_bufferSize(
                    MUBLAS_SVECT_ALL,
                    MUBLAS_SVECT_ALL,
                    4,
                    3,
                    1,
                    MUBLAS_OUTOFPLACE,
                    &mut bs,
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(bs, 4096);

        // geqrf：A 保持原样，tau 置 0
        let mut a = [1.0f64; 12];
        let mut tau = [9.9f64; 3];
        assert_eq!(
            unsafe {
                musolverDgeqrf(
                    h,
                    4,
                    3,
                    a.as_mut_ptr(),
                    4,
                    tau.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(a, [1.0; 12]);
        assert_eq!(tau, [0.0; 3]);

        // orgqr：缓冲原样返回
        assert_eq!(
            unsafe {
                musolverDorgqr(
                    h,
                    4,
                    3,
                    3,
                    a.as_mut_ptr(),
                    4,
                    tau.as_ptr(),
                    std::ptr::null_mut(),
                )
            },
            MUBLAS_STATUS_SUCCESS
        );

        // gesvd：S 降序 k..1，U/V 单位阵，info=0
        let mut s = [0.0f64; 3];
        let mut u = [0.0f64; 16]; // m×m=4×4
        let mut v = [0.0f64; 9]; // n×n=3×3
        let mut e = [0.0f64; 4];
        let mut info: c_int = -1;
        assert_eq!(
            unsafe {
                musolverDgesvd(
                    h,
                    MUBLAS_SVECT_ALL,
                    MUBLAS_SVECT_ALL,
                    4,
                    3,
                    a.as_mut_ptr(),
                    4,
                    s.as_mut_ptr(),
                    u.as_mut_ptr(),
                    4,
                    v.as_mut_ptr(),
                    3,
                    e.as_mut_ptr(),
                    MUBLAS_OUTOFPLACE,
                    &mut info,
                    std::ptr::null_mut(),
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(s, [3.0, 2.0, 1.0]); // 降序
        assert_eq!(info, 0);
        // U 单位阵（4×4 列主序直读 U[i + j*ldu]）
        for j in 0..4 {
            for i in 0..4 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert_eq!(u[i + j * 4], expect, "U[{i}][{j}]");
            }
        }
        // V 单位阵（3×3）
        for j in 0..3 {
            for i in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert_eq!(v[i + j * 3], expect, "V[{i}][{j}]");
            }
        }

        // NONE 模式：U/V 可为 null
        let mut s2 = [0.0f64; 3];
        let mut info2: c_int = -1;
        assert_eq!(
            unsafe {
                musolverDgesvd(
                    h,
                    MUBLAS_SVECT_NONE,
                    MUBLAS_SVECT_NONE,
                    4,
                    3,
                    a.as_mut_ptr(),
                    4,
                    s2.as_mut_ptr(),
                    std::ptr::null_mut(),
                    4,
                    std::ptr::null_mut(),
                    3,
                    e.as_mut_ptr(),
                    MUBLAS_OUTOFPLACE,
                    &mut info2,
                    std::ptr::null_mut(),
                )
            },
            MUBLAS_STATUS_SUCCESS
        );
        assert_eq!(s2, [3.0, 2.0, 1.0]);

        assert_eq!(unsafe { mublasDestroy(h) }, MUBLAS_STATUS_SUCCESS);
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
