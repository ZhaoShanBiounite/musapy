//! 稀疏矩阵算子（v0.3-alpha Phase 6，ADR-003 003-D4/D7）
//!
//! `ms.sparse.csr_matrix` + `spmv`/`spmm`/`toarray`，**GPU-only**
//! （003-D4 修订：数学库算子不建 CPU fallback，CPU 设备抛 DeviceError）。
//! 实现走 muSPARSE 泛型 API（musparseCreateCsr + CreateDnVec/DnMat + SpMV/SpMM）。
//!
//! 本轮范围（用户确认，2026-08-08）：
//!   - 只做 `csr_matrix`（data/indices/indptr 3 个 device buffer）；`coo_matrix`
//!     与 coo→csr 归并推迟到 v0.3 后期。
//!   - `@` 支持：csr @ vec（spmv，1D）/ csr @ dense（spmm，2D）。
//!   - `toarray()` 物化稠密 Array（正确性优先：D2H → host 构建 → H2D）。
//!
//! 数值语义（对齐 SciPy / NumPy）：
//!   - spmv/spmm 用 host 标量 alpha=1, beta=0。
//!   - mock 模式：musparse stub 用 host CSR 循环数值仿真（无 GPU CI 对照）。

use musapy_core::math_handle;
use musapy_core::musa_ffi;
use musapy_core::resolution;
use musapy_core::{
    Array, Buffer, BufferRef, Device, Dtype, Layout, Result, Stream, musa_x_ffi,
};
use musapy_core::musa_x_ffi::{
    MUSA_R_32F, MUSA_R_64F, MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO,
    MUSPARSE_OPERATION_NON_TRANSPOSE, MUSPARSE_ORDER_ROW, MUSPARSE_SPMM_ALG_DEFAULT,
    MUSPARSE_SPMM_STAGE_AUTO, MUSPARSE_SPMV_ALG_DEFAULT,
};
use std::sync::Arc;

use crate::linalg::require_musa;

/// CSR 稀疏矩阵（GPU-only）。
///
/// 持有 3 个 device buffer：data（f32/f64 nnz 个）、indices（i32 nnz 个列索引）、
/// indptr（i32 (rows+1) 个行偏移）。0-based 索引（MUSPARSE_INDEX_BASE_ZERO）。
#[derive(Clone)]
pub struct CsrMatrix {
    shape: (usize, usize),
    dtype: Dtype,
    device: Device,
    data: BufferRef,
    indices: BufferRef,
    indptr: BufferRef,
    nnz: usize,
    stream: Arc<Stream>,
}

impl CsrMatrix {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }
}

/// 从 host bytes 构造 CSR（H2D 到 device buffer）。
///
/// 参数均为提取好的原始字节：data（nnz×elem）、indices（nnz×4，i32）、
/// indptr（(rows+1)×4，i32）。shape 由调用方校验。
#[allow(clippy::too_many_arguments)]
pub fn csr_from_host(
    data_bytes: &[u8],
    indices_bytes: &[u8],
    indptr_bytes: &[u8],
    rows: usize,
    cols: usize,
    nnz: usize,
    dtype: Dtype,
    device: Device,
    stream: &Arc<Stream>,
) -> Result<CsrMatrix> {
    // dtype 白名单（f32/f64；complex 推迟）
    match dtype {
        Dtype::Float32 | Dtype::Float64 => {}
        _ => {
            return Err(musapy_core::error::DtypeError::Unsupported(format!(
                "csr_matrix: dtype {} not supported (whitelist: float32/float64)",
                dtype
            ))
            .into());
        }
    }

    // 字节数校验（data 长度 = nnz×elem；indices = nnz×4；indptr = (rows+1)×4）
    let elem = dtype.element_size();
    if data_bytes.len() != nnz * elem {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "csr_matrix: data length {} != nnz {} × {} bytes",
            data_bytes.len(),
            nnz,
            elem
        ))
        .into());
    }
    if indices_bytes.len() != nnz * 4 {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "csr_matrix: indices length {} != nnz {} × 4 bytes",
            indices_bytes.len(),
            nnz
        ))
        .into());
    }
    if indptr_bytes.len() != (rows + 1) * 4 {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "csr_matrix: indptr length {} != (rows+1) {} × 4 bytes",
            indptr_bytes.len(),
            rows + 1
        ))
        .into());
    }

    // H2D 拷贝（仿 ms.array 的 copy_to_buffer 模式）
    let alloc_h2d = |bytes: &[u8]| -> Result<BufferRef> {
        let buffer = Buffer::alloc(bytes.len(), device.clone(), stream)?;
        let data_ref = BufferRef::new(Arc::new(buffer));
        let dst = data_ref.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "csr_matrix: null buffer pointer".into(),
                ),
            )
        })?;
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaMemcpy(
                    dst.as_ptr() as *mut std::ffi::c_void,
                    bytes.as_ptr() as *const std::ffi::c_void,
                    bytes.len(),
                    musa_ffi::musaMemcpyKind::HostToDevice,
                ),
                "csr_matrix H2D",
            )?;
        }
        Ok(data_ref)
    };

    let data = alloc_h2d(data_bytes)?;
    let indices = alloc_h2d(indices_bytes)?;
    let indptr = alloc_h2d(indptr_bytes)?;

    Ok(CsrMatrix {
        shape: (rows, cols),
        dtype,
        device,
        data,
        indices,
        indptr,
        nnz,
        stream: Arc::clone(stream),
    })
}

/// 从 3 个 device Array（data/indices/indptr）构造 CSR（零拷贝借用 buffer）。
///
/// 要求：data dtype f32/f64，indices/indptr dtype int32（musparse INDEX_32I）。
/// shape 由调用方校验；三个 Array 须在 mat.device 上且已连续。
pub fn csr_from_arrays(
    data: &Array,
    indices: &Array,
    indptr: &Array,
    rows: usize,
    cols: usize,
) -> Result<CsrMatrix> {
    // GPU-only（003-D4）：CPU 设备上构造 CSR 抛 DeviceError
    require_musa("csr_matrix", data.device())?;
    // dtype 白名单
    match data.dtype() {
        Dtype::Float32 | Dtype::Float64 => {}
        _ => {
            return Err(musapy_core::error::DtypeError::Unsupported(format!(
                "csr_matrix: data dtype {} not supported (whitelist: float32/float64)",
                data.dtype()
            ))
            .into());
        }
    }
    if indices.dtype() != Dtype::Int32 || indptr.dtype() != Dtype::Int32 {
        return Err(musapy_core::error::DtypeError::Unsupported(
            "csr_matrix: indices/indptr must be int32 (use dtype=ms.int32)".into(),
        )
        .into());
    }
    let nnz = data.shape()[0];
    if indices.shape()[0] != nnz {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "csr_matrix: indices length {} != data length {}",
            indices.shape()[0],
            nnz
        ))
        .into());
    }
    if indptr.shape()[0] != rows + 1 {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "csr_matrix: indptr length {} != rows+1 {}",
            indptr.shape()[0],
            rows + 1
        ))
        .into());
    }
    // 设备一致
    if data.device() != indices.device() || data.device() != indptr.device() {
        return Err(musapy_core::error::DeviceError::Mismatch(
            "csr_matrix: data/indices/indptr device mismatch".into(),
        )
        .into());
    }

    Ok(CsrMatrix {
        shape: (rows, cols),
        dtype: data.dtype(),
        device: data.device().clone(),
        data: data.data().clone(),
        indices: indices.data().clone(),
        indptr: indptr.data().clone(),
        nnz,
        stream: Arc::clone(data.stream()),
    })
}

/// `spmv(mat, vec)` — csr @ 1D 向量（m×n @ n → m）。GPU-only。
pub fn spmv(mat: &CsrMatrix, vec: &Array) -> Result<Array> {
    let op_name = "spmv";

    // Device 校验 + GPU-only（003-D4）
    require_musa(op_name, &mat.device)?;
    if vec.device() != &mat.device {
        return Err(musapy_core::error::DeviceError::Mismatch(format!(
            "{}: device mismatch {} vs {}",
            op_name,
            vec.device(),
            mat.device
        ))
        .into());
    }

    // shape 校验：vec 必须 1D，长度 = cols
    if vec.shape().len() != 1 {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "spmv: vec must be 1D (got {:?})",
            vec.shape()
        ))
        .into());
    }
    let (rows, cols) = mat.shape;
    if vec.shape()[0] != cols {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "spmv: vec length {} != cols {}",
            vec.shape()[0],
            cols
        ))
        .into());
    }

    // dtype 匹配
    if vec.dtype() != mat.dtype {
        return Err(musapy_core::error::DtypeError::Unsupported(format!(
            "spmv: vec dtype {} != mat dtype {}",
            vec.dtype(),
            mat.dtype
        ))
        .into());
    }

    // Stream（沿用 mat 的 stream）
    let out_stream: Arc<Stream> =
        resolution::get_current_stream().unwrap_or_else(|| Arc::clone(&mat.stream));

    // nnz==0：输出全零（0 字节 buffer 无 ptr，早退不走 musparse）
    if mat.nnz == 0 {
        let out_shape = vec![rows];
        let nbytes = rows * mat.dtype.element_size();
        let out_buffer = Buffer::alloc(nbytes, mat.device.clone(), &out_stream)?;
        let out_ref = BufferRef::new(Arc::new(out_buffer));
        fill_zeros(
            out_ref.buffer().ptr(),
            nbytes,
            &mat.device,
            &out_stream,
        )?;
        out_ref.buffer().record_write(&out_stream);
        return Ok(Array::new(
            out_ref,
            Layout::from_shape(out_shape),
            mat.dtype,
            out_stream,
            musapy_core::DeviceResolution::new(
                mat.device.clone(),
                musapy_core::ResolutionSource::InputArray,
            ),
            musapy_core::DtypeResolution::new(mat.dtype, musapy_core::ResolutionSource::InputArray),
        ));
    }

    // vec 连续化
    let vec_contig = crate::indexing::contiguous(vec)?;

    // 输出 buffer（m 个元素）
    let out_nbytes = rows * mat.dtype.element_size();
    let out_buffer = Buffer::alloc(out_nbytes, mat.device.clone(), &out_stream)?;
    let out_ref = BufferRef::new(Arc::new(out_buffer));
    let out_ptr = out_ref.buffer().ptr().ok_or_else(|| {
        musapy_core::error::MusapyError::Device(
            musapy_core::error::DeviceError::MathLibCallFailed(
                "spmv: null output pointer".into(),
            ),
        )
    })?;

    // stream 依赖
    vec_contig.data().buffer().wait_last_write_on(&out_stream)?;
    mat.data.buffer().wait_last_write_on(&out_stream)?;
    mat.indices.buffer().wait_last_write_on(&out_stream)?;
    mat.indptr.buffer().wait_last_write_on(&out_stream)?;

    // ── muSPARSE 执行（两段式）──
    let dtype_code = match mat.dtype {
        Dtype::Float32 => MUSA_R_32F,
        Dtype::Float64 => MUSA_R_64F,
        _ => unreachable!("dtype whitelisted"),
    };
    // alpha/beta 须按 compute_type 宽度传标量（f32 传 f32 指针，f64 传 f64 指针；
    // 传错宽度会读错字节 → 输出全零）。
    let alpha_f32 = 1.0f32;
    let beta_f32 = 0.0f32;
    let alpha_f64 = 1.0f64;
    let beta_f64 = 0.0f64;
    let (rows_i, cols_i, nnz_i) = (
        rows as i64,
        cols as i64,
        mat.nnz as i64,
    );

    let result = math_handle::with_musparse_handle(&mat.device, &out_stream, |handle| {
        // 描述符创建
        let mut spmat: musa_x_ffi::musparseSpMatDescr_t = std::ptr::null_mut();
        let mut vec_x: musa_x_ffi::musparseDnVecDescr_t = std::ptr::null_mut();
        let mut vec_y: musa_x_ffi::musparseDnVecDescr_t = std::ptr::null_mut();

        let vx_ptr = vec_contig.data().buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "spmv: null vec pointer".into(),
                ),
            )
        })?;
        let indptr_ptr = mat.indptr.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "spmv: null indptr pointer".into(),
                ),
            )
        })?;
        let indices_ptr = mat.indices.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "spmv: null indices pointer".into(),
                ),
            )
        })?;
        let data_ptr = mat.data.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "spmv: null data pointer".into(),
                ),
            )
        })?;

        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseCreateCsr(
                    &mut spmat,
                    rows_i,
                    cols_i,
                    nnz_i,
                    indptr_ptr.as_ptr() as *mut std::ffi::c_void,
                    indices_ptr.as_ptr() as *mut std::ffi::c_void,
                    data_ptr.as_ptr() as *mut std::ffi::c_void,
                    MUSPARSE_INDEX_32I,
                    MUSPARSE_INDEX_32I,
                    MUSPARSE_INDEX_BASE_ZERO,
                    dtype_code,
                )
            },
            "musparseCreateCsr",
        )?;
        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseCreateDnVec(
                    &mut vec_x,
                    cols_i,
                    vx_ptr.as_ptr() as *mut std::ffi::c_void,
                    dtype_code,
                )
            },
            "musparseCreateDnVec(x)",
        )?;
        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseCreateDnVec(
                    &mut vec_y,
                    rows_i,
                    out_ptr.as_ptr() as *mut std::ffi::c_void,
                    dtype_code,
                )
            },
            "musparseCreateDnVec(y)",
        )?;

        // 两段式：查询 → workspace → 计算
        // alpha/beta 指针按 dtype 选宽度（host 标量模式）
        let (alpha_ptr, beta_ptr): (*const std::ffi::c_void, *const std::ffi::c_void) =
            match mat.dtype {
                Dtype::Float32 => (
                    &alpha_f32 as *const f32 as *const std::ffi::c_void,
                    &beta_f32 as *const f32 as *const std::ffi::c_void,
                ),
                Dtype::Float64 => (
                    &alpha_f64 as *const f64 as *const std::ffi::c_void,
                    &beta_f64 as *const f64 as *const std::ffi::c_void,
                ),
                _ => unreachable!("dtype whitelisted"),
            };
        let mut buf_size: usize = 0;
        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseSpMV(
                    handle,
                    MUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha_ptr,
                    spmat,
                    vec_x,
                    beta_ptr,
                    vec_y,
                    dtype_code,
                    MUSPARSE_SPMV_ALG_DEFAULT,
                    &mut buf_size,
                    std::ptr::null_mut(),
                )
            },
            "musparseSpMV(query)",
        )?;
        let ws = math_handle::get_workspace(&mat.device, buf_size)?;
        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseSpMV(
                    handle,
                    MUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha_ptr,
                    spmat,
                    vec_x,
                    beta_ptr,
                    vec_y,
                    dtype_code,
                    MUSPARSE_SPMV_ALG_DEFAULT,
                    &mut buf_size,
                    ws.ptr(),
                )
            },
            "musparseSpMV",
        )?;

        // 描述符销毁
        let _ = unsafe { musa_x_ffi::musparseDestroyDnVec(vec_y) };
        let _ = unsafe { musa_x_ffi::musparseDestroyDnVec(vec_x) };
        let _ = unsafe { musa_x_ffi::musparseDestroySpMat(spmat) };
        Ok(())
    });

    result?;

    // 后处理
    vec_contig.data().buffer().record_read(&out_stream);
    out_ref.buffer().record_write(&out_stream);

    let out_shape = vec![rows];
    Ok(Array::new(
        out_ref,
        Layout::from_shape(out_shape),
        mat.dtype,
        out_stream,
        musapy_core::DeviceResolution::new(
            mat.device.clone(),
            musapy_core::ResolutionSource::InputArray,
        ),
        musapy_core::DtypeResolution::new(mat.dtype, musapy_core::ResolutionSource::InputArray),
    ))
}

/// `spmm(mat, dense)` — csr @ 2D 矩阵（m×n @ n×k → m×k）。GPU-only。
pub fn spmm(mat: &CsrMatrix, dense: &Array) -> Result<Array> {
    let op_name = "spmm";

    // Device 校验 + GPU-only（003-D4）
    require_musa(op_name, &mat.device)?;
    if dense.device() != &mat.device {
        return Err(musapy_core::error::DeviceError::Mismatch(format!(
            "{}: device mismatch {} vs {}",
            op_name,
            dense.device(),
            mat.device
        ))
        .into());
    }

    // shape 校验：dense 必须 2D，rows = cols
    if dense.shape().len() != 2 {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "spmm: dense must be 2D (got {:?})",
            dense.shape()
        ))
        .into());
    }
    let (rows, cols) = mat.shape;
    let (dr, dc) = (dense.shape()[0], dense.shape()[1]);
    if dr != cols {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "spmm: dense rows {} != cols {}",
            dr,
            cols
        ))
        .into());
    }

    // dtype 匹配
    if dense.dtype() != mat.dtype {
        return Err(musapy_core::error::DtypeError::Unsupported(format!(
            "spmm: dense dtype {} != mat dtype {}",
            dense.dtype(),
            mat.dtype
        ))
        .into());
    }

    let out_stream: Arc<Stream> =
        resolution::get_current_stream().unwrap_or_else(|| Arc::clone(&mat.stream));

    // nnz==0：输出全零（早退不走 musparse）
    if mat.nnz == 0 {
        let out_shape = vec![rows, dc];
        let nbytes = rows * dc * mat.dtype.element_size();
        let out_buffer = Buffer::alloc(nbytes, mat.device.clone(), &out_stream)?;
        let out_ref = BufferRef::new(Arc::new(out_buffer));
        fill_zeros(out_ref.buffer().ptr(), nbytes, &mat.device, &out_stream)?;
        out_ref.buffer().record_write(&out_stream);
        return Ok(Array::new(
            out_ref,
            Layout::from_shape(out_shape),
            mat.dtype,
            out_stream,
            musapy_core::DeviceResolution::new(
                mat.device.clone(),
                musapy_core::ResolutionSource::InputArray,
            ),
            musapy_core::DtypeResolution::new(mat.dtype, musapy_core::ResolutionSource::InputArray),
        ));
    }

    // dense 连续化
    let dense_contig = crate::indexing::contiguous(dense)?;

    // 输出 buffer（m×k）
    let out_nbytes = rows * dc * mat.dtype.element_size();
    let out_buffer = Buffer::alloc(out_nbytes, mat.device.clone(), &out_stream)?;
    let out_ref = BufferRef::new(Arc::new(out_buffer));
    let out_ptr = out_ref.buffer().ptr().ok_or_else(|| {
        musapy_core::error::MusapyError::Device(
            musapy_core::error::DeviceError::MathLibCallFailed(
                "spmm: null output pointer".into(),
            ),
        )
    })?;

    dense_contig.data().buffer().wait_last_write_on(&out_stream)?;
    mat.data.buffer().wait_last_write_on(&out_stream)?;
    mat.indices.buffer().wait_last_write_on(&out_stream)?;
    mat.indptr.buffer().wait_last_write_on(&out_stream)?;

    let dtype_code = match mat.dtype {
        Dtype::Float32 => MUSA_R_32F,
        Dtype::Float64 => MUSA_R_64F,
        _ => unreachable!("dtype whitelisted"),
    };
    // alpha/beta 按 dtype 宽度传标量（同 spmv）
    let alpha_f32 = 1.0f32;
    let beta_f32 = 0.0f32;
    let alpha_f64 = 1.0f64;
    let beta_f64 = 0.0f64;
    let (rows_i, cols_i, nnz_i, dc_i) = (
        rows as i64,
        cols as i64,
        mat.nnz as i64,
        dc as i64,
    );

    let result = math_handle::with_musparse_handle(&mat.device, &out_stream, |handle| {
        let mut spmat: musa_x_ffi::musparseSpMatDescr_t = std::ptr::null_mut();
        let mut dnb: musa_x_ffi::musparseDnMatDescr_t = std::ptr::null_mut();
        let mut dnc: musa_x_ffi::musparseDnMatDescr_t = std::ptr::null_mut();

        let db_ptr = dense_contig.data().buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "spmm: null dense pointer".into(),
                ),
            )
        })?;
        let indptr_ptr = mat.indptr.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "spmm: null indptr pointer".into(),
                ),
            )
        })?;
        let indices_ptr = mat.indices.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "spmm: null indices pointer".into(),
                ),
            )
        })?;
        let data_ptr = mat.data.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "spmm: null data pointer".into(),
                ),
            )
        })?;

        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseCreateCsr(
                    &mut spmat,
                    rows_i,
                    cols_i,
                    nnz_i,
                    indptr_ptr.as_ptr() as *mut std::ffi::c_void,
                    indices_ptr.as_ptr() as *mut std::ffi::c_void,
                    data_ptr.as_ptr() as *mut std::ffi::c_void,
                    MUSPARSE_INDEX_32I,
                    MUSPARSE_INDEX_32I,
                    MUSPARSE_INDEX_BASE_ZERO,
                    dtype_code,
                )
            },
            "musparseCreateCsr",
        )?;
        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseCreateDnMat(
                    &mut dnb,
                    cols_i,
                    dc_i,
                    dc_i, // ld = k（行主序连续）
                    db_ptr.as_ptr() as *mut std::ffi::c_void,
                    dtype_code,
                    MUSPARSE_ORDER_ROW,
                )
            },
            "musparseCreateDnMat(B)",
        )?;
        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseCreateDnMat(
                    &mut dnc,
                    rows_i,
                    dc_i,
                    dc_i,
                    out_ptr.as_ptr() as *mut std::ffi::c_void,
                    dtype_code,
                    MUSPARSE_ORDER_ROW,
                )
            },
            "musparseCreateDnMat(C)",
        )?;

        let mut buf_size: usize = 0;
        // alpha/beta 指针按 dtype 选宽度（同 spmv）
        let (alpha_ptr, beta_ptr): (*const std::ffi::c_void, *const std::ffi::c_void) =
            match mat.dtype {
                Dtype::Float32 => (
                    &alpha_f32 as *const f32 as *const std::ffi::c_void,
                    &beta_f32 as *const f32 as *const std::ffi::c_void,
                ),
                Dtype::Float64 => (
                    &alpha_f64 as *const f64 as *const std::ffi::c_void,
                    &beta_f64 as *const f64 as *const std::ffi::c_void,
                ),
                _ => unreachable!("dtype whitelisted"),
            };
        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseSpMM(
                    handle,
                    MUSPARSE_OPERATION_NON_TRANSPOSE,
                    MUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha_ptr,
                    spmat,
                    dnb,
                    beta_ptr,
                    dnc,
                    dtype_code,
                    MUSPARSE_SPMM_ALG_DEFAULT,
                    MUSPARSE_SPMM_STAGE_AUTO,
                    &mut buf_size,
                    std::ptr::null_mut(),
                )
            },
            "musparseSpMM(query)",
        )?;
        let ws = math_handle::get_workspace(&mat.device, buf_size)?;
        musa_x_ffi::check_musparse(
            unsafe {
                musa_x_ffi::musparseSpMM(
                    handle,
                    MUSPARSE_OPERATION_NON_TRANSPOSE,
                    MUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha_ptr,
                    spmat,
                    dnb,
                    beta_ptr,
                    dnc,
                    dtype_code,
                    MUSPARSE_SPMM_ALG_DEFAULT,
                    MUSPARSE_SPMM_STAGE_AUTO,
                    &mut buf_size,
                    ws.ptr(),
                )
            },
            "musparseSpMM",
        )?;

        let _ = unsafe { musa_x_ffi::musparseDestroyDnMat(dnc) };
        let _ = unsafe { musa_x_ffi::musparseDestroyDnMat(dnb) };
        let _ = unsafe { musa_x_ffi::musparseDestroySpMat(spmat) };
        Ok(())
    });

    result?;

    dense_contig.data().buffer().record_read(&out_stream);
    out_ref.buffer().record_write(&out_stream);

    let out_shape = vec![rows, dc];
    Ok(Array::new(
        out_ref,
        Layout::from_shape(out_shape),
        mat.dtype,
        out_stream,
        musapy_core::DeviceResolution::new(
            mat.device.clone(),
            musapy_core::ResolutionSource::InputArray,
        ),
        musapy_core::DtypeResolution::new(mat.dtype, musapy_core::ResolutionSource::InputArray),
    ))
}

/// `toarray(mat)` — 物化稠密 Array（D2H → host 构建 → H2D，正确性优先）。
pub fn toarray(mat: &CsrMatrix) -> Result<Array> {
    let op_name = "toarray";
    require_musa(op_name, &mat.device)?;

    let (rows, cols) = mat.shape;
    let elem = mat.dtype.element_size();
    let nnz = mat.nnz;

    // nnz==0：直接返回全零稠密（0 字节 buffer 无 ptr，走不了 D2H）
    if nnz == 0 {
        let out_stream: Arc<Stream> =
            resolution::get_current_stream().unwrap_or_else(|| Arc::clone(&mat.stream));
        let nbytes = rows * cols * elem;
        let out_buffer = Buffer::alloc(nbytes, mat.device.clone(), &out_stream)?;
        let out_ref = BufferRef::new(Arc::new(out_buffer));
        fill_zeros(out_ref.buffer().ptr(), nbytes, &mat.device, &out_stream)?;
        out_ref.buffer().record_write(&out_stream);
        return Ok(Array::new(
            out_ref,
            Layout::from_shape(vec![rows, cols]),
            mat.dtype,
            out_stream,
            musapy_core::DeviceResolution::new(
                mat.device.clone(),
                musapy_core::ResolutionSource::InputArray,
            ),
            musapy_core::DtypeResolution::new(mat.dtype, musapy_core::ResolutionSource::InputArray),
        ));
    }

    // 同步 + D2H 全部 3 个 buffer
    mat.stream.synchronize()?;

    let h2h = |buf: &BufferRef, nbytes: usize| -> Result<Vec<u8>> {
        let mut host = vec![0u8; nbytes];
        let src = buf.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "toarray: null buffer pointer".into(),
                ),
            )
        })?;
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaMemcpy(
                    host.as_mut_ptr() as *mut std::ffi::c_void,
                    src.as_ptr() as *const std::ffi::c_void,
                    nbytes,
                    musa_ffi::musaMemcpyKind::DeviceToHost,
                ),
                "toarray D2H",
            )?;
        }
        Ok(host)
    };

    let data_host = h2h(&mat.data, nnz * elem)?;
    let indices_host = h2h(&mat.indices, nnz * 4)?;
    let indptr_host = h2h(&mat.indptr, (rows + 1) * 4)?;

    // host 构建稠密（rows×cols）
    let mut dense_host = vec![0u8; rows * cols * elem];
    let indices: &[i32] =
        unsafe { std::slice::from_raw_parts(indices_host.as_ptr() as *const i32, nnz) };
    let indptr: &[i32] =
        unsafe { std::slice::from_raw_parts(indptr_host.as_ptr() as *const i32, rows + 1) };

    match mat.dtype {
        Dtype::Float32 => {
            let vals: &[f32] =
                unsafe { std::slice::from_raw_parts(data_host.as_ptr() as *const f32, nnz) };
            let out: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(dense_host.as_mut_ptr() as *mut f32, rows * cols)
            };
            for i in 0..rows {
                let s = indptr[i] as usize;
                let e = indptr[i + 1] as usize;
                for k in s..e {
                    out[i * cols + indices[k] as usize] = vals[k];
                }
            }
        }
        Dtype::Float64 => {
            let vals: &[f64] =
                unsafe { std::slice::from_raw_parts(data_host.as_ptr() as *const f64, nnz) };
            let out: &mut [f64] = unsafe {
                std::slice::from_raw_parts_mut(dense_host.as_mut_ptr() as *mut f64, rows * cols)
            };
            for i in 0..rows {
                let s = indptr[i] as usize;
                let e = indptr[i + 1] as usize;
                for k in s..e {
                    out[i * cols + indices[k] as usize] = vals[k];
                }
            }
        }
        _ => unreachable!("dtype whitelisted"),
    }

    // H2D 构造稠密 Array
    let out_stream: Arc<Stream> =
        resolution::get_current_stream().unwrap_or_else(|| Arc::clone(&mat.stream));
    let buffer = Buffer::alloc(dense_host.len(), mat.device.clone(), &out_stream)?;
    let out_ref = BufferRef::new(Arc::new(buffer));
    let dst = out_ref.buffer().ptr().ok_or_else(|| {
        musapy_core::error::MusapyError::Device(
            musapy_core::error::DeviceError::MathLibCallFailed(
                "toarray: null output pointer".into(),
            ),
        )
    })?;
    unsafe {
        musa_ffi::check_musa(
            musa_ffi::musaMemcpy(
                dst.as_ptr() as *mut std::ffi::c_void,
                dense_host.as_ptr() as *const std::ffi::c_void,
                dense_host.len(),
                musa_ffi::musaMemcpyKind::HostToDevice,
            ),
            "toarray H2D",
        )?;
    }
    out_ref.buffer().record_write(&out_stream);

    let out_shape = vec![rows, cols];
    Ok(Array::new(
        out_ref,
        Layout::from_shape(out_shape),
        mat.dtype,
        out_stream,
        musapy_core::DeviceResolution::new(
            mat.device.clone(),
            musapy_core::ResolutionSource::InputArray,
        ),
        musapy_core::DtypeResolution::new(mat.dtype, musapy_core::ResolutionSource::InputArray),
    ))
}

/// 设备端零填充（nnz==0 早退路径；仿 linalg::fill_zeros）。
fn fill_zeros(
    ptr: Option<std::ptr::NonNull<u8>>,
    nbytes: usize,
    _device: &Device,
    _stream: &Arc<Stream>,
) -> Result<()> {
    let Some(p) = ptr else { return Ok(()) };
    if nbytes == 0 {
        return Ok(());
    }
    let zeros = vec![0u8; nbytes];
    unsafe {
        musa_ffi::check_musa(
            musa_ffi::musaMemcpy(
                p.as_ptr() as *mut std::ffi::c_void,
                zeros.as_ptr() as *const std::ffi::c_void,
                nbytes,
                musa_ffi::musaMemcpyKind::HostToDevice,
            ),
            "musaMemcpy(H2D zero fill)",
        )?;
    }
    Ok(())
}
