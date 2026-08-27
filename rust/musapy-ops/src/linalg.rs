//! 线性代数算子（v0.3-alpha Phase 2，ADR-003 003-D3/D6）
//!
//! `matmul` / `dot` / `solve` 三个算子，**GPU-only**（v0.3 策略：
//! 数学库算子一律不建 CPU fallback，CPU 设备上调用抛 `DeviceError`；
//! v0.2 及以前算子的 CPU 支持不受影响，见 ADR-003 003-D4 修订）。
//! 实现走 muBLAS/muSOLVER（mublas?gemm / mublas?dot / musolver?getrf+getrs）。
//!
//! 关键语义：
//!   - **行主序→列主序转置技巧**：row-major `C(m×n)=A(m×k)·B(k×n)` 等价于
//!     col-major `Cᵀ=Bᵀ·Aᵀ`——调用 gemm 时对调 A/B 且 m/n 互换，全程 OP_N，
//!     无需物化转置。
//!   - **pointer mode = HOST**（math_handle 句柄创建时一次性配置）：gemm 的
//!     alpha/beta 用宿主栈标量；dot 的 result 写宿主临时变量后 musaMemcpy H2D
//!     进输出 buffer。
//!   - **solve 奇异检测含一次同步**：getrf 后设备端 extract_diag kernel 提取
//!     LU 对角线为连续数组，单次连续 D2H 读回判奇异（P0，2026-08-08；原
//!     musaMemcpy2D 跨步 D2H 26.5ms/8KB 且行为非确定，muSOLVER 3.1.0 不写
//!     info 输出，见 gpu_solve 注释）——solve 独有的同步点（matmul/dot 无）。
//!   - **输入连续性**：非连续输入（transpose 视图等）先经 `indexing::contiguous`
//!     物化（不做 strided-gemm 优化，v0.4 评估）。
//!   - **matmul 形状语义**（NumPy 对齐）：1D 侧补 1 → 2D gemm → 对应轴 squeeze
//!     （零拷贝视图）；3D+ batch matmul 范围外（v0.3 计划 §1.2）。

use musapy_core::error::{
    DeviceError, DtypeError, LinAlgError, MemoryError, MusapyError, ShapeError,
};
use musapy_core::musa_ffi;
use musapy_core::resolution;
use musapy_core::{
    Array, Buffer, BufferRef, Device, DeviceResolution, Dtype, DtypeResolution, Layout, OpContext,
    ResolutionSource, Result, Stream, math_handle, musa_x_ffi, promote,
};
use std::ffi::c_int;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::kernels;

// ============================================================
// 公开 API
// ============================================================

/// `ms.matmul(a, b, out=None)` — 矩阵乘法（NumPy `@` 语义）。
///
/// 形状规则（仅 1D/2D，batch 推迟到 v0.4）：
/// - `(m,n) @ (n,k)` → `(m,k)`
/// - `(n,) @ (n,k)` → `(k,)`（左侧补 1，结果 squeeze 首维）
/// - `(m,n) @ (n,)` → `(m,)`（右侧补 1，结果 squeeze 末维）
/// - `(n,) @ (n,)` → 0-dim（内积）
///
/// dtype：promote 后必须落在 float32/float64 白名单。
pub fn matmul(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    let op_name = "matmul";

    // ═══════════════════════════════════════════════════════════════
    // Phase A：参数解析（一次性，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    // 1. Device 校验
    if a.device() != b.device() {
        return Err(DeviceError::Mismatch(format!(
            "{}: device mismatch {} vs {}",
            op_name,
            a.device(),
            b.device()
        ))
        .into());
    }
    let device = a.device().clone();
    require_musa(op_name, &device)?;

    // 2. 形状推导（1D 补 1 规则）
    let (a_2d, b_2d, squeeze_first, squeeze_last) = matmul_shapes(a.shape(), b.shape())?;
    let (m, k) = (a_2d[0], a_2d[1]);
    let n = b_2d[1];
    let out_shape_2d = vec![m, n];

    // 3. 类型提升（白名单 f32/f64；complex 声明就位但测试挂 Phase 5）
    let all_gpu = true; // GPU-only（v0.3 策略，见模块注释）
    let dtype = promote(a.dtype(), b.dtype(), all_gpu)?;
    check_float_whitelist(op_name, dtype)?;

    // 4. 最终输出 shape（squeeze 后）
    let out_shape = final_matmul_shape(&out_shape_2d, squeeze_first, squeeze_last);
    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let nbytes = out_size * dtype.element_size();

    // 5. out= 校验
    check_out(op_name, out, &out_shape, dtype, &device)?;

    // 6. Stream 选择（ADR L1-8）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 7. 内部 cast（输入 dtype != 提升结果 dtype）
    let a_cast = (a.dtype() != dtype)
        .then(|| crate::op_builder::cast_array(a, dtype, &out_stream))
        .transpose()?;
    let b_cast = (b.dtype() != dtype)
        .then(|| crate::op_builder::cast_array(b, dtype, &out_stream))
        .transpose()?;
    let a_casted: &Array = a_cast.as_ref().unwrap_or(a);
    let b_casted: &Array = b_cast.as_ref().unwrap_or(b);

    // 8. 连续化（gemm 要求连续输入；视图走 contiguous 物化）
    let a_contig = crate::indexing::contiguous(a_casted)?;
    let b_contig = crate::indexing::contiguous(b_casted)?;

    // 9. 输出 buffer 分配 + 别名检测（ADR L2-5）
    let (out_data_ref, out_ptr) =
        alloc_or_reuse_out(out, &a_contig, &b_contig, nbytes, &device, &out_stream)?;

    // 10. 自动 stream wait（ADR L1-8）
    a_contig.data().buffer().wait_last_write_on(&out_stream)?;
    b_contig.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：计算（GPU: mublas?gemm / CPU: cblas|朴素）
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = adjusted_ptr(&a_contig, dtype);
    let b_ptr = adjusted_ptr(&b_contig, dtype);
    if k == 0 && m > 0 && n > 0 {
        // 内维为 0：结果为全零（NumPy 语义）；buffer 未初始化，必须显式清零
        fill_zeros(out_ptr, m * n * dtype.element_size(), &device)?;
    } else {
        gpu_gemm(&device, &out_stream, a_ptr, b_ptr, out_ptr, m, n, k, dtype)?;
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理 + 组装
    // ═══════════════════════════════════════════════════════════════

    a_contig.data().buffer().record_read(&out_stream);
    b_contig.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    record_op_context(
        op_name,
        &[a.shape(), b.shape()],
        &[a.device(), b.device()],
        &[a.dtype(), b.dtype()],
        &out_shape,
        &out_stream,
    );

    let result_2d = Array::new(
        out_data_ref,
        Layout::from_shape(out_shape_2d.clone()),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    );

    // squeeze 1D 侧补出的维度（零拷贝视图）
    squeeze_result(result_2d, squeeze_first, squeeze_last)
}

/// `ms.dot(a, b, out=None)` — 点积（ADR-003 003-D6）。
///
/// - `(n,) · (n,)` → 0-dim（内积）
/// - 2D 组合委托 matmul（NumPy 语义一致）
/// - 0-dim / 3D+ 抛 ShapeError
pub fn dot(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    let op_name = "dot";

    // 0-dim 与 3D+ 拒绝（003-D6）
    if a.ndim() == 0 || b.ndim() == 0 {
        return Err(ShapeError::Mismatch(format!(
            "{}: 0-dim input not supported (a.ndim={}, b.ndim={})",
            op_name,
            a.ndim(),
            b.ndim()
        ))
        .into());
    }
    if a.ndim() > 2 || b.ndim() > 2 {
        return Err(ShapeError::Mismatch(format!(
            "{}: N-D (ndim>2) not supported yet (a.ndim={}, b.ndim={})",
            op_name,
            a.ndim(),
            b.ndim()
        ))
        .into());
    }

    // 2D 涉及 → 委托 matmul（(m,n)·(n,k)/(n,)·(n,k)/(m,n)·(n,) 全覆盖）
    if a.ndim() == 2 || b.ndim() == 2 {
        return matmul(a, b, out);
    }

    // ── 1D·1D 内积主路径 ──

    // 1. Device 校验
    if a.device() != b.device() {
        return Err(DeviceError::Mismatch(format!(
            "{}: device mismatch {} vs {}",
            op_name,
            a.device(),
            b.device()
        ))
        .into());
    }
    let device = a.device().clone();
    require_musa(op_name, &device)?;

    // 2. 形状校验
    if a.shape() != b.shape() {
        return Err(ShapeError::Mismatch(format!(
            "{}: shape mismatch {:?} vs {:?}",
            op_name,
            a.shape(),
            b.shape()
        ))
        .into());
    }
    let n = a.shape()[0];
    let out_shape: Vec<usize> = vec![]; // 0-dim

    // 3. 类型提升
    let all_gpu = true; // GPU-only（v0.3 策略，见模块注释）
    let dtype = promote(a.dtype(), b.dtype(), all_gpu)?;
    check_float_whitelist(op_name, dtype)?;

    let nbytes = dtype.element_size().max(1);

    // 4. out= 校验（0-dim 输出）
    check_out(op_name, out, &out_shape, dtype, &device)?;

    // 5. Stream 选择
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 6. 内部 cast
    let a_cast = (a.dtype() != dtype)
        .then(|| crate::op_builder::cast_array(a, dtype, &out_stream))
        .transpose()?;
    let b_cast = (b.dtype() != dtype)
        .then(|| crate::op_builder::cast_array(b, dtype, &out_stream))
        .transpose()?;
    let a_casted: &Array = a_cast.as_ref().unwrap_or(a);
    let b_casted: &Array = b_cast.as_ref().unwrap_or(b);

    // 7. 连续化（1D 视图也可能非连续，如 slice step>1）
    let a_contig = crate::indexing::contiguous(a_casted)?;
    let b_contig = crate::indexing::contiguous(b_casted)?;

    // 8. 输出 buffer + 别名检测
    let (out_data_ref, out_ptr) =
        alloc_or_reuse_out(out, &a_contig, &b_contig, nbytes, &device, &out_stream)?;

    // 9. Stream wait
    a_contig.data().buffer().wait_last_write_on(&out_stream)?;
    b_contig.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：计算
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = adjusted_ptr(&a_contig, dtype);
    let b_ptr = adjusted_ptr(&b_contig, dtype);
    // HOST pointer mode：result 写宿主栈临时量，再 musaMemcpy H2D 进输出
    // buffer（4-8 字节，同步拷贝开销可忽略）。
    gpu_dot(&device, &out_stream, a_ptr, b_ptr, out_ptr, n, dtype)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理 + 组装
    // ═══════════════════════════════════════════════════════════════

    a_contig.data().buffer().record_read(&out_stream);
    b_contig.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    record_op_context(
        op_name,
        &[a.shape(), b.shape()],
        &[a.device(), b.device()],
        &[a.dtype(), b.dtype()],
        &out_shape,
        &out_stream,
    );

    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

/// `ms.solve(a, b)` — 解线性方程组 `a @ x = b`（LU 分解 + 回代）。
///
/// - `a`：方阵 `(n,n)`；`b`：`(n,)` → x `(n,)`；`(n,k)` → x `(n,k)`（k 个 rhs）
/// - 奇异矩阵（getrf info > 0）抛 `LinAlgError::Singular`（003-D3）
/// - **同步点**：info 为设备指针，D2H 读回判奇异（musaMemcpy 同步语义）
pub fn solve(a: &Array, b: &Array) -> Result<Array> {
    let op_name = "solve";

    // 1. Device 校验
    if a.device() != b.device() {
        return Err(DeviceError::Mismatch(format!(
            "{}: device mismatch {} vs {}",
            op_name,
            a.device(),
            b.device()
        ))
        .into());
    }
    let device = a.device().clone();
    require_musa(op_name, &device)?;

    // 2. 形状校验（a 方阵；b 行数匹配）
    if a.ndim() != 2 {
        return Err(ShapeError::Mismatch(format!(
            "{}: a must be 2-D square matrix, got ndim {}",
            op_name,
            a.ndim()
        ))
        .into());
    }
    if a.shape()[0] != a.shape()[1] {
        return Err(ShapeError::Mismatch(format!(
            "{}: a must be square, got {:?}",
            op_name,
            a.shape()
        ))
        .into());
    }
    let n = a.shape()[0];
    if b.ndim() < 1 || b.ndim() > 2 {
        return Err(ShapeError::Mismatch(format!(
            "{}: b must be 1-D or 2-D, got ndim {}",
            op_name,
            b.ndim()
        ))
        .into());
    }
    if b.shape()[0] != n {
        return Err(ShapeError::Mismatch(format!(
            "{}: b leading dim {} != a size {}",
            op_name,
            b.shape()[0],
            n
        ))
        .into());
    }
    let nrhs = if b.ndim() == 2 { b.shape()[1] } else { 1 };
    let out_shape = b.shape().clone();

    // 空矩阵（n=0）：直接返回空输出（不调用 musolver）
    if n == 0 {
        let out_stream: Arc<Stream> =
            resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream()));
        let nbytes = (out_shape.iter().product::<usize>().max(1)) * a.dtype().element_size();
        let buffer = Buffer::alloc(nbytes.max(1), device.clone(), &out_stream)?;
        return Ok(Array::new(
            BufferRef::new(Arc::new(buffer)),
            Layout::from_shape(out_shape),
            a.dtype(),
            out_stream,
            DeviceResolution::new(device, ResolutionSource::InputArray),
            DtypeResolution::new(a.dtype(), ResolutionSource::InputArray),
        ));
    }

    // 3. 类型提升
    let all_gpu = true; // GPU-only（v0.3 策略，见模块注释）
    let dtype = promote(a.dtype(), b.dtype(), all_gpu)?;
    check_float_whitelist(op_name, dtype)?;

    // 4. Stream 选择
    let out_stream: Arc<Stream> =
        resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream()));

    // 5. 内部 cast + 连续化
    let a_cast = (a.dtype() != dtype)
        .then(|| crate::op_builder::cast_array(a, dtype, &out_stream))
        .transpose()?;
    let b_cast = (b.dtype() != dtype)
        .then(|| crate::op_builder::cast_array(b, dtype, &out_stream))
        .transpose()?;
    let a_contig = crate::indexing::contiguous(a_cast.as_ref().unwrap_or(a))?;
    let b_contig = crate::indexing::contiguous(b_cast.as_ref().unwrap_or(b))?;

    a_contig.data().buffer().wait_last_write_on(&out_stream)?;
    b_contig.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：求解（getrf 是 in-place，必须先复制 A/B）
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = adjusted_ptr(&a_contig, dtype);
    let b_ptr = adjusted_ptr(&b_contig, dtype);
    let elem = dtype.element_size();

    // GPU：设备内存复制 A/B（getrf/getrs 原地破坏），再 getrf→奇异检测→getrs
    let a_bytes = n * n * elem;
    let b_bytes = n * nrhs * elem;

    let lu_buf = Buffer::alloc(a_bytes, device.clone(), &out_stream)?;
    let lu_ref = BufferRef::new(Arc::new(lu_buf));
    let rhs_buf = Buffer::alloc(b_bytes, device.clone(), &out_stream)?;
    let rhs_ref = BufferRef::new(Arc::new(rhs_buf));

    let (Some(lu_ptr), Some(rhs_ptr)) = (lu_ref.buffer().ptr(), rhs_ref.buffer().ptr()) else {
        return Err(MemoryError::OutOfMemory("solve: null buffer ptr".into()).into());
    };

    // 同步拷贝 A/B 到工作 buffer（musaMemcpy 同步语义，同 stream 有序）
    unsafe {
        if let Some(p) = a_ptr {
            musa_ffi::check_musa(
                musa_ffi::musaMemcpy(
                    lu_ptr.as_ptr() as *mut std::ffi::c_void,
                    p.as_ptr() as *const std::ffi::c_void,
                    a_bytes,
                    musa_ffi::musaMemcpyKind::DeviceToDevice,
                ),
                "musaMemcpy(D2D solve A copy)",
            )?;
        }
        if let Some(p) = b_ptr {
            musa_ffi::check_musa(
                musa_ffi::musaMemcpy(
                    rhs_ptr.as_ptr() as *mut std::ffi::c_void,
                    p.as_ptr() as *const std::ffi::c_void,
                    b_bytes,
                    musa_ffi::musaMemcpyKind::DeviceToDevice,
                ),
                "musaMemcpy(D2D solve b copy)",
            )?;
        }
    }

    gpu_solve(&device, &out_stream, lu_ptr, rhs_ptr, n, nrhs, dtype)?;
    let out_data_ref = rhs_ref;

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理 + 组装
    // ═══════════════════════════════════════════════════════════════

    a_contig.data().buffer().record_read(&out_stream);
    b_contig.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    record_op_context(
        op_name,
        &[a.shape(), b.shape()],
        &[a.device(), b.device()],
        &[a.dtype(), b.dtype()],
        &out_shape,
        &out_stream,
    );

    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

// ============================================================
// 通用助手
// ============================================================

/// GPU-only 校验（v0.3 策略：数学库算子不建 CPU fallback，ADR-003 003-D4 修订）。
///
/// CPU 设备上调用 matmul/dot/solve 一律拒绝——与 `math_handle::musa_id`
/// 对 CPU 的既有报错风格一致（"MUSA-X math libraries require a musa device"）。
pub(crate) fn require_musa(op_name: &str, device: &Device) -> Result<()> {
    match device {
        Device::Musa(_) => Ok(()),
        Device::Cpu => Err(DeviceError::Mismatch(format!(
            "{}: requires a musa device, got cpu (v0.3 math-lib ops are GPU-only)",
            op_name
        ))
        .into()),
    }
}

/// f32/f64 计算白名单（complex 声明就位，Phase 5 放开）。
pub(crate) fn check_float_whitelist(op_name: &str, dtype: Dtype) -> Result<()> {
    match dtype {
        Dtype::Float32 | Dtype::Float64 => Ok(()),
        _ => Err(DtypeError::Unsupported(format!(
            "{}: promoted dtype {} not supported (compute whitelist: float32/float64)",
            op_name, dtype
        ))
        .into()),
    }
}

/// out= 参数校验（shape/dtype/device）。
fn check_out(
    op_name: &str,
    out: Option<&Array>,
    out_shape: &[usize],
    dtype: Dtype,
    device: &Device,
) -> Result<()> {
    if let Some(o) = out {
        if o.shape() != out_shape {
            return Err(ShapeError::Mismatch(format!(
                "{}: out shape {:?} != expected {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != dtype {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != expected {}",
                op_name,
                o.dtype(),
                dtype
            ))
            .into());
        }
        if o.device() != device {
            return Err(DeviceError::Mismatch(format!(
                "{}: out device {} != input device {}",
                op_name,
                o.device(),
                device
            ))
            .into());
        }
    }
    Ok(())
}

/// 输出 buffer 分配（或复用 out=）+ 别名检测（ADR L2-5）。
fn alloc_or_reuse_out(
    out: Option<&Array>,
    a_work: &Array,
    b_work: &Array,
    nbytes: usize,
    device: &Device,
    out_stream: &Arc<Stream>,
) -> Result<(BufferRef, Option<NonNull<u8>>)> {
    match out {
        Some(o) => {
            if o.data() == a_work.data() || o.data() == b_work.data() {
                return Err(MemoryError::AliasDetected.into());
            }
            Ok((o.data().clone(), o.data().buffer().ptr()))
        }
        None => {
            let buffer = Buffer::alloc(nbytes.max(1), device.clone(), out_stream)?;
            let buffer_arc = Arc::new(buffer);
            let data_ref = BufferRef::new(buffer_arc);
            let ptr = data_ref.buffer().ptr();
            Ok((data_ref, ptr))
        }
    }
}

/// 输出清零（k=0 退化场景；CPU 直写，GPU 走 H2D）。
/// 输出清零（k=0 退化场景；GPU-only，H2D 零拷贝）。
fn fill_zeros(ptr: Option<NonNull<u8>>, nbytes: usize, device: &Device) -> Result<()> {
    let Some(p) = ptr else { return Ok(()) };
    if nbytes == 0 {
        return Ok(());
    }
    let zeros = vec![0u8; nbytes];
    match device {
        Device::Musa(_) => {
            musa_ffi::check_musa(
                unsafe {
                    musa_ffi::musaMemcpy(
                        p.as_ptr() as *mut std::ffi::c_void,
                        zeros.as_ptr() as *const std::ffi::c_void,
                        nbytes,
                        musa_ffi::musaMemcpyKind::HostToDevice,
                    )
                },
                "musaMemcpy(H2D zero fill)",
            )?;
        }
        // CPU 分支已随 v0.3 GPU-only 策略移除（require_musa 前置拦截）
        Device::Cpu => unreachable!("linalg ops are GPU-only (require_musa rejected cpu)"),
    }
    Ok(())
}

/// 指针偏移调整（layout offset；contiguous 后恒为 0，保留防御）。
fn adjusted_ptr(a: &Array, dtype: Dtype) -> Option<NonNull<u8>> {
    crate::op_builder::adjust_ptr_offset(
        a.data().buffer().ptr(),
        a.layout().offset,
        dtype.element_size(),
    )
}

/// debug 模式 OpContext 记录（仿 op_builder 惯例）。
fn record_op_context(
    op_name: &'static str,
    shapes: &[&[usize]],
    devices: &[&Device],
    dtypes: &[Dtype],
    out_shape: &[usize],
    out_stream: &Arc<Stream>,
) {
    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            op_name,
            shapes.iter().map(|s| s.to_vec()).collect(),
            devices.iter().map(|d| (*d).clone()).collect(),
            dtypes.to_vec(),
            out_shape.to_vec(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }
}

// ── matmul 形状推导 ─────────────────────────────────────────

/// 推导 2D 工作形状与 squeeze 标志。
///
/// 返回 `(a_2d, b_2d, squeeze_first, squeeze_last)`：
/// - `(n,)@(n,m)` → a 前补 1，结果 squeeze 首维
/// - `(m,n)@(n,)` → b 后补 1，结果 squeeze 末维
/// - `(n,)@(n,)` → 两侧补 1，结果 squeeze 两维（0-dim）
fn matmul_shapes(
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<([usize; 2], [usize; 2], bool, bool)> {
    if a_shape.is_empty() || b_shape.is_empty() {
        return Err(ShapeError::Mismatch(format!(
            "matmul: 0-dim input not supported (a.ndim={}, b.ndim={})",
            a_shape.len(),
            b_shape.len()
        ))
        .into());
    }
    if a_shape.len() > 2 || b_shape.len() > 2 {
        return Err(ShapeError::Mismatch(format!(
            "matmul: batch matmul (ndim>2) not supported yet (a.ndim={}, b.ndim={})",
            a_shape.len(),
            b_shape.len()
        ))
        .into());
    }

    let a_left_1d = a_shape.len() == 1;
    let b_right_1d = b_shape.len() == 1;
    let a_2d = if a_left_1d {
        [1, a_shape[0]]
    } else {
        [a_shape[0], a_shape[1]]
    };
    let b_2d = if b_right_1d {
        [b_shape[0], 1]
    } else {
        [b_shape[0], b_shape[1]]
    };

    if a_2d[1] != b_2d[0] {
        return Err(ShapeError::Mismatch(format!(
            "matmul: inner dims mismatch: a {:?} vs b {:?}",
            a_shape, b_shape
        ))
        .into());
    }

    Ok((a_2d, b_2d, a_left_1d, b_right_1d))
}

/// 2D 输出 shape 经 squeeze 后的最终 shape。
fn final_matmul_shape(out_2d: &[usize], squeeze_first: bool, squeeze_last: bool) -> Vec<usize> {
    match (squeeze_first, squeeze_last) {
        (true, true) => vec![],
        (true, false) => vec![out_2d[1]],
        (false, true) => vec![out_2d[0]],
        (false, false) => out_2d.to_vec(),
    }
}

/// squeeze 补出的 1 维（零拷贝视图）。
fn squeeze_result(a: Array, squeeze_first: bool, squeeze_last: bool) -> Result<Array> {
    if !squeeze_first && !squeeze_last {
        return Ok(a);
    }
    // 2D 连续布局 squeeze 任一维度后仍连续（被 squeeze 的维度 size=1）
    let shape = a.shape().clone();
    let new_shape = match (squeeze_first, squeeze_last) {
        (true, true) => vec![],
        (true, false) => vec![shape[1]],
        (false, true) => vec![shape[0]],
        (false, false) => unreachable!(),
    };
    Ok(Array::new_view(&a, Layout::from_shape(new_shape)))
}

// ============================================================
// GPU 路径（muBLAS / muSOLVER）
// ============================================================

/// GPU gemm：row-major `C(m×n)=A(m×k)·B(k×n)` 经转置技巧映射为
/// col-major `Cᵀ=Bᵀ·Aᵀ`——gemm(OP_N, OP_N, m=n, n=m, k=k, A=B_ptr, lda=n,
/// B=A_ptr, ldb=k, C=C_ptr, ldc=n)。
#[allow(clippy::too_many_arguments)]
fn gpu_gemm(
    device: &Device,
    stream: &Arc<Stream>,
    a_ptr: Option<NonNull<u8>>,
    b_ptr: Option<NonNull<u8>>,
    c_ptr: Option<NonNull<u8>>,
    m: usize,
    n: usize,
    k: usize,
    dtype: Dtype,
) -> Result<()> {
    let (Some(ap), Some(bp), Some(cp)) = (a_ptr, b_ptr, c_ptr) else {
        return Ok(()); // 0 元素或空 buffer：无需计算
    };
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    math_handle::with_mublas_handle(device, stream, |handle| {
        // mublas_int = i32（LP64）；shape 已在调用侧受 Python 层约束
        let (mi, ni, ki) = (m as c_int, n as c_int, k as c_int);
        let (lda, ldb, ldc) = (ni, ki, ni); // 转置技巧：见函数文档
        let status = match dtype {
            Dtype::Float32 => {
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                unsafe {
                    musa_x_ffi::mublasSgemm(
                        handle,
                        musa_x_ffi::MUBLAS_OP_N,
                        musa_x_ffi::MUBLAS_OP_N,
                        ni,
                        mi,
                        ki,
                        &alpha,
                        bp.as_ptr() as *const f32,
                        lda,
                        ap.as_ptr() as *const f32,
                        ldb,
                        &beta,
                        cp.as_ptr() as *mut f32,
                        ldc,
                    )
                }
            }
            Dtype::Float64 => {
                let alpha: f64 = 1.0;
                let beta: f64 = 0.0;
                unsafe {
                    musa_x_ffi::mublasDgemm(
                        handle,
                        musa_x_ffi::MUBLAS_OP_N,
                        musa_x_ffi::MUBLAS_OP_N,
                        ni,
                        mi,
                        ki,
                        &alpha,
                        bp.as_ptr() as *const f64,
                        lda,
                        ap.as_ptr() as *const f64,
                        ldb,
                        &beta,
                        cp.as_ptr() as *mut f64,
                        ldc,
                    )
                }
            }
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "mublas gemm")
    })
}

/// GPU dot：HOST pointer mode 下 result 写宿主栈临时量，musaMemcpy H2D 进输出。
#[allow(clippy::too_many_arguments)]
fn gpu_dot(
    device: &Device,
    stream: &Arc<Stream>,
    a_ptr: Option<NonNull<u8>>,
    b_ptr: Option<NonNull<u8>>,
    out_ptr: Option<NonNull<u8>>,
    n: usize,
    dtype: Dtype,
) -> Result<()> {
    let (Some(ap), Some(bp), Some(op)) = (a_ptr, b_ptr, out_ptr) else {
        return Ok(());
    };
    let Some(id) = device.musa_id() else {
        return Ok(());
    };

    if n == 0 {
        // 空向量内积 = 0（0-dim 标量）
        musa_ffi::set_device(id as i32)?;
        write_zero_device(op.as_ptr(), dtype)?;
        return Ok(());
    }

    let elem = dtype.element_size();
    let mut host_tmp = vec![0u8; elem];

    math_handle::with_mublas_handle(device, stream, |handle| {
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::mublasSdot(
                    handle,
                    n as c_int,
                    ap.as_ptr() as *const f32,
                    1,
                    bp.as_ptr() as *const f32,
                    1,
                    host_tmp.as_mut_ptr() as *mut f32,
                )
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::mublasDdot(
                    handle,
                    n as c_int,
                    ap.as_ptr() as *const f64,
                    1,
                    bp.as_ptr() as *const f64,
                    1,
                    host_tmp.as_mut_ptr() as *mut f64,
                )
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "mublas dot")
    })?;

    // result 写宿主临时量 → H2D 进输出 buffer（4-8 字节，同步拷贝）
    musa_ffi::set_device(id as i32)?;
    musa_ffi::check_musa(
        unsafe {
            musa_ffi::musaMemcpy(
                op.as_ptr() as *mut std::ffi::c_void,
                host_tmp.as_ptr() as *const std::ffi::c_void,
                elem,
                musa_ffi::musaMemcpyKind::HostToDevice,
            )
        },
        "musaMemcpy(H2D dot result)",
    )
}

/// GPU solve：getrf（LU 对角 D2H 奇异检测）→ getrs。
///
/// 调用前提：lu/rhs 已是 A/b 的设备内存副本（getrf/getrs 原地破坏）。
#[allow(clippy::too_many_arguments)]
fn gpu_solve(
    device: &Device,
    stream: &Arc<Stream>,
    lu_ptr: NonNull<u8>,
    rhs_ptr: NonNull<u8>,
    n: usize,
    nrhs: usize,
    dtype: Dtype,
) -> Result<()> {
    let ni = n as c_int;
    let nrhsi = nrhs as c_int;
    let elem = dtype.element_size();

    // ipiv / info：设备内存（getrf/getrs 要求设备指针）。
    // 注：muSOLVER 3.1.0 不写 info（SDK 缺陷，见奇异检测处注释），
    // 仍分配 4 字节占位以满足 FFI 签名，值不读取。
    let ipiv_bytes = n * std::mem::size_of::<c_int>();

    // getrf（共享 helper：bufferSize → workspace → 分解 → host ipiv）
    let (ipiv_dev, info_dev, _ipiv_guard) = alloc_ipiv_info(device, ipiv_bytes)?;
    let _ipiv_host = gpu_getrf(device, stream, lu_ptr, n, n, ipiv_dev, info_dev, dtype)?;

    math_handle::with_mublas_handle(device, stream, |handle| {
        // ── 奇异检测（solve 独有的同步点）──
        // muSOLVER 3.1.0 的 getrf **从不写 info 输出**（2026-08-07 真机 C 探针
        // 实测：非奇异与奇异矩阵 info 均保持预填值，status=0、ipiv 正常）——
        // 故不能依赖 info 判奇异。改用 LAPACK 等价判据：info = 首个
        // U(k,k) == 0 的行号（LAPACK 正是 `IF A(K,K).EQ.ZERO → info=K`）。
        // U 的对角元素在 LU buffer（列主序 M=Aᵀ）偏移 k·(n+1) 处，跨步
        // (n+1)·elem。P0（2026-08-08）：原 musaMemcpy2D 跨步 D2H 逐行
        // ~26µs/行（8KB 对角实测 26.5ms，且该 API 小 pitch D2H 行为非确定
        // 性，见 sdk-3.1.0-limitations.md）→ 改为设备端 extract_diag kernel
        // 提取为连续数组 + 单次连续 D2H（0.18ms），host 扫描逻辑不变。
        let diag_buf = alloc_buf_ref(n * elem, device, stream)?;
        let diag_ptr = diag_buf.buffer().ptr().ok_or_else(|| {
            MusapyError::Device(DeviceError::MathLibCallFailed(
                "solve: null diag buffer pointer".into(),
            ))
        })?;
        unsafe {
            match dtype {
                Dtype::Float32 => {
                    kernels::musapy_extract_diag_f32_v1(
                        lu_ptr.as_ptr() as *const f32,
                        diag_ptr.as_ptr() as *mut f32,
                        n,
                        n + 1,
                        stream.raw(),
                    );
                }
                Dtype::Float64 => {
                    kernels::musapy_extract_diag_f64_v1(
                        lu_ptr.as_ptr() as *const f64,
                        diag_ptr.as_ptr() as *mut f64,
                        n,
                        n + 1,
                        stream.raw(),
                    );
                }
                _ => unreachable!("dtype already validated as float32/float64"),
            }
        }
        musa_ffi::check_last_kernel_launch("extract_diag")?;
        stream.synchronize()?;
        let mut diag_host = vec![0u8; n * elem];
        musa_ffi::check_musa(
            unsafe {
                musa_ffi::musaMemcpy(
                    diag_host.as_mut_ptr() as *mut std::ffi::c_void,
                    diag_ptr.as_ptr() as *const std::ffi::c_void,
                    n * elem,
                    musa_ffi::musaMemcpyKind::DeviceToHost,
                )
            },
            "musaMemcpy(D2H solve LU diagonal)",
        )?;
        let first_zero = match dtype {
            Dtype::Float32 => diag_host
                .chunks_exact(4)
                .position(|b| f32::from_le_bytes(b.try_into().unwrap()) == 0.0),
            Dtype::Float64 => diag_host
                .chunks_exact(8)
                .position(|b| f64::from_le_bytes(b.try_into().unwrap()) == 0.0),
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        if let Some(k) = first_zero {
            return Err(LinAlgError::Singular(format!(
                "solve: singular matrix (U({},{}) is zero; muSOLVER info unavailable on SDK 3.1.0)",
                k + 1,
                k + 1
            ))
            .into());
        }

        // ── getrs：回代求解（x 覆盖 rhs buffer）──
        // LAPACK 语义（muSOLVER 同）：getrs 求解 op(M)·X = B，M/B/X 均为
        // 列主序。行主序 A 的副本按列主序解释是 M = Aᵀ（M[i][j] = A[j][i]），
        // 故用 OP_T 使 op(M) = Mᵀ = A，直接解 A·X = B（零拷贝，无需转置 A）。
        // 行主序 b（n×nrhs）在列主序视角（ldb=n）下是乱序矩阵：
        // nrhs > 1 时必须先把 b 按列主序拷贝（b_cm[i + j·n] = b[i·nrhs + j]），
        // 求解后再把 X 拷回行主序；nrhs == 1 时两种布局重合，零拷贝。
        let mut rs_ws: c_int = 0;
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgetrs_bufferSize(
                    musa_x_ffi::MUBLAS_OP_T,
                    ni,
                    nrhsi,
                    &mut rs_ws,
                )
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgetrs_bufferSize(
                    musa_x_ffi::MUBLAS_OP_T,
                    ni,
                    nrhsi,
                    &mut rs_ws,
                )
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver getrs_bufferSize")?;

        let ws2 = math_handle::get_workspace(device, rs_ws as usize)?;

        // nrhs>1：行主序 b → 列主序 b_cm（host 中转；solve 本身含同步点，开销可忽略）
        let b_cm_ptr: *mut std::ffi::c_void = if nrhs > 1 {
            let mut host_rhs = vec![0u8; n * nrhs * elem];
            let mut host_cm = vec![0u8; n * nrhs * elem];
            // D2H 读回当前 rhs buffer（原始 b）
            musa_ffi::check_musa(
                unsafe {
                    musa_ffi::musaMemcpy(
                        host_rhs.as_mut_ptr() as *mut std::ffi::c_void,
                        rhs_ptr.as_ptr() as *const std::ffi::c_void,
                        n * nrhs * elem,
                        musa_ffi::musaMemcpyKind::DeviceToHost,
                    )
                },
                "musaMemcpy(D2H solve rhs)",
            )?;
            // 行主序 → 列主序（按元素，f32/f64 通用字节级复制）
            for j in 0..nrhs {
                for i in 0..n {
                    let src = &host_rhs[(i * nrhs + j) * elem..(i * nrhs + j + 1) * elem];
                    host_cm[(i + j * n) * elem..(i + j * n + 1) * elem].copy_from_slice(src);
                }
            }
            let mut b_cm: *mut std::ffi::c_void = std::ptr::null_mut();
            musa_ffi::check_musa(
                unsafe { musa_ffi::musaMalloc(&mut b_cm, (n * nrhs * elem).max(1)) },
                "musaMalloc(solve b_cm)",
            )?;
            // H2D 写入列主序 b
            musa_ffi::check_musa(
                unsafe {
                    musa_ffi::musaMemcpy(
                        b_cm,
                        host_cm.as_ptr() as *const std::ffi::c_void,
                        n * nrhs * elem,
                        musa_ffi::musaMemcpyKind::HostToDevice,
                    )
                },
                "musaMemcpy(H2D solve b_cm)",
            )?;
            b_cm
        } else {
            rhs_ptr.as_ptr() as *mut std::ffi::c_void
        };
        // RAII：b_cm（nrhs>1 时分配）离开闭包前释放
        struct CmGuard(*mut std::ffi::c_void);
        impl Drop for CmGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { musa_ffi::musaFree(self.0) };
                }
            }
        }
        let _cm_guard = CmGuard(if nrhs > 1 {
            b_cm_ptr
        } else {
            std::ptr::null_mut()
        });

        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgetrs(
                    handle,
                    musa_x_ffi::MUBLAS_OP_T,
                    ni,
                    nrhsi,
                    lu_ptr.as_ptr() as *const f32,
                    ni,
                    ipiv_dev as *const c_int,
                    b_cm_ptr as *mut f32,
                    ni,
                    ws2.ptr(),
                )
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgetrs(
                    handle,
                    musa_x_ffi::MUBLAS_OP_T,
                    ni,
                    nrhsi,
                    lu_ptr.as_ptr() as *const f64,
                    ni,
                    ipiv_dev as *const c_int,
                    b_cm_ptr as *mut f64,
                    ni,
                    ws2.ptr(),
                )
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver getrs")?;

        // nrhs>1：解 X（列主序 b_cm 中）→ 行主序拷回 rhs buffer
        if nrhs > 1 {
            let mut host_cm = vec![0u8; n * nrhs * elem];
            let mut host_x = vec![0u8; n * nrhs * elem];
            musa_ffi::check_musa(
                unsafe {
                    musa_ffi::musaMemcpy(
                        host_cm.as_mut_ptr() as *mut std::ffi::c_void,
                        b_cm_ptr,
                        n * nrhs * elem,
                        musa_ffi::musaMemcpyKind::DeviceToHost,
                    )
                },
                "musaMemcpy(D2H solve x)",
            )?;
            for j in 0..nrhs {
                for i in 0..n {
                    let src = &host_cm[(i + j * n) * elem..(i + j * n + 1) * elem];
                    host_x[(i * nrhs + j) * elem..(i * nrhs + j + 1) * elem].copy_from_slice(src);
                }
            }
            musa_ffi::check_musa(
                unsafe {
                    musa_ffi::musaMemcpy(
                        rhs_ptr.as_ptr() as *mut std::ffi::c_void,
                        host_x.as_ptr() as *const std::ffi::c_void,
                        n * nrhs * elem,
                        musa_ffi::musaMemcpyKind::HostToDevice,
                    )
                },
                "musaMemcpy(H2D solve x)",
            )?;
        }
        Ok(())
    })
}

/// 设备内存写 0（dot 空向量 = 0）。
fn write_zero_device(ptr: *mut u8, dtype: Dtype) -> Result<()> {
    let elem = dtype.element_size();
    let zeros = vec![0u8; elem];
    musa_ffi::check_musa(
        unsafe {
            musa_ffi::musaMemcpy(
                ptr as *mut std::ffi::c_void,
                zeros.as_ptr() as *const std::ffi::c_void,
                elem,
                musa_ffi::musaMemcpyKind::HostToDevice,
            )
        },
        "musaMemcpy(H2D zero scalar)",
    )
}

// ============================================================
// Phase 3：分解类算子（lu / qr / svd，GPU-only，003-D3/D6）
// ============================================================
//
// 统一「列主序物化输入」：`col_major_copy` = contiguous(transpose(a))
// （复用 transpose 视图 + copy_into 内核，零新增 kernel，符合 v0.3 约束）。
// 输出用跨步视图零拷贝呈现：列主序输出读为行主序 = strides `(1, lda)` 视图
// （lu/qr/svd 语义由 2026-08-07 真机 C 探针锁定，见 plan-phase3 文档）。

/// `ms.lu(a)` → `(lu, piv)` — LU 分解（torch.linalg.lu 语义）。
///
/// - `lu`：(m×n) 行主序，L 单位下三角 + U 上三角（LAPACK getrf 布局），
///   重建 `a = P·L·U`（P 由 piv 按行交换构造）
/// - `piv`：`min(m,n)` 个 int64，1-based（LAPACK ipiv 语义）
pub fn lu(a: &Array) -> Result<(Array, Array)> {
    let op_name = "lu";
    require_musa(op_name, a.device())?;
    check_float_whitelist(op_name, a.dtype())?;
    if a.ndim() != 2 {
        return Err(ShapeError::Mismatch(format!(
            "{}: a must be 2-D, got ndim {}",
            op_name,
            a.ndim()
        ))
        .into());
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let k = m.min(n);
    let device = a.device().clone();
    let dtype = a.dtype();
    let out_stream: Arc<Stream> =
        resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream()));

    // 空矩阵早退（0 行/列：空 lu + 空 piv）
    if m == 0 || n == 0 {
        let lu_buf = Buffer::alloc(1, device.clone(), &out_stream)?;
        let piv_buf = Buffer::alloc(1, device.clone(), &out_stream)?;
        return Ok((
            Array::new(
                BufferRef::new(Arc::new(lu_buf)),
                Layout::from_shape(vec![m, n]),
                dtype,
                Arc::clone(&out_stream),
                DeviceResolution::new(device.clone(), ResolutionSource::InputArray),
                DtypeResolution::new(dtype, ResolutionSource::InputArray),
            ),
            Array::new(
                BufferRef::new(Arc::new(piv_buf)),
                Layout::from_shape(vec![k]),
                Dtype::Int64,
                out_stream,
                DeviceResolution::new(device, ResolutionSource::InputArray),
                DtypeResolution::new(Dtype::Int64, ResolutionSource::InputArray),
            ),
        ));
    }

    // 列主序物化副本（getrf 的分解对象是 A 本身而非 Aᵀ——探针锁定）
    let cm = col_major_copy(a, &out_stream)?;
    cm.data().buffer().wait_last_write_on(&out_stream)?;
    cm.data().buffer().record_read(&out_stream);
    record_op_context(
        "lu",
        &[a.shape()],
        &[a.device()],
        &[a.dtype()],
        &[m, n],
        &out_stream,
    );
    let (lu_ptr, lu_data) = device_ptr(cm.data())?;

    // getrf：ipiv/info 设备分配 + 两段式分解
    let (ipiv_dev, info_dev, _guards) = alloc_ipiv_info(&device, k * std::mem::size_of::<c_int>())?;
    let ipiv = gpu_getrf(
        &device,
        &out_stream,
        lu_ptr,
        m,
        n,
        ipiv_dev,
        info_dev,
        dtype,
    )?;

    // lu = strides (1, m) 视图：列主序 L·U 读为行主序标准布局（零拷贝）
    let lu_arr = strided_view(
        &lu_data,
        [m, n],
        [1, m as isize],
        dtype,
        &device,
        &out_stream,
    );

    // piv：D2H 的 1-based ipiv → int64 设备数组
    let piv_arr = int64_array_from(&ipiv, k, &device, &out_stream)?;
    lu_data.buffer().record_write(&out_stream);
    piv_arr.data().buffer().record_write(&out_stream);
    Ok((lu_arr, piv_arr))
}

/// `ms.qr(a, mode="reduced")` → `(q, r)` — QR 分解（NumPy 语义）。
///
/// - `mode="reduced"`：q (m,k)、r (k,n)，k=min(m,n)
/// - `mode="complete"`：q (m,m)、r (m,n)（r 下三角补零）
pub fn qr(a: &Array, mode: &str) -> Result<(Array, Array)> {
    let op_name = "qr";
    require_musa(op_name, a.device())?;
    check_float_whitelist(op_name, a.dtype())?;
    if a.ndim() != 2 {
        return Err(ShapeError::Mismatch(format!(
            "{}: a must be 2-D, got ndim {}",
            op_name,
            a.ndim()
        ))
        .into());
    }
    let complete = match mode {
        "reduced" => false,
        "complete" => true,
        other => {
            return Err(ShapeError::Mismatch(format!(
                "{}: mode must be 'reduced' or 'complete', got {:?}",
                op_name, other
            ))
            .into());
        }
    };
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let k = m.min(n);
    let device = a.device().clone();
    let dtype = a.dtype();
    let elem = dtype.element_size();
    let out_stream: Arc<Stream> =
        resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream()));

    if m == 0 || n == 0 {
        let q_shape = if complete { vec![m, m] } else { vec![m, k] };
        let r_shape = if complete { vec![m, n] } else { vec![k, n] };
        let q_buf = Buffer::alloc(1, device.clone(), &out_stream)?;
        let r_buf = Buffer::alloc(1, device.clone(), &out_stream)?;
        return Ok((
            Array::new(
                BufferRef::new(Arc::new(q_buf)),
                Layout::from_shape(q_shape),
                dtype,
                Arc::clone(&out_stream),
                DeviceResolution::new(device.clone(), ResolutionSource::InputArray),
                DtypeResolution::new(dtype, ResolutionSource::InputArray),
            ),
            Array::new(
                BufferRef::new(Arc::new(r_buf)),
                Layout::from_shape(r_shape),
                dtype,
                out_stream,
                DeviceResolution::new(device, ResolutionSource::InputArray),
                DtypeResolution::new(dtype, ResolutionSource::InputArray),
            ),
        ));
    }

    // 列主序物化副本（geqrf/orgqr 全程作用于此缓冲）
    let cm = col_major_copy(a, &out_stream)?;
    cm.data().buffer().wait_last_write_on(&out_stream)?;
    cm.data().buffer().record_read(&out_stream);
    record_op_context(
        "qr",
        &[a.shape()],
        &[a.device()],
        &[a.dtype()],
        &[m, if complete { m } else { k }],
        &out_stream,
    );
    let (a_ptr, cm_data) = device_ptr(cm.data())?;

    // tau：设备缓冲（geqrf 反射系数；orgqr 需要）
    let (tau_dev, _tau_guard) = alloc_dev_bytes(&device, k * elem)?;

    // Phase 1: geqrf（两段式）
    math_handle::with_mublas_handle(&device, &out_stream, |handle| {
        let (mi, ni) = (m as c_int, n as c_int);
        let mut ws_size: c_int = 0;
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgeqrf_bufferSize(mi, ni, &mut ws_size)
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgeqrf_bufferSize(mi, ni, &mut ws_size)
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver geqrf_bufferSize")?;
        let ws = math_handle::get_workspace(&device, ws_size as usize)?;
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgeqrf(
                    handle,
                    mi,
                    ni,
                    a_ptr.as_ptr() as *mut f32,
                    mi,
                    tau_dev as *mut f32,
                    ws.ptr(),
                )
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgeqrf(
                    handle,
                    mi,
                    ni,
                    a_ptr.as_ptr() as *mut f64,
                    mi,
                    tau_dev as *mut f64,
                    ws.ptr(),
                )
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver geqrf")
    })?;

    // Phase 2: R 提取（geqrf 后、orgqr 前——orgqr 覆盖前 k 列；
    // host 中转组装行主序 R，下三角清零）
    let r_rows = if complete { m } else { k };
    let r = extract_upper_r(&cm_data, m, n, k, r_rows, dtype, &device, &out_stream)?;

    // Phase 3: orgqr 生成 Q
    let (q_data, q_shape) = if complete && m > n {
        // m×m 零填充缓冲：前 k 列拷入反射子（host 组装列主序）
        let q_buf = Buffer::alloc(m * m * elem, device.clone(), &out_stream)?;
        let q_ref = BufferRef::new(Arc::new(q_buf));
        let (q_ptr, q_data) = device_ptr(&q_ref)?;
        // host：D2H 前 k 列 → 组装 m×m 列主序（其余零）→ H2D
        let mut host_a = vec![0u8; m * n * elem];
        musa_ffi::check_musa(
            unsafe {
                musa_ffi::musaMemcpy(
                    host_a.as_mut_ptr() as *mut std::ffi::c_void,
                    a_ptr.as_ptr() as *const std::ffi::c_void,
                    m * n * elem,
                    musa_ffi::musaMemcpyKind::DeviceToHost,
                )
            },
            "musaMemcpy(D2H qr reflectors)",
        )?;
        let mut host_q = vec![0u8; m * m * elem];
        for j in 0..k {
            for i in 0..m {
                let src = &host_a[(i + j * m) * elem..(i + j * m + 1) * elem];
                host_q[(i + j * m) * elem..(i + j * m + 1) * elem].copy_from_slice(src);
            }
        }
        musa_ffi::check_musa(
            unsafe {
                musa_ffi::musaMemcpy(
                    q_ptr.as_ptr() as *mut std::ffi::c_void,
                    host_q.as_ptr() as *const std::ffi::c_void,
                    m * m * elem,
                    musa_ffi::musaMemcpyKind::HostToDevice,
                )
            },
            "musaMemcpy(H2D qr Q padded)",
        )?;
        math_handle::with_mublas_handle(&device, &out_stream, |handle| {
            let (mi, ni, ki) = (m as c_int, m as c_int, k as c_int);
            let mut ws_size: c_int = 0;
            let status = match dtype {
                Dtype::Float32 => unsafe {
                    musa_x_ffi::musolverSorgqr_bufferSize(mi, ni, ki, &mut ws_size)
                },
                Dtype::Float64 => unsafe {
                    musa_x_ffi::musolverDorgqr_bufferSize(mi, ni, ki, &mut ws_size)
                },
                _ => unreachable!("dtype already validated as float32/float64"),
            };
            musa_x_ffi::check_mublas(status, "musolver orgqr_bufferSize")?;
            let ws = math_handle::get_workspace(&device, ws_size as usize)?;
            let status = match dtype {
                Dtype::Float32 => unsafe {
                    musa_x_ffi::musolverSorgqr(
                        handle,
                        mi,
                        ni,
                        ki,
                        q_ptr.as_ptr() as *mut f32,
                        mi,
                        tau_dev as *const f32,
                        ws.ptr(),
                    )
                },
                Dtype::Float64 => unsafe {
                    musa_x_ffi::musolverDorgqr(
                        handle,
                        mi,
                        ni,
                        ki,
                        q_ptr.as_ptr() as *mut f64,
                        mi,
                        tau_dev as *const f64,
                        ws.ptr(),
                    )
                },
                _ => unreachable!("dtype already validated as float32/float64"),
            };
            musa_x_ffi::check_mublas(status, "musolver orgqr")
        })?;
        (q_data, [m, m])
    } else {
        // reduced：orgqr(m, k, k) 原地（m ≤ n 的 complete 亦如此：k=m）
        let (q_rows, q_cols) = if complete { (m, m) } else { (m, k) };
        math_handle::with_mublas_handle(&device, &out_stream, |handle| {
            let (mi, ni, ki) = (q_rows as c_int, q_cols as c_int, k as c_int);
            let mut ws_size: c_int = 0;
            let status = match dtype {
                Dtype::Float32 => unsafe {
                    musa_x_ffi::musolverSorgqr_bufferSize(mi, ni, ki, &mut ws_size)
                },
                Dtype::Float64 => unsafe {
                    musa_x_ffi::musolverDorgqr_bufferSize(mi, ni, ki, &mut ws_size)
                },
                _ => unreachable!("dtype already validated as float32/float64"),
            };
            musa_x_ffi::check_mublas(status, "musolver orgqr_bufferSize")?;
            let ws = math_handle::get_workspace(&device, ws_size as usize)?;
            let status = match dtype {
                Dtype::Float32 => unsafe {
                    musa_x_ffi::musolverSorgqr(
                        handle,
                        mi,
                        ni,
                        ki,
                        a_ptr.as_ptr() as *mut f32,
                        mi,
                        tau_dev as *const f32,
                        ws.ptr(),
                    )
                },
                Dtype::Float64 => unsafe {
                    musa_x_ffi::musolverDorgqr(
                        handle,
                        mi,
                        ni,
                        ki,
                        a_ptr.as_ptr() as *mut f64,
                        mi,
                        tau_dev as *const f64,
                        ws.ptr(),
                    )
                },
                _ => unreachable!("dtype already validated as float32/float64"),
            };
            musa_x_ffi::check_mublas(status, "musolver orgqr")
        })?;
        (cm_data, [q_rows, q_cols])
    };

    let q = strided_view(
        &q_data,
        q_shape,
        [1, m as isize],
        dtype,
        &device,
        &out_stream,
    );
    q.data().buffer().record_write(&out_stream);
    r.data().buffer().record_write(&out_stream);
    Ok((q, r))
}

/// `ms.svd(a, full_matrices=True, compute_uv=True)` → `(u, s, vh)`。
///
/// - `u` (m×m|m×k)、`s` (k,) 降序、`vh` (n×n|k×n)（NumPy 语义）
/// - `compute_uv=False`：仅返回 `s`（u/vh 为 None，Python 层折出单值）
/// - gesvd 的 V 输出即 Vᵀ（头文件 "stored as rows (transposed)"，
///   2026-08-07 探针 2 实锤），vh = strides `(1, n)` 视图
/// - 一律 ALL 模式 + 薄视图切片：SDK 3.1.0 的 SINGULAR 模式 U 输出有 bug
///   （6×4 复现损坏，探针 3），ALL 全形状正确
pub fn svd(
    a: &Array,
    full_matrices: bool,
    compute_uv: bool,
) -> Result<(Option<Array>, Array, Option<Array>)> {
    let op_name = "svd";
    require_musa(op_name, a.device())?;
    check_float_whitelist(op_name, a.dtype())?;
    if a.ndim() != 2 {
        return Err(ShapeError::Mismatch(format!(
            "{}: a must be 2-D, got ndim {}",
            op_name,
            a.ndim()
        ))
        .into());
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let k = m.min(n);
    let device = a.device().clone();
    let dtype = a.dtype();
    let elem = dtype.element_size();
    let out_stream: Arc<Stream> =
        resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream()));

    // 空矩阵早退（u/vh 按 compute_uv 返回空数组，NumPy 语义）
    if m == 0 || n == 0 {
        let s_buf = Buffer::alloc(1, device.clone(), &out_stream)?;
        let s = Array::new(
            BufferRef::new(Arc::new(s_buf)),
            Layout::from_shape(vec![k]),
            dtype,
            Arc::clone(&out_stream),
            DeviceResolution::new(device.clone(), ResolutionSource::InputArray),
            DtypeResolution::new(dtype, ResolutionSource::InputArray),
        );
        let u = if compute_uv {
            let u_cols = if full_matrices { m } else { k };
            let b = Buffer::alloc(1, device.clone(), &out_stream)?;
            Some(Array::new(
                BufferRef::new(Arc::new(b)),
                Layout::from_shape(vec![m, u_cols]),
                dtype,
                Arc::clone(&out_stream),
                DeviceResolution::new(device.clone(), ResolutionSource::InputArray),
                DtypeResolution::new(dtype, ResolutionSource::InputArray),
            ))
        } else {
            None
        };
        let vh = if compute_uv {
            let v_cols = if full_matrices { n } else { k };
            let b = Buffer::alloc(1, device.clone(), &out_stream)?;
            Some(Array::new(
                BufferRef::new(Arc::new(b)),
                Layout::from_shape(vec![v_cols, n]),
                dtype,
                out_stream,
                DeviceResolution::new(device, ResolutionSource::InputArray),
                DtypeResolution::new(dtype, ResolutionSource::InputArray),
            ))
        } else {
            None
        };
        return Ok((u, s, vh));
    }

    // svect 模式映射：compute_uv → ALL/ALL（薄输出 = 全尺寸缓冲的跨步视图
    // 切片，见下）；compute_uv=False → NONE/NONE。
    // 注 1：SDK 3.1.0 的 SINGULAR 模式有 bug——m>n 时 U 输出损坏
    // （2026-08-07 真机探针 3：6×4 下 UᵀU−I=4.5，OUTOFPLACE/INPLACE 均复现，
    // ALL 模式同一矩阵误差 1e-15）→ 一律走 ALL 并切片，绕开该模式。
    // 注 2：compute_uv=False 时仍分配薄尺寸 U/V 缓冲（NONE+NULL 未验证，
    // 稳妥给有效指针）。
    let left = if compute_uv {
        musa_x_ffi::MUBLAS_SVECT_ALL
    } else {
        musa_x_ffi::MUBLAS_SVECT_NONE
    };
    let right = left;
    // 输出视图尺寸：full → U (m,m)/Vh (n,n)；thin → U (m,k)/Vh (k,n)
    let u_cols = if compute_uv && full_matrices { m } else { k };
    let v_cols = if compute_uv && full_matrices { n } else { k };
    // 缓冲分配尺寸：compute_uv 时恒为全尺寸（ALL 输出）；否则薄尺寸占位
    let u_alloc = if compute_uv { m } else { k };
    let v_alloc = if compute_uv { n } else { k };
    // V 缓冲 leading dim：ALL → n（V' 列主序存储，头文件 "stored as rows"）；
    // NONE 时不被引用，传 k 即可（ldv >= 1 合法）
    let ldv = v_alloc as c_int;

    // 列主序物化副本（gesvd 原地破坏 A）
    let cm = col_major_copy(a, &out_stream)?;
    cm.data().buffer().wait_last_write_on(&out_stream)?;
    cm.data().buffer().record_read(&out_stream);
    record_op_context(
        "svd",
        &[a.shape()],
        &[a.device()],
        &[a.dtype()],
        &[k],
        &out_stream,
    );
    let (a_ptr, _) = device_ptr(cm.data())?;

    // 输出缓冲：S (k)、U (m×u_alloc)、V (n×v_alloc)、E (max(m,n))、info
    let s_ref = alloc_buf_ref(k * elem, &device, &out_stream)?;
    let u_ref = alloc_buf_ref(m * u_alloc * elem, &device, &out_stream)?;
    let v_ref = alloc_buf_ref(n * v_alloc * elem, &device, &out_stream)?;
    let e_ref = alloc_buf_ref(m.max(n) * elem, &device, &out_stream)?;
    let (s_ptr, _) = device_ptr(&s_ref)?;
    let (u_ptr, _) = device_ptr(&u_ref)?;
    let (v_ptr, _) = device_ptr(&v_ref)?;
    let (e_ptr, _) = device_ptr(&e_ref)?;
    let (info_dev, _info_guard) = alloc_dev_bytes(&device, 4)?;

    math_handle::with_mublas_handle(&device, &out_stream, |handle| {
        let (mi, ni) = (m as c_int, n as c_int);
        let mut ws_size: c_int = 0;
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgesvd_bufferSize(
                    left,
                    right,
                    mi,
                    ni,
                    1,
                    musa_x_ffi::MUBLAS_OUTOFPLACE,
                    &mut ws_size,
                )
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgesvd_bufferSize(
                    left,
                    right,
                    mi,
                    ni,
                    1,
                    musa_x_ffi::MUBLAS_OUTOFPLACE,
                    &mut ws_size,
                )
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver gesvd_bufferSize")?;
        let ws = math_handle::get_workspace(&device, ws_size as usize)?;
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgesvd(
                    handle,
                    left,
                    right,
                    mi,
                    ni,
                    a_ptr.as_ptr() as *mut f32,
                    mi,
                    s_ptr.as_ptr() as *mut f32,
                    u_ptr.as_ptr() as *mut f32,
                    mi,
                    v_ptr.as_ptr() as *mut f32,
                    ldv,
                    e_ptr.as_ptr() as *mut f32,
                    musa_x_ffi::MUBLAS_OUTOFPLACE,
                    info_dev as *mut c_int,
                    ws.ptr(),
                )
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgesvd(
                    handle,
                    left,
                    right,
                    mi,
                    ni,
                    a_ptr.as_ptr() as *mut f64,
                    mi,
                    s_ptr.as_ptr() as *mut f64,
                    u_ptr.as_ptr() as *mut f64,
                    mi,
                    v_ptr.as_ptr() as *mut f64,
                    ldv,
                    e_ptr.as_ptr() as *mut f64,
                    musa_x_ffi::MUBLAS_OUTOFPLACE,
                    info_dev as *mut c_int,
                    ws.ptr(),
                )
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver gesvd")
    })?;

    // 奇异值合理性校验（info 可靠性存疑的兜底：S 必须全部 ≥0 且有限）
    // 注：getrf 的 info 探针显示 SDK 3.1.0 可能不写 info；gesvd 探针显示写了 0。
    // 收敛失败时 S 会出现负值/NaN —— 此处兜底检测。
    let mut s_host = vec![0u8; k * elem];
    musa_ffi::check_musa(
        unsafe {
            musa_ffi::musaMemcpy(
                s_host.as_mut_ptr() as *mut std::ffi::c_void,
                s_ptr.as_ptr() as *const std::ffi::c_void,
                k * elem,
                musa_ffi::musaMemcpyKind::DeviceToHost,
            )
        },
        "musaMemcpy(D2H svd s)",
    )?;
    let s_ok = match dtype {
        Dtype::Float32 => s_host.chunks_exact(4).all(|b| {
            let v = f32::from_le_bytes(b.try_into().unwrap());
            v >= 0.0 && v.is_finite()
        }),
        Dtype::Float64 => s_host.chunks_exact(8).all(|b| {
            let v = f64::from_le_bytes(b.try_into().unwrap());
            v >= 0.0 && v.is_finite()
        }),
        _ => unreachable!("dtype already validated as float32/float64"),
    };
    if !s_ok {
        return Err(MusapyError::Device(DeviceError::MathLibCallFailed(
            format!(
                "{}: gesvd returned invalid singular values (negative/NaN; convergence failure?)",
                op_name
            ),
        )));
    }

    // s：独立 1D 连续数组
    let s = Array::new(
        s_ref,
        Layout::from_shape(vec![k]),
        dtype,
        Arc::clone(&out_stream),
        DeviceResolution::new(device.clone(), ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    );

    // u：strides (1, m) 视图（列主序 U 直接读；thin = 前 k 列切片）
    // vh：V 缓冲本身就是 Vᵀ（头文件 "stored as rows (transposed)"，探针 2
    // 实锤：V' 列主序重建误差 1e-15 vs V 解释 4.1）→ vh = strides (1, n) 视图
    // （thin = 前 k 行切片）
    let u = if compute_uv {
        Some(strided_view(
            &u_ref,
            [m, u_cols],
            [1, m as isize],
            dtype,
            &device,
            &out_stream,
        ))
    } else {
        None
    };
    let vh = if compute_uv {
        Some(strided_view(
            &v_ref,
            [v_cols, n],
            [1, n as isize],
            dtype,
            &device,
            &out_stream,
        ))
    } else {
        None
    };
    s.data().buffer().record_write(&out_stream);
    if let Some(ref ua) = u {
        ua.data().buffer().record_write(&out_stream);
    }
    if let Some(ref va) = vh {
        va.data().buffer().record_write(&out_stream);
    }
    Ok((u, s, vh))
}

// ── Phase 3 公共助手 ────────────────────────────────────────

/// 列主序物化副本：row-major A 的列主序存储 ≡ Aᵀ 的行主序物化
/// （transpose 视图 + contiguous，零新增 kernel）。
fn col_major_copy(a: &Array, stream: &Arc<Stream>) -> Result<Array> {
    let at = crate::indexing::transpose(a, None)?;
    let at_mat = crate::indexing::contiguous(&at)?;
    let _ = stream; // contiguous 内部已选流（当前流或输入流）
    Ok(at_mat)
}

/// 从 BufferRef 取设备指针。
fn device_ptr(buf: &BufferRef) -> Result<(NonNull<u8>, BufferRef)> {
    let ptr = buf.buffer().ptr().ok_or_else(|| {
        MusapyError::Device(DeviceError::MathLibCallFailed(
            "linalg decomp: null buffer pointer".into(),
        ))
    })?;
    Ok((ptr, buf.clone()))
}

/// 跨步视图（零拷贝）：列主序缓冲读为行主序标准布局。
#[allow(clippy::too_many_arguments)]
fn strided_view(
    buf: &BufferRef,
    shape: [usize; 2],
    strides: [isize; 2],
    dtype: Dtype,
    device: &Device,
    stream: &Arc<Stream>,
) -> Array {
    let layout = Layout {
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        offset: 0,
    };
    Array::new(
        buf.clone(),
        layout,
        dtype,
        Arc::clone(stream),
        DeviceResolution::new(device.clone(), ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    )
}

/// 设备内存分配（Buffer 路径，自动记账与流追踪）。
fn alloc_buf_ref(nbytes: usize, device: &Device, stream: &Arc<Stream>) -> Result<BufferRef> {
    let buf = Buffer::alloc(nbytes.max(1), device.clone(), stream)?;
    Ok(BufferRef::new(Arc::new(buf)))
}

/// musaMalloc 前绑定设备（同 Buffer::alloc / get_workspace 纪律）。
fn set_device_for_alloc(device: &Device) -> Result<()> {
    let Some(id) = device.musa_id() else {
        return Err(DeviceError::Mismatch("linalg: device has no musa id".into()).into());
    };
    musa_ffi::set_device(id as i32)
}

/// 裸设备内存分配（ipiv/info/tau 等 FFI 要求；RAII 释放）。
fn alloc_dev_bytes(device: &Device, nbytes: usize) -> Result<(*mut std::ffi::c_void, DevPtrGuard)> {
    set_device_for_alloc(device)?;
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    musa_ffi::check_musa(
        unsafe { musa_ffi::musaMalloc(&mut ptr, nbytes.max(1)) },
        "musaMalloc(linalg decomp scratch)",
    )?;
    Ok((ptr, DevPtrGuard(ptr)))
}

/// ipiv/info 成对分配（getrf 共用；RAII 释放）。
fn alloc_ipiv_info(
    device: &Device,
    ipiv_bytes: usize,
) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void, DevPtrGuard)> {
    set_device_for_alloc(device)?;
    let mut ipiv: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut info: *mut std::ffi::c_void = std::ptr::null_mut();
    musa_ffi::check_musa(
        unsafe { musa_ffi::musaMalloc(&mut ipiv, ipiv_bytes.max(1)) },
        "musaMalloc(linalg ipiv)",
    )?;
    musa_ffi::check_musa(
        unsafe { musa_ffi::musaMalloc(&mut info, 4) },
        "musaMalloc(linalg info)",
    )?;
    Ok((ipiv, info, DevPtrGuard(ipiv)))
}

/// 裸设备指针 RAII 释放。
struct DevPtrGuard(*mut std::ffi::c_void);

impl Drop for DevPtrGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { musa_ffi::musaFree(self.0) };
        }
    }
}

/// 共享 getrf：两段式（bufferSize → workspace）列主序 LU 分解（m×n）。
///
/// 返回 host 侧 ipiv（1-based，LAPACK 语义）；`ipiv_dev`/`info_dev` 由调用方
/// 分配与释放（solve 的 getrs 复用 ipiv）。solve 与 lu 共用（003-D4 重构）。
#[allow(clippy::too_many_arguments)]
fn gpu_getrf(
    device: &Device,
    stream: &Arc<Stream>,
    lu_ptr: NonNull<u8>,
    m: usize,
    n: usize,
    ipiv_dev: *mut std::ffi::c_void,
    info_dev: *mut std::ffi::c_void,
    dtype: Dtype,
) -> Result<Vec<c_int>> {
    let (mi, ni) = (m as c_int, n as c_int);
    let k = m.min(n);
    math_handle::with_mublas_handle(device, stream, |handle| {
        let mut ws_size: c_int = 0;
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgetrf_bufferSize(mi, ni, true, &mut ws_size)
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgetrf_bufferSize(mi, ni, true, &mut ws_size)
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver getrf_bufferSize")?;
        let ws = math_handle::get_workspace(device, ws_size as usize)?;
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgetrf(
                    handle,
                    mi,
                    ni,
                    lu_ptr.as_ptr() as *mut f32,
                    mi,
                    ipiv_dev as *mut c_int,
                    info_dev as *mut c_int,
                    ws.ptr(),
                )
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgetrf(
                    handle,
                    mi,
                    ni,
                    lu_ptr.as_ptr() as *mut f64,
                    mi,
                    ipiv_dev as *mut c_int,
                    info_dev as *mut c_int,
                    ws.ptr(),
                )
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver getrf")?;
        // D2H ipiv（1-based）
        let mut ipiv = vec![0 as c_int; k];
        musa_ffi::check_musa(
            unsafe {
                musa_ffi::musaMemcpy(
                    ipiv.as_mut_ptr() as *mut std::ffi::c_void,
                    ipiv_dev as *const std::ffi::c_void,
                    k * std::mem::size_of::<c_int>(),
                    musa_ffi::musaMemcpyKind::DeviceToHost,
                )
            },
            "musaMemcpy(D2H lu ipiv)",
        )?;
        Ok(ipiv)
    })
}

/// qr 的 R 提取：geqrf 后、orgqr 前；host 中转组装行主序 R，下三角清零。
/// `r_rows`：reduced=k / complete=m（complete 多出的行全零）。
#[allow(clippy::too_many_arguments)]
fn extract_upper_r(
    cm_data: &BufferRef,
    m: usize,
    n: usize,
    k: usize,
    r_rows: usize,
    dtype: Dtype,
    device: &Device,
    stream: &Arc<Stream>,
) -> Result<Array> {
    let elem = dtype.element_size();
    let Some(base) = cm_data.buffer().ptr() else {
        return Err(MusapyError::Device(DeviceError::MathLibCallFailed(
            "qr: null buffer pointer".into(),
        )));
    };
    // D2H 整个 m×n 列主序缓冲
    let mut host_cm = vec![0u8; m * n * elem];
    musa_ffi::check_musa(
        unsafe {
            musa_ffi::musaMemcpy(
                host_cm.as_mut_ptr() as *mut std::ffi::c_void,
                base.as_ptr() as *const std::ffi::c_void,
                m * n * elem,
                musa_ffi::musaMemcpyKind::DeviceToHost,
            )
        },
        "musaMemcpy(D2H qr R)",
    )?;
    // 组装行主序 R（r_rows×n）：R[i][j] = cm[i + j*m]（i<k 且 i<=j；其余 0）
    let mut host_r = vec![0u8; r_rows * n * elem];
    for i in 0..r_rows {
        for j in 0..n {
            if i < k && i <= j {
                let src = &host_cm[(i + j * m) * elem..(i + j * m + 1) * elem];
                host_r[(i * n + j) * elem..(i * n + j + 1) * elem].copy_from_slice(src);
            }
        }
    }
    // H2D → Array（连续行主序）
    let buf_ref = alloc_buf_ref(r_rows * n * elem, device, stream)?;
    let Some(dst) = buf_ref.buffer().ptr() else {
        return Err(MusapyError::Device(DeviceError::MathLibCallFailed(
            "qr: null R buffer pointer".into(),
        )));
    };
    musa_ffi::check_musa(
        unsafe {
            musa_ffi::musaMemcpy(
                dst.as_ptr() as *mut std::ffi::c_void,
                host_r.as_ptr() as *const std::ffi::c_void,
                r_rows * n * elem,
                musa_ffi::musaMemcpyKind::HostToDevice,
            )
        },
        "musaMemcpy(H2D qr R)",
    )?;
    Ok(Array::new(
        buf_ref,
        Layout::from_shape(vec![r_rows, n]),
        dtype,
        Arc::clone(stream),
        DeviceResolution::new(device.clone(), ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

/// host ipiv（1-based int32）→ int64 设备数组。
fn int64_array_from(
    ipiv: &[c_int],
    k: usize,
    device: &Device,
    stream: &Arc<Stream>,
) -> Result<Array> {
    let data: Vec<i64> = ipiv.iter().map(|&v| v as i64).collect();
    let buf_ref = alloc_buf_ref(k * 8, device, stream)?;
    let Some(dst) = buf_ref.buffer().ptr() else {
        return Err(MusapyError::Device(DeviceError::MathLibCallFailed(
            "lu: null piv buffer pointer".into(),
        )));
    };
    musa_ffi::check_musa(
        unsafe {
            musa_ffi::musaMemcpy(
                dst.as_ptr() as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                k * 8,
                musa_ffi::musaMemcpyKind::HostToDevice,
            )
        },
        "musaMemcpy(H2D lu pivots)",
    )?;
    Ok(Array::new(
        buf_ref,
        Layout::from_shape(vec![k]),
        Dtype::Int64,
        Arc::clone(stream),
        DeviceResolution::new(device.clone(), ResolutionSource::InputArray),
        DtypeResolution::new(Dtype::Int64, ResolutionSource::InputArray),
    ))
}

// ============================================================
// 测试（形状推导纯逻辑 + mock 端到端）
// ============================================================

#[cfg(test)]
mod tests {
    use super::{final_matmul_shape, matmul_shapes};

    // ── 形状推导（纯逻辑，无需设备）──

    #[test]
    fn test_matmul_shapes_2d_2d() {
        let (a2, b2, sf, sl) = matmul_shapes(&[3, 4], &[4, 5]).unwrap();
        assert_eq!(a2, [3, 4]);
        assert_eq!(b2, [4, 5]);
        assert!(!sf && !sl);
        assert_eq!(final_matmul_shape(&[3, 5], sf, sl), vec![3, 5]);
    }

    #[test]
    fn test_matmul_shapes_1d_left() {
        let (a2, b2, sf, sl) = matmul_shapes(&[4], &[4, 5]).unwrap();
        assert_eq!(a2, [1, 4]);
        assert_eq!(b2, [4, 5]);
        assert!(sf && !sl);
        assert_eq!(final_matmul_shape(&[1, 5], sf, sl), vec![5]);
    }

    #[test]
    fn test_matmul_shapes_1d_right() {
        let (a2, b2, sf, sl) = matmul_shapes(&[3, 4], &[4]).unwrap();
        assert_eq!(a2, [3, 4]);
        assert_eq!(b2, [4, 1]);
        assert!(!sf && sl);
        assert_eq!(final_matmul_shape(&[3, 1], sf, sl), vec![3]);
    }

    #[test]
    fn test_matmul_shapes_1d_1d() {
        let (a2, b2, sf, sl) = matmul_shapes(&[4], &[4]).unwrap();
        assert_eq!(a2, [1, 4]);
        assert_eq!(b2, [4, 1]);
        assert!(sf && sl);
        assert_eq!(final_matmul_shape(&[1, 1], sf, sl), Vec::<usize>::new());
    }

    #[test]
    fn test_matmul_shapes_errors() {
        assert!(matmul_shapes(&[3, 4], &[5, 6]).is_err()); // 内维不匹配
        assert!(matmul_shapes(&[], &[4]).is_err()); // 0-dim
        assert!(matmul_shapes(&[2, 3, 4], &[4, 5]).is_err()); // 3D batch
        assert!(matmul_shapes(&[4, 5], &[2, 3, 4]).is_err()); // 3D batch
    }

    // ── GPU mock 端到端（仅 musapy_mock_musa：验证形状管线 + mock 数值）──

    #[cfg(musapy_mock_musa)]
    mod gpu_mock_e2e {
        use super::super::{dot, matmul, solve};
        use crate::creation;
        use musapy_core::{Array, Device, DeviceError, Dtype};

        /// 从 Array buffer 读出 f64 数据（mock 设备内存即宿主内存）。
        fn read_f64(a: &Array) -> Vec<f64> {
            let n = a.size().max(1);
            let mut out = vec![0f64; n];
            if let Some(ptr) = a.data().buffer().ptr() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ptr.as_ptr() as *const u8,
                        out.as_mut_ptr() as *mut u8,
                        a.size() * 8,
                    );
                }
            }
            out
        }

        fn make_musa_f64(shape: &[usize], data: &[f64]) -> Array {
            let dev = Device::Musa(0);
            let a = creation::zeros(shape, Some(Dtype::Float64), Some(dev)).unwrap();
            // mock 模式下设备内存即宿主内存，可直接写
            if let Some(ptr) = a.data().buffer().ptr() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        ptr.as_ptr(),
                        data.len() * 8,
                    );
                }
            }
            a
        }

        #[test]
        fn matmul_mock_shapes_and_fill() {
            // mock Sgemm/Dgemm 把 C 全填 1.0——验证形状管线端到端
            let a = make_musa_f64(&[2, 3], &[0.0; 6]);
            let b = make_musa_f64(&[3, 4], &[0.0; 12]);
            let c = matmul(&a, &b, None).unwrap();
            assert_eq!(c.shape(), &vec![2, 4]);
            assert_eq!(read_f64(&c), vec![1.0; 8]);

            // 1D 组合 squeeze
            let v = make_musa_f64(&[3], &[0.0; 3]);
            let r = matmul(&v, &b, None).unwrap();
            assert_eq!(r.shape(), &vec![4]);
            let r2 = matmul(&a, &v, None).unwrap();
            assert_eq!(r2.shape(), &vec![2]);
            let r3 = matmul(&v, &make_musa_f64(&[3], &[0.0; 3]), None).unwrap();
            assert_eq!(r3.shape(), &Vec::<usize>::new());
        }

        #[test]
        fn dot_mock_returns_n() {
            let a = make_musa_f64(&[5], &[0.0; 5]);
            let b = make_musa_f64(&[5], &[0.0; 5]);
            let r = dot(&a, &b, None).unwrap();
            assert_eq!(r.shape(), &Vec::<usize>::new());
            assert_eq!(read_f64(&r)[0], 5.0); // mock Ddot 返回 n
        }

        #[test]
        fn solve_mock_passthrough() {
            // mock getrf/getrs 不改数据：x ≡ b
            let a = make_musa_f64(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
            let b = make_musa_f64(&[2], &[7.0, 9.0]);
            let x = solve(&a, &b).unwrap();
            assert_eq!(x.shape(), &vec![2]);
            assert_eq!(read_f64(&x), vec![7.0, 9.0]);

            // 2D rhs
            let b2 = make_musa_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let x2 = solve(&a, &b2).unwrap();
            assert_eq!(x2.shape(), &vec![2, 2]);
            assert_eq!(read_f64(&x2), vec![1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn cpu_input_rejected() {
            // v0.3 GPU-only 策略：CPU 设备输入 → DeviceError（require_musa）
            let a = make_musa_f64(&[2, 2], &[0.0; 4]);
            let b_cpu = creation::zeros(&[2, 2], Some(Dtype::Float64), Some(Device::Cpu)).unwrap();

            // 跨设备混合输入：device mismatch 先报
            assert!(matches!(
                matmul(&a, &b_cpu, None).unwrap_err(),
                musapy_core::MusapyError::Device(DeviceError::Mismatch(_))
            ));
            assert!(dot(&a, &b_cpu, None).is_err());
            assert!(solve(&a, &b_cpu).is_err());

            // 同设备 CPU 输入：require_musa 拒绝
            let a_cpu = creation::zeros(&[2, 2], Some(Dtype::Float64), Some(Device::Cpu)).unwrap();
            let err = matmul(&a_cpu, &b_cpu, None).unwrap_err();
            assert!(matches!(
                err,
                musapy_core::MusapyError::Device(DeviceError::Mismatch(msg))
                    if msg.contains("GPU-only")
            ));
            assert!(dot(&a_cpu, &b_cpu, None).is_err());
            assert!(solve(&a_cpu, &b_cpu).is_err());
        }

        #[test]
        fn matmul_mock_k_zero() {
            // k=0 退化：输出全零（fill_zeros 路径）
            let a = make_musa_f64(&[2, 0], &[]);
            let b = make_musa_f64(&[0, 3], &[]);
            let c = matmul(&a, &b, None).unwrap();
            assert_eq!(c.shape(), &vec![2, 3]);
            assert_eq!(read_f64(&c), vec![0.0; 6]);
        }
    }
}
