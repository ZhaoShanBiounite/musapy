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
//!   - **solve 奇异检测含一次同步**：getrf 后 D2H 读回 LU 对角线判奇异
//!     （musaMemcpy2D 单次拷贝；muSOLVER 3.1.0 不写 info 输出，见 gpu_solve
//!     注释）——solve 独有的同步点（matmul/dot 无）。
//!   - **输入连续性**：非连续输入（transpose 视图等）先经 `indexing::contiguous`
//!     物化（不做 strided-gemm 优化，v0.4 评估）。
//!   - **matmul 形状语义**（NumPy 对齐）：1D 侧补 1 → 2D gemm → 对应轴 squeeze
//!     （零拷贝视图）；3D+ batch matmul 范围外（v0.3 计划 §1.2）。

use musapy_core::error::{DeviceError, DtypeError, LinAlgError, MemoryError, ShapeError};
use musapy_core::musa_ffi;
use musapy_core::resolution;
use musapy_core::{
    Array, Buffer, BufferRef, Device, DeviceResolution, Dtype, DtypeResolution, Layout, OpContext,
    ResolutionSource, Result, Stream, math_handle, musa_x_ffi, promote,
};
use std::ffi::c_int;
use std::ptr::NonNull;
use std::sync::Arc;

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
fn require_musa(op_name: &str, device: &Device) -> Result<()> {
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
fn check_float_whitelist(op_name: &str, dtype: Dtype) -> Result<()> {
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

    math_handle::with_mublas_handle(device, stream, |handle| {
        // ── getrf：LU 分解（两段式：bufferSize → workspace → 计算）──
        let mut lu_ws: c_int = 0;
        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgetrf_bufferSize(ni, ni, true, &mut lu_ws)
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgetrf_bufferSize(ni, ni, true, &mut lu_ws)
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver getrf_bufferSize")?;

        let ws = math_handle::get_workspace(device, lu_ws as usize)?;

        // ipiv/info 设备分配（musaMalloc，绑定当前设备）
        let mut ipiv_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut info_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        musa_ffi::check_musa(
            unsafe { musa_ffi::musaMalloc(&mut ipiv_dev, ipiv_bytes.max(1)) },
            "musaMalloc(solve ipiv)",
        )?;
        musa_ffi::check_musa(
            unsafe { musa_ffi::musaMalloc(&mut info_dev, 4) },
            "musaMalloc(solve info)",
        )?;
        // RAII guard：任何错误路径都释放 ipiv/info
        struct DevPtrGuard(*mut std::ffi::c_void);
        impl Drop for DevPtrGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { musa_ffi::musaFree(self.0) };
                }
            }
        }
        let _ipiv_guard = DevPtrGuard(ipiv_dev);
        let _info_guard = DevPtrGuard(info_dev);

        let status = match dtype {
            Dtype::Float32 => unsafe {
                musa_x_ffi::musolverSgetrf(
                    handle,
                    ni,
                    ni,
                    lu_ptr.as_ptr() as *mut f32,
                    ni,
                    ipiv_dev as *mut c_int,
                    info_dev as *mut c_int,
                    ws.ptr(),
                )
            },
            Dtype::Float64 => unsafe {
                musa_x_ffi::musolverDgetrf(
                    handle,
                    ni,
                    ni,
                    lu_ptr.as_ptr() as *mut f64,
                    ni,
                    ipiv_dev as *mut c_int,
                    info_dev as *mut c_int,
                    ws.ptr(),
                )
            },
            _ => unreachable!("dtype already validated as float32/float64"),
        };
        musa_x_ffi::check_mublas(status, "musolver getrf")?;

        // ── 奇异检测（solve 独有的同步点）──
        // muSOLVER 3.1.0 的 getrf **从不写 info 输出**（2026-08-07 真机 C 探针
        // 实测：非奇异与奇异矩阵 info 均保持预填值，status=0、ipiv 正常）——
        // 故不能依赖 info 判奇异。改用 LAPACK 等价判据：info = 首个
        // U(k,k) == 0 的行号（LAPACK 正是 `IF A(K,K).EQ.ZERO → info=K`）。
        // U 的对角元素在 LU buffer（列主序 M=Aᵀ）偏移 k·(n+1) 处，跨步
        // (n+1)·elem；用 musaMemcpy2D 单次读回（1 调用，n·elem 字节）。
        stream.synchronize()?;
        let mut diag_host = vec![0u8; n * elem];
        musa_ffi::check_musa(
            unsafe {
                musa_ffi::musaMemcpy2D(
                    diag_host.as_mut_ptr() as *mut std::ffi::c_void,
                    elem,
                    lu_ptr.as_ptr() as *const std::ffi::c_void,
                    (n + 1) * elem,
                    elem,
                    n,
                    musa_ffi::musaMemcpyKind::DeviceToHost,
                )
            },
            "musaMemcpy2D(D2H solve LU diagonal)",
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
