//! FFT 算子（v0.3-alpha Phase 5，ADR-003 003-D5/D7）
//!
//! `ms.fft.fft` / `ms.fft.ifft`（C2C/Z2Z，axis=-1）+ `ms.fft.rfft`（R2C/D2Z，
//! 输出形状 N//2+1），**GPU-only**（003-D4 修订：数学库算子不建 CPU fallback）。
//! 实现走 muFFT：plan 经 `math_handle::with_mufft_plan` 按 `MufftPlanSpec` 池化复用。
//!
//! 本轮范围（用户确认，2026-08-08）：
//!   - **axis=-1 起步**：只支持沿最后一维（内存连续维）；`axis != -1` 抛错
//!     （fftn/多轴推迟到 v0.3 后期）。
//!   - `n` 截断/补零：`n < last_dim` 截断、`n > last_dim` 补零（resize kernel）。
//!   - `norm`："backward"（默认）/ "ortho" / "forward"（NumPy 完整语义）。
//!   - `out=`：支持（对齐 linalg 的 check_out 惯例）。
//!
//! 数值语义（对齐 NumPy）：
//!   - `fft` 无缩放；`ifft` 缩放 1/N（backward 时；norm 变换见 `FftNorm::scale`）。
//!   - real 输入 → 先扩展为 complex（re=x, im=0，复用 cast f32→c64 / f64→c128）。
//!   - `rfft` 只做 forward，输出前 N//2+1 个。
//!   - mock 模式：mufft stub 用 naive O(N²) DFT 数值仿真（无 GPU CI 对照 np.fft）。

use musapy_core::math_handle;
use musapy_core::musa_ffi;
use musapy_core::musa_x_ffi::{
    MUFFT_C2C, MUFFT_D2Z, MUFFT_FORWARD, MUFFT_INVERSE, MUFFT_R2C, MUFFT_Z2Z, muComplex,
    muDoubleComplex,
};
use musapy_core::resolution;
use musapy_core::{Array, Buffer, BufferRef, Device, Dtype, Layout, Result, Stream, musa_x_ffi};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::kernels;
use crate::linalg::require_musa;

/// NumPy norm 参数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FftNorm {
    Backward,
    Ortho,
    Forward,
}

impl FftNorm {
    /// 解析 norm 参数（NumPy 三值）。
    pub fn parse(s: Option<&str>) -> Result<Self> {
        match s {
            None | Some("backward") => Ok(FftNorm::Backward),
            Some("ortho") => Ok(FftNorm::Ortho),
            Some("forward") => Ok(FftNorm::Forward),
            Some(other) => Err(musapy_core::error::ShapeError::Mismatch(format!(
                "fft: invalid norm {other:?} (expected 'backward' | 'ortho' | 'forward')"
            ))
            .into()),
        }
    }

    /// 变换后输出缩放因子。
    /// NumPy：ifft 的 backward 缩放 1/N、ortho 缩放 1/sqrt(N)、forward 缩放 1；
    /// fft 的 backward 缩放 1、ortho 缩放 1/sqrt(N)、forward 缩放 1/N。
    fn scale(self, forward: bool, n: usize) -> f64 {
        let n = n as f64;
        match (self, forward) {
            (FftNorm::Backward, true) => 1.0,
            (FftNorm::Backward, false) => 1.0 / n,
            (FftNorm::Ortho, _) => 1.0 / n.sqrt(),
            (FftNorm::Forward, true) => 1.0 / n,
            (FftNorm::Forward, false) => 1.0,
        }
    }
}

/// 输出 complex dtype（real 输入按 real 宽度；complex 输入同宽）。
fn out_dtype_of(a: &Array) -> Dtype {
    match a.dtype() {
        Dtype::Complex64 => Dtype::Complex64,
        Dtype::Complex128 => Dtype::Complex128,
        Dtype::Float32 => Dtype::Complex64,
        Dtype::Float64 => Dtype::Complex128,
        other => unreachable!("fft input dtype validated: {other}"),
    }
}

/// `fft(a, n=None, axis=-1, norm="backward", out=None)` — 复数 FFT（axis=-1）。
pub fn fft(
    a: &Array,
    n: Option<usize>,
    axis: i32,
    norm: FftNorm,
    out: Option<&Array>,
) -> Result<Array> {
    fft_impl(a, n, axis, norm, true, false, out)
}

/// `ifft(a, n=None, axis=-1, norm="backward", out=None)` — 逆变换（缩放 1/N）。
pub fn ifft(
    a: &Array,
    n: Option<usize>,
    axis: i32,
    norm: FftNorm,
    out: Option<&Array>,
) -> Result<Array> {
    fft_impl(a, n, axis, norm, false, false, out)
}

/// `rfft(a, n=None, axis=-1, norm="backward", out=None)` — 实输入 FFT，输出 N//2+1。
pub fn rfft(
    a: &Array,
    n: Option<usize>,
    axis: i32,
    norm: FftNorm,
    out: Option<&Array>,
) -> Result<Array> {
    if matches!(a.dtype(), Dtype::Complex64 | Dtype::Complex128) {
        return Err(musapy_core::error::DtypeError::Unsupported(
            "rfft: input must be real (got complex; use fft for complex input)".into(),
        )
        .into());
    }
    fft_impl(a, n, axis, norm, true, true, out)
}

/// fft/ifft/rfft 的公共骨架（3-phase，GPU-only）。
#[allow(clippy::too_many_arguments)]
fn fft_impl(
    a: &Array,
    n_arg: Option<usize>,
    axis: i32,
    norm: FftNorm,
    forward: bool,
    real_input: bool,
    out: Option<&Array>,
) -> Result<Array> {
    let op_name = if real_input {
        "rfft"
    } else if forward {
        "fft"
    } else {
        "ifft"
    };

    // ═══════════════════════════════════════════════════════════════
    // Phase A：参数解析（一次性，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    // 1. Device 校验 + GPU-only（003-D4）
    let device = a.device().clone();
    require_musa(op_name, &device)?;

    // 2. 输入 dtype 白名单（real f32/f64 或 complex；rfft 已在上层拒绝 complex）
    if !matches!(
        a.dtype(),
        Dtype::Float32 | Dtype::Float64 | Dtype::Complex64 | Dtype::Complex128
    ) {
        return Err(musapy_core::error::DtypeError::Unsupported(format!(
            "{}: dtype {} not supported (fft whitelist: float32/float64/complex64/complex128)",
            op_name,
            a.dtype()
        ))
        .into());
    }

    // 3. axis 校验（axis=-1 起步；非最后一维抛错）
    let ndim = a.shape().len();
    if ndim == 0 {
        return Err(musapy_core::error::ShapeError::Mismatch(
            "fft: input must have at least 1 dimension (got 0-dim scalar)".into(),
        )
        .into());
    }
    let axis_u = if axis < 0 {
        (axis + ndim as i32) as usize
    } else {
        axis as usize
    };
    if axis_u != ndim - 1 {
        return Err(musapy_core::error::ShapeError::Mismatch(format!(
            "{}: axis {} not supported yet (Phase 5 axis=-1 only; multi-axis/fftn deferred)",
            op_name, axis
        ))
        .into());
    }
    let last_dim = a.shape()[ndim - 1];

    // 4. n 解析（None → 输入长度）；输出末维：rfft → N//2+1，fft/ifft → n
    let n = n_arg.unwrap_or(last_dim);
    let result_len = if real_input { n / 2 + 1 } else { n };
    let out_shape: Vec<usize> = {
        let mut s = a.shape()[..ndim - 1].to_vec();
        s.push(result_len);
        s
    };
    let out_dtype = out_dtype_of(a);
    let outer: usize = a.shape()[..ndim - 1].iter().product();

    // 5. out= 校验（shape/dtype/device；对齐 linalg::check_out 惯例）
    if let Some(o) = out {
        if o.shape() != &out_shape {
            return Err(musapy_core::error::ShapeError::Mismatch(format!(
                "{}: out shape {:?} != expected {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != out_dtype {
            return Err(musapy_core::error::DtypeError::Unsupported(format!(
                "{}: out dtype {} != expected {}",
                op_name,
                o.dtype(),
                out_dtype
            ))
            .into());
        }
        if o.device() != &device {
            return Err(musapy_core::error::DeviceError::Mismatch(format!(
                "{}: out device {} != input device {}",
                op_name,
                o.device(),
                device
            ))
            .into());
        }
    }

    // 6. Stream 选择（ADR L1-8）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 7. 输入预处理 → 连续 buffer [outer, n]
    //    - fft/ifft：complex 目标（complex 借用 / resize；real cast → complex）
    //    - rfft：real 目标（R2C/D2Z 的输入必须是 real buffer；real 借用 / real resize）
    let (work_data, work_ptr) =
        prepare_input(a, n, last_dim, out_dtype, real_input, &device, &out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：FFT 执行
    // ═══════════════════════════════════════════════════════════════

    let scale = norm.scale(forward, n);
    let result_data: BufferRef;
    let result_ptr: Option<NonNull<u8>>;
    // 每行输出 complex 数（rfft → N//2+1，已在 Phase A 算出）
    let out_nbytes = outer * result_len * out_dtype.element_size();
    let (res_data, res_ptr) =
        alloc_or_reuse_out(out, &out_shape, out_nbytes, &device, &out_stream)?;
    result_data = res_data;
    result_ptr = res_ptr;

    if outer * result_len > 0 {
        let ftype = match out_dtype {
            Dtype::Complex64 => {
                if real_input {
                    MUFFT_R2C
                } else {
                    MUFFT_C2C
                }
            }
            Dtype::Complex128 => {
                if real_input {
                    MUFFT_D2Z
                } else {
                    MUFFT_Z2Z
                }
            }
            _ => unreachable!(),
        };
        // 统一为 c_void 指针，Exec 调用处按 out_dtype 转回具体 complex 类型
        let src = work_ptr.map(|p| p.as_ptr() as *mut std::ffi::c_void);
        let dst = result_ptr.map(|p| p.as_ptr() as *mut std::ffi::c_void);
        let spec = math_handle::MufftPlanSpec::OneD {
            nx: n as i32,
            ftype,
            batch: 1,
        };
        math_handle::with_mufft_plan(&device, &out_stream, &spec, |plan| {
            let (Some(src), Some(dst)) = (src, dst) else {
                return Err(musapy_core::error::DeviceError::MathLibCallFailed(
                    "fft: null buffer pointer".into(),
                )
                .into());
            };
            // Plan1d batch=1 只处理一行：逐行执行 Exec（外层维逐行偏移）。
            // mock 的 naive DFT 与真机 muFFT 均按单行语义。
            let src_elem = a.dtype().element_size();
            let dst_elem = out_dtype.element_size();
            for row in 0..outer {
                let src_row = unsafe { src.add(row * n * src_elem) };
                let dst_row = unsafe { dst.add(row * result_len * dst_elem) };
                let status = match (out_dtype, real_input) {
                    (Dtype::Complex64, true) => unsafe {
                        musa_x_ffi::mufftExecR2C(
                            plan,
                            src_row as *mut f32,
                            dst_row as *mut muComplex,
                        )
                    },
                    (Dtype::Complex64, false) => unsafe {
                        musa_x_ffi::mufftExecC2C(
                            plan,
                            src_row as *mut muComplex,
                            dst_row as *mut muComplex,
                            if forward {
                                MUFFT_FORWARD
                            } else {
                                MUFFT_INVERSE
                            },
                        )
                    },
                    (Dtype::Complex128, true) => unsafe {
                        musa_x_ffi::mufftExecD2Z(
                            plan,
                            src_row as *mut f64,
                            dst_row as *mut muDoubleComplex,
                        )
                    },
                    (Dtype::Complex128, false) => unsafe {
                        musa_x_ffi::mufftExecZ2Z(
                            plan,
                            src_row as *mut muDoubleComplex,
                            dst_row as *mut muDoubleComplex,
                            if forward {
                                MUFFT_FORWARD
                            } else {
                                MUFFT_INVERSE
                            },
                        )
                    },
                    _ => unreachable!(),
                };
                musa_x_ffi::check_mufft(status, "mufftExec")?;
            }
            // 归一化：对 [outer, result_len] 结果就地缩放
            if scale != 1.0 {
                let total = outer * result_len;
                match out_dtype {
                    Dtype::Complex64 => unsafe {
                        kernels::musapy_scale_c64_v2(
                            dst as *mut muComplex,
                            scale,
                            total,
                            out_stream.raw(),
                        )
                    },
                    Dtype::Complex128 => unsafe {
                        kernels::musapy_scale_c128_v2(
                            dst as *mut muDoubleComplex,
                            scale,
                            total,
                            out_stream.raw(),
                        )
                    },
                    _ => unreachable!(),
                };
                musa_ffi::check_last_kernel_launch("fft_scale")?;
            }
            Ok(())
        })?;
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理 + 组装
    // ═══════════════════════════════════════════════════════════════

    work_data.buffer().record_read(&out_stream);
    result_data.buffer().record_write(&out_stream);

    if musapy_core::debug::is_debug() {
        let mut ctx = musapy_core::OpContext::new(
            op_name,
            vec![a.shape().clone()],
            vec![device.clone()],
            vec![a.dtype()],
            out_shape.clone(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }

    Ok(Array::new(
        result_data,
        Layout::from_shape(out_shape),
        out_dtype,
        out_stream,
        musapy_core::DeviceResolution::new(device, musapy_core::ResolutionSource::InputArray),
        musapy_core::DtypeResolution::new(out_dtype, musapy_core::ResolutionSource::InputArray),
    ))
}

/// 输入预处理 → 连续 buffer [outer, n]（返回 data_ref + 调整后指针）。
///
/// - fft/ifft（real_input=false）：输出 complex buffer；real 输入先 cast 扩 complex。
/// - rfft（real_input=true）：输出 real buffer（R2C/D2Z 的输入必须是 real）。
fn prepare_input(
    a: &Array,
    n: usize,
    last_dim: usize,
    out_dtype: Dtype,
    real_input: bool,
    device: &Device,
    out_stream: &Arc<Stream>,
) -> Result<(BufferRef, Option<NonNull<u8>>)> {
    match a.dtype() {
        Dtype::Complex64 | Dtype::Complex128 => {
            // complex 输入：借用连续视图（n == last_dim）或 resize（截断/补零）
            if n == last_dim {
                let contig = crate::indexing::contiguous(a)?;
                let ptr = adjusted_ptr(&contig, out_dtype);
                Ok((contig.data().clone(), ptr))
            } else {
                let nbytes = a.shape()[..a.shape().len() - 1].iter().product::<usize>()
                    * n
                    * out_dtype.element_size();
                let buffer = Buffer::alloc(nbytes, device.clone(), out_stream)?;
                let data_ref = BufferRef::new(Arc::new(buffer));
                a.data().buffer().wait_last_write_on(out_stream)?;
                let ptr = data_ref.buffer().ptr();
                launch_resize(a, ptr, n, out_dtype, out_stream)?;
                Ok((data_ref, ptr))
            }
        }
        Dtype::Float32 | Dtype::Float64 => {
            if real_input {
                // rfft：保持 real 输入（借用连续视图或 real resize）
                if n == last_dim {
                    let contig = crate::indexing::contiguous(a)?;
                    let ptr = adjusted_ptr(&contig, a.dtype());
                    Ok((contig.data().clone(), ptr))
                } else {
                    let nbytes = a.shape()[..a.shape().len() - 1].iter().product::<usize>()
                        * n
                        * a.dtype().element_size();
                    let buffer = Buffer::alloc(nbytes, device.clone(), out_stream)?;
                    let data_ref = BufferRef::new(Arc::new(buffer));
                    a.data().buffer().wait_last_write_on(out_stream)?;
                    let ptr = data_ref.buffer().ptr();
                    launch_real_resize(a, ptr, n, out_stream)?;
                    Ok((data_ref, ptr))
                }
            } else {
                // fft/ifft：real → complex 扩展（re=x, im=0）
                let casted = crate::op_builder::cast_array(a, out_dtype, out_stream)?;
                if n == last_dim {
                    let ptr = adjusted_ptr(&casted, out_dtype);
                    Ok((casted.data().clone(), ptr))
                } else {
                    let nbytes = a.shape()[..a.shape().len() - 1].iter().product::<usize>()
                        * n
                        * out_dtype.element_size();
                    let buffer = Buffer::alloc(nbytes, device.clone(), out_stream)?;
                    let data_ref = BufferRef::new(Arc::new(buffer));
                    casted.data().buffer().wait_last_write_on(out_stream)?;
                    let ptr = data_ref.buffer().ptr();
                    launch_resize(&casted, ptr, n, out_dtype, out_stream)?;
                    Ok((data_ref, ptr))
                }
            }
        }
        _ => unreachable!("fft input dtype validated"),
    }
}

/// 输入 complex 数组 → 连续 [outer, n]（截断/补零）。输入为完整 shape [..., last_dim]。
fn launch_resize(
    a: &Array,
    dst: Option<NonNull<u8>>,
    n: usize,
    out_dtype: Dtype,
    stream: &Arc<Stream>,
) -> Result<()> {
    let ndim = a.shape().len();
    let last_dim = a.shape()[ndim - 1];
    let a_ptr = adjusted_ptr(a, a.dtype());
    let shape = a.shape().to_vec();
    let a_strides = a.layout().strides.clone();
    let ndim_i = ndim as i32;
    let stream_raw = stream.raw();
    let (Some(ap), Some(cp)) = (a_ptr, dst) else {
        return Ok(());
    };
    match (out_dtype, a.dtype()) {
        (Dtype::Complex64, Dtype::Complex64) => unsafe {
            kernels::musapy_resize_c64_v2(
                ap.as_ptr() as *const muComplex,
                cp.as_ptr() as *mut muComplex,
                ndim_i,
                shape.as_ptr(),
                a_strides.as_ptr(),
                last_dim,
                n,
                stream_raw,
            )
        },
        (Dtype::Complex128, Dtype::Complex128) => unsafe {
            kernels::musapy_resize_c128_v2(
                ap.as_ptr() as *const muDoubleComplex,
                cp.as_ptr() as *mut muDoubleComplex,
                ndim_i,
                shape.as_ptr(),
                a_strides.as_ptr(),
                last_dim,
                n,
                stream_raw,
            )
        },
        _ => unreachable!("resize input is complex matching out_dtype"),
    }
    musa_ffi::check_last_kernel_launch("fft_resize")
}

/// real 输入 → 连续 [outer, n]（截断/补零；rfft 的 n 参数用，输入保持 real）。
fn launch_real_resize(
    a: &Array,
    dst: Option<NonNull<u8>>,
    n: usize,
    stream: &Arc<Stream>,
) -> Result<()> {
    let ndim = a.shape().len();
    let last_dim = a.shape()[ndim - 1];
    let a_ptr = adjusted_ptr(a, a.dtype());
    let shape = a.shape().to_vec();
    let a_strides = a.layout().strides.clone();
    let ndim_i = ndim as i32;
    let stream_raw = stream.raw();
    let (Some(ap), Some(cp)) = (a_ptr, dst) else {
        return Ok(());
    };
    match a.dtype() {
        Dtype::Float32 => unsafe {
            kernels::musapy_resize_f32_real_v2(
                ap.as_ptr() as *const f32,
                cp.as_ptr() as *mut f32,
                ndim_i,
                shape.as_ptr(),
                a_strides.as_ptr(),
                last_dim,
                n,
                stream_raw,
            )
        },
        Dtype::Float64 => unsafe {
            kernels::musapy_resize_f64_real_v2(
                ap.as_ptr() as *const f64,
                cp.as_ptr() as *mut f64,
                ndim_i,
                shape.as_ptr(),
                a_strides.as_ptr(),
                last_dim,
                n,
                stream_raw,
            )
        },
        _ => unreachable!("real resize only for f32/f64 input"),
    }
    musa_ffi::check_last_kernel_launch("fft_real_resize")
}

/// 输出 buffer 分配（或复用 out=）。out= 的 shape/dtype/device 已在骨架校验。
fn alloc_or_reuse_out(
    out: Option<&Array>,
    _out_shape: &[usize],
    nbytes: usize,
    device: &Device,
    out_stream: &Arc<Stream>,
) -> Result<(BufferRef, Option<NonNull<u8>>)> {
    match out {
        Some(o) => Ok((o.data().clone(), o.data().buffer().ptr())),
        None => {
            let buffer = Buffer::alloc(nbytes, device.clone(), out_stream)?;
            let data_ref = BufferRef::new(Arc::new(buffer));
            let ptr = data_ref.buffer().ptr();
            Ok((data_ref, ptr))
        }
    }
}

/// 输入指针调整（包含 layout offset；元素单位）。
fn adjusted_ptr(a: &Array, dtype: Dtype) -> Option<NonNull<u8>> {
    let ptr = a.data().buffer().ptr()?;
    let elem_size = dtype.element_size();
    unsafe {
        Some(NonNull::new_unchecked(
            ptr.as_ptr().add(a.layout().offset * elem_size),
        ))
    }
}
