//! OpBuilder — 算子构建器（ADR L1-12, L1-16, L2-4, L2-5）
//!
//! Capture-safe 设计：参数解析（一次性）与 kernel launch（可重放）分离，
//! 为未来 MUSA Graphs capture 保留 lazy hook。
//!
//! 约束（ADR L2-4）：执行阶段不读 host 侧可变状态。
//!
//! v0.2 Phase 1: 提取通用 binary elementwise 3-phase 骨架（P1.6），
//! add 迁移到 stride-aware _v2 ABI + broadcast（P1.7, ADR-002-D2）。
//!
//! v0.2 Phase 2: elementwise 全家桶骨架（binary/unary/clamp/cast）+ 类型提升：
//! - binary 激活 `promote`（ADR L1-5, L1-14），计算白名单 float32/float64，
//!   输入 dtype 与结果 dtype 不一致时内部 `cast_array` 转换；
//! - unary/clamp 单输入骨架（输出 shape = 输入 shape，stride-aware）；
//! - cast 骨架（`cast_array` 内部助手 + `astype_op` 公开路径）。
//!
//! 本文件只导出 `pub(crate)` 骨架；公开 API 在 `elementwise.rs`。

use crate::broadcast;
use crate::kernels;
use musapy_core::error::{DeviceError, DtypeError, MemoryError, ShapeError};
use musapy_core::musa_ffi;
use musapy_core::resolution;
use musapy_core::{
    promote, Array, Buffer, BufferRef, Device, DeviceResolution, Dtype, DtypeResolution, Layout,
    OpContext, ResolutionSource, Result, Stream,
};
use std::ptr::NonNull;
use std::sync::Arc;

// ── Kernel launch 宏（消除 (op × dtype) 组合的 unsafe 样板）─────
//
// 宏体内的 `kernels::` / `musa_ffi::` 路径在本模块 `use` 中解析；
// 所有展开点都在本模块内，文本序要求宏定义先于使用处。

/// Binary kernel launch（_v2 stride-aware + broadcast strides）。
macro_rules! launch_binary {
    ($fn_name:ident, $ap:expr, $bp:expr, $op:expr, $ndim:expr, $shape:expr, $as:expr, $bs:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(bp), Some(op)) = ($ap, $bp, $op) {
            unsafe {
                kernels::$fn_name(
                    ap.as_ptr() as _,
                    bp.as_ptr() as _,
                    op.as_ptr() as _,
                    $ndim,
                    $shape.as_ptr(),
                    $as.as_ptr(),
                    $bs.as_ptr(),
                    $stream,
                );
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Unary kernel launch（_v2 stride-aware）。
macro_rules! launch_unary {
    ($fn_name:ident, $ap:expr, $op:expr, $ndim:expr, $shape:expr, $as:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(op)) = ($ap, $op) {
            unsafe {
                kernels::$fn_name(
                    ap.as_ptr() as _,
                    op.as_ptr() as _,
                    $ndim,
                    $shape.as_ptr(),
                    $as.as_ptr(),
                    $stream,
                );
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Clamp kernel launch（_v2 stride-aware，lo/hi 已按 dtype 转换）。
macro_rules! launch_clamp {
    ($fn_name:ident, $ap:expr, $op:expr, $lo:expr, $hi:expr, $ndim:expr, $shape:expr, $as:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(op)) = ($ap, $op) {
            unsafe {
                kernels::$fn_name(
                    ap.as_ptr() as _,
                    op.as_ptr() as _,
                    $lo,
                    $hi,
                    $ndim,
                    $shape.as_ptr(),
                    $as.as_ptr(),
                    $stream,
                );
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Cast kernel launch（_v2 stride-aware）。
macro_rules! launch_cast {
    ($fn_name:ident, $ap:expr, $op:expr, $ndim:expr, $shape:expr, $as:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(op)) = ($ap, $op) {
            unsafe {
                kernels::$fn_name(
                    ap.as_ptr() as _,
                    op.as_ptr() as _,
                    $ndim,
                    $shape.as_ptr(),
                    $as.as_ptr(),
                    $stream,
                );
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Comparison kernel launch（_v2 stride-aware + broadcast strides，输出 u8/bool）。
macro_rules! launch_compare {
    ($fn:ident, $a:expr, $b:expr, $out:expr, $ndim:expr, $shape:expr,
     $a_strides:expr, $b_strides:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(bp), Some(op)) = ($a, $b, $out) {
            unsafe {
                kernels::$fn(ap.as_ptr() as _, bp.as_ptr() as _, op.as_ptr() as _,
                    $ndim, $shape.as_ptr(), $a_strides.as_ptr(), $b_strides.as_ptr(), $stream);
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// CPU cast 单 (src, dst) 类型对的 `as` 转换循环（stride-aware）。
macro_rules! cpu_cast_pair {
    ($src_t:ty, $dst_t:ty, $a:expr, $c:expr, $shape:expr, $strides:expr) => {{
        let n: usize = $shape.iter().product();
        if n > 0 {
            if let (Some(ap), Some(cp)) = ($a, $c) {
                unsafe {
                    let base_a = ap.as_ptr() as *const $src_t;
                    let base_c = cp.as_ptr() as *mut $dst_t;
                    for idx in 0..n {
                        let off = cpu_offset_nd(idx, $shape, $strides);
                        *base_c.add(idx) = *base_a.add(off) as $dst_t;
                    }
                }
            }
        }
    }};
}

// ── Kernel 分派标识 ───────────────────────────────────────────

/// 具体 binary kernel 标识（用于骨架分派）。
#[derive(Clone, Copy, Debug)]
pub(crate) enum BinaryKernel {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl BinaryKernel {
    fn name(&self) -> &'static str {
        match self {
            BinaryKernel::Add => "add",
            BinaryKernel::Sub => "sub",
            BinaryKernel::Mul => "mul",
            BinaryKernel::Div => "div",
            BinaryKernel::Pow => "pow",
        }
    }
}

/// 具体 unary kernel 标识（用于骨架分派）。
#[derive(Clone, Copy, Debug)]
pub(crate) enum UnaryKernel {
    Sin,
    Cos,
    Exp,
    Log,
    Abs,
    Sign,
    Neg,
}

impl UnaryKernel {
    fn name(&self) -> &'static str {
        match self {
            UnaryKernel::Sin => "sin",
            UnaryKernel::Cos => "cos",
            UnaryKernel::Exp => "exp",
            UnaryKernel::Log => "log",
            UnaryKernel::Abs => "abs",
            UnaryKernel::Sign => "sign",
            UnaryKernel::Neg => "neg",
        }
    }
}

// ── 通用 binary elementwise 骨架（P1.6, Phase 2 类型提升）──────

/// 通用 binary elementwise 3-phase 骨架。
///
/// Phase A（参数解析，capture-safe）：
///   1. Device 匹配 → 2. Broadcast shape → 3. 类型提升（promote）→
///   4. 计算白名单（f32/f64）→ 5. out= 校验 → 6. Stream 选择 →
///   7. 内部 cast（输入 dtype != 结果 dtype）→
///   8. Buffer 分配 + alias 检测 → 9. Stream wait
///
/// Phase B（kernel launch，可重放）：
///   CPU fallback 或 GPU kernel（_v2 stride-aware）
///
/// Phase C（后处理）：
///   事件记录 + OpContext + 构造输出 Array
///
/// 类型提升（ADR L1-5, L1-14）：
///   `promote(a.dtype, b.dtype, all_gpu)`，`all_gpu = 输入全在 MUSA 设备`。
///   GPU narrow 策略下同类别取最窄输入；结果必须落在计算白名单
///   float32/float64，否则报 DtypeError（int/complex 计算后续 Phase）。
pub(crate) fn binary_elementwise(
    a: &Array,
    b: &Array,
    out: Option<&Array>,
    kernel: BinaryKernel,
) -> Result<Array> {
    let op_name = kernel.name();

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

    // 2. Broadcast shape 计算（ADR-002-D2, NumPy 规则）
    let out_shape = broadcast::broadcast_shape(&[a.shape(), b.shape()])?;

    // 3. 类型提升（Phase 2 激活，ADR L1-5, L1-14）
    //    all_gpu：输入全在 MUSA 设备时用 GPU narrow 策略（性能优先）
    let all_gpu = matches!(device, Device::Musa(_));
    let dtype = promote(a.dtype(), b.dtype(), all_gpu)?;

    // 4. 计算白名单（仅 f32/f64，其他计算 dtype 后续 Phase 添加）
    match dtype {
        Dtype::Float32 | Dtype::Float64 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "{}: promoted dtype {} not supported (compute whitelist: float32/float64)",
                op_name, dtype
            ))
            .into());
        }
    }

    let out_size: usize = out_shape.iter().product();
    let nbytes = out_size * dtype.element_size();

    // 5. out= 参数校验（若提供）
    if let Some(o) = out {
        if o.shape() != &out_shape {
            return Err(ShapeError::Mismatch(format!(
                "{}: out shape {:?} != broadcast output shape {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != dtype {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != promoted dtype {}",
                op_name,
                o.dtype(),
                dtype
            ))
            .into());
        }
        if o.device() != a.device() {
            return Err(DeviceError::Mismatch(format!(
                "{}: out device {} != input device {}",
                op_name,
                o.device(),
                a.device()
            ))
            .into());
        }
    }

    // 6. Stream 选择（ADR L1-8）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 7. 内部 cast（输入 dtype != 提升结果 dtype 时，Phase 2）
    //    cast_array 分配新 Buffer 并在 out_stream 上执行转换 kernel；
    //    无需转换时借用原输入（零拷贝）。
    let a_cast = (a.dtype() != dtype)
        .then(|| cast_array(a, dtype, &out_stream))
        .transpose()?;
    let b_cast = (b.dtype() != dtype)
        .then(|| cast_array(b, dtype, &out_stream))
        .transpose()?;
    let a_work: &Array = a_cast.as_ref().unwrap_or(a);
    let b_work: &Array = b_cast.as_ref().unwrap_or(b);

    // 8. out= 处理 + 别名检测（ADR L2-5，对实际参与 kernel 的 work 数组检测）
    let (out_data_ref, out_ptr) = match out {
        Some(o) => {
            if o.data() == a_work.data() || o.data() == b_work.data() {
                return Err(MemoryError::AliasDetected.into());
            }
            (o.data().clone(), o.data().buffer().ptr())
        }
        None => {
            let buffer = Buffer::alloc(nbytes, device.clone(), &out_stream)?;
            let buffer_arc = Arc::new(buffer);
            let data_ref = BufferRef::new(buffer_arc);
            let ptr = data_ref.buffer().ptr();
            (data_ref, ptr)
        }
    };

    // 9. 自动 stream wait（ADR L1-8）
    a_work.data().buffer().wait_last_write_on(&out_stream)?;
    b_work.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：Kernel launch（可重放，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = a_work.data().buffer().ptr();
    let b_ptr = b_work.data().buffer().ptr();

    if out_size > 0 {
        // 计算每个输入的广播 strides（组合输入自身 strides）
        let a_strides = broadcast::broadcast_strides(a_work.layout(), &out_shape);
        let b_strides = broadcast::broadcast_strides(b_work.layout(), &out_shape);
        let ndim = out_shape.len() as i32;
        let stream_raw = out_stream.raw();

        match &device {
            Device::Cpu => {
                cpu_binary_elementwise(
                    a_ptr, b_ptr, out_ptr, &out_shape, &a_strides, &b_strides, dtype, &kernel,
                );
            }
            Device::Musa(_) => match (&kernel, dtype) {
                (BinaryKernel::Add, Dtype::Float32) => launch_binary!(musapy_add_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "add_f32_v2"),
                (BinaryKernel::Add, Dtype::Float64) => launch_binary!(musapy_add_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "add_f64_v2"),
                (BinaryKernel::Sub, Dtype::Float32) => launch_binary!(musapy_sub_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "sub_f32_v2"),
                (BinaryKernel::Sub, Dtype::Float64) => launch_binary!(musapy_sub_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "sub_f64_v2"),
                (BinaryKernel::Mul, Dtype::Float32) => launch_binary!(musapy_mul_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "mul_f32_v2"),
                (BinaryKernel::Mul, Dtype::Float64) => launch_binary!(musapy_mul_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "mul_f64_v2"),
                (BinaryKernel::Div, Dtype::Float32) => launch_binary!(musapy_div_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "div_f32_v2"),
                (BinaryKernel::Div, Dtype::Float64) => launch_binary!(musapy_div_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "div_f64_v2"),
                (BinaryKernel::Pow, Dtype::Float32) => launch_binary!(musapy_pow_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "pow_f32_v2"),
                (BinaryKernel::Pow, Dtype::Float64) => launch_binary!(musapy_pow_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "pow_f64_v2"),
                _ => unreachable!("dtype already validated as float32/float64"),
            },
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理
    // ═══════════════════════════════════════════════════════════════

    // 10. 事件记录（ADR L3-10）
    a_work.data().buffer().record_read(&out_stream);
    b_work.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    // 11. OpContext 记录（ADR L3-2，记录用户视角的原始 dtype）
    let mut ctx = OpContext::new(
        op_name,
        vec![a.shape().clone(), b.shape().clone()],
        vec![a.device().clone(), b.device().clone()],
        vec![a.dtype(), b.dtype()],
        out_shape.clone(),
        out_stream.id(),
    );
    if musapy_core::debug::is_debug() {
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
    }
    out_stream.record_op(ctx);

    // 12. 构造输出 Array（连续布局，shape = broadcast output，dtype = 提升结果）
    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

// ── 通用 unary elementwise 骨架（Phase 2）─────────────────────

/// 通用 unary elementwise 3-phase 骨架。
///
/// 单输入，无广播：输出 shape = 输入 shape，kernel 直接使用输入布局的
/// strides（stride-aware，支持非连续视图）。
///
/// Phase A（参数解析，capture-safe）：
///   1. Dtype 白名单（f32/f64）→ 2. out= 校验 → 3. Stream 选择 →
///   4. Buffer 分配 + alias 检测 → 5. Stream wait
///
/// Phase B（kernel launch，可重放）：
///   CPU fallback 或 GPU kernel（_v2 stride-aware）
///
/// Phase C（后处理）：
///   事件记录 + OpContext + 构造输出 Array
pub(crate) fn unary_elementwise(
    a: &Array,
    out: Option<&Array>,
    kernel: UnaryKernel,
) -> Result<Array> {
    let op_name = kernel.name();

    // ═══════════════════════════════════════════════════════════════
    // Phase A：参数解析（一次性，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    // 1. Dtype 白名单（仅 f32/f64，其他 dtype 后续 Phase 添加）
    let dtype = a.dtype();
    match dtype {
        Dtype::Float32 | Dtype::Float64 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "{}: dtype {} not supported (only float32/float64)",
                op_name, dtype
            ))
            .into());
        }
    }

    let device = a.device().clone();
    let out_shape = a.shape().clone();
    let out_size: usize = out_shape.iter().product();
    let nbytes = out_size * dtype.element_size();

    // 2. out= 参数校验（若提供）
    if let Some(o) = out {
        if o.shape() != &out_shape {
            return Err(ShapeError::Mismatch(format!(
                "{}: out shape {:?} != input shape {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != dtype {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != input dtype {}",
                op_name,
                o.dtype(),
                dtype
            ))
            .into());
        }
        if o.device() != a.device() {
            return Err(DeviceError::Mismatch(format!(
                "{}: out device {} != input device {}",
                op_name,
                o.device(),
                a.device()
            ))
            .into());
        }
    }

    // 3. Stream 选择（ADR L1-8）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 4. out= 处理 + 别名检测（ADR L2-5）
    let (out_data_ref, out_ptr) = match out {
        Some(o) => {
            if o.data() == a.data() {
                return Err(MemoryError::AliasDetected.into());
            }
            (o.data().clone(), o.data().buffer().ptr())
        }
        None => {
            let buffer = Buffer::alloc(nbytes, device.clone(), &out_stream)?;
            let buffer_arc = Arc::new(buffer);
            let data_ref = BufferRef::new(buffer_arc);
            let ptr = data_ref.buffer().ptr();
            (data_ref, ptr)
        }
    };

    // 5. 自动 stream wait（ADR L1-8）
    a.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：Kernel launch（可重放，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = a.data().buffer().ptr();

    if out_size > 0 {
        // 直接使用输入布局的 strides（无广播，stride-aware）
        let a_strides: Vec<isize> = a.layout().strides.iter().map(|&s| s as isize).collect();
        let ndim = out_shape.len() as i32;
        let stream_raw = out_stream.raw();

        match &device {
            Device::Cpu => {
                cpu_unary_elementwise(a_ptr, out_ptr, &out_shape, &a_strides, dtype, &kernel);
            }
            Device::Musa(_) => match (&kernel, dtype) {
                (UnaryKernel::Sin, Dtype::Float32) => launch_unary!(musapy_sin_f32_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "sin_f32_v2"),
                (UnaryKernel::Sin, Dtype::Float64) => launch_unary!(musapy_sin_f64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "sin_f64_v2"),
                (UnaryKernel::Cos, Dtype::Float32) => launch_unary!(musapy_cos_f32_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "cos_f32_v2"),
                (UnaryKernel::Cos, Dtype::Float64) => launch_unary!(musapy_cos_f64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "cos_f64_v2"),
                (UnaryKernel::Exp, Dtype::Float32) => launch_unary!(musapy_exp_f32_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "exp_f32_v2"),
                (UnaryKernel::Exp, Dtype::Float64) => launch_unary!(musapy_exp_f64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "exp_f64_v2"),
                (UnaryKernel::Log, Dtype::Float32) => launch_unary!(musapy_log_f32_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "log_f32_v2"),
                (UnaryKernel::Log, Dtype::Float64) => launch_unary!(musapy_log_f64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "log_f64_v2"),
                (UnaryKernel::Abs, Dtype::Float32) => launch_unary!(musapy_abs_f32_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "abs_f32_v2"),
                (UnaryKernel::Abs, Dtype::Float64) => launch_unary!(musapy_abs_f64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "abs_f64_v2"),
                (UnaryKernel::Sign, Dtype::Float32) => launch_unary!(musapy_sign_f32_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "sign_f32_v2"),
                (UnaryKernel::Sign, Dtype::Float64) => launch_unary!(musapy_sign_f64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "sign_f64_v2"),
                (UnaryKernel::Neg, Dtype::Float32) => launch_unary!(musapy_neg_f32_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "neg_f32_v2"),
                (UnaryKernel::Neg, Dtype::Float64) => launch_unary!(musapy_neg_f64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "neg_f64_v2"),
                _ => unreachable!("dtype already validated as float32/float64"),
            },
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理
    // ═══════════════════════════════════════════════════════════════

    // 6. 事件记录（ADR L3-10）
    a.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    // 7. OpContext 记录（ADR L3-2）
    let mut ctx = OpContext::new(
        op_name,
        vec![a.shape().clone()],
        vec![a.device().clone()],
        vec![a.dtype()],
        out_shape.clone(),
        out_stream.id(),
    );
    if musapy_core::debug::is_debug() {
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
    }
    out_stream.record_op(ctx);

    // 8. 构造输出 Array（连续布局，shape = 输入 shape）
    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

// ── 通用 clamp elementwise 骨架（Phase 2）─────────────────────

/// 通用 clamp elementwise 3-phase 骨架。
///
/// 与 unary 骨架同构，额外携带 lo/hi 标量参数（f64 传入，按 dtype 转换）。
/// 语义：`out[i] = min(max(a[i], lo), hi)`。
pub(crate) fn clamp_elementwise(
    a: &Array,
    lo: f64,
    hi: f64,
    out: Option<&Array>,
) -> Result<Array> {
    let op_name = "clamp";

    // ═══════════════════════════════════════════════════════════════
    // Phase A：参数解析（一次性，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    // 1. Dtype 白名单（仅 f32/f64）
    let dtype = a.dtype();
    match dtype {
        Dtype::Float32 | Dtype::Float64 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "{}: dtype {} not supported (only float32/float64)",
                op_name, dtype
            ))
            .into());
        }
    }

    let device = a.device().clone();
    let out_shape = a.shape().clone();
    let out_size: usize = out_shape.iter().product();
    let nbytes = out_size * dtype.element_size();

    // 2. out= 参数校验（若提供）
    if let Some(o) = out {
        if o.shape() != &out_shape {
            return Err(ShapeError::Mismatch(format!(
                "{}: out shape {:?} != input shape {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != dtype {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != input dtype {}",
                op_name,
                o.dtype(),
                dtype
            ))
            .into());
        }
        if o.device() != a.device() {
            return Err(DeviceError::Mismatch(format!(
                "{}: out device {} != input device {}",
                op_name,
                o.device(),
                a.device()
            ))
            .into());
        }
    }

    // 3. Stream 选择（ADR L1-8）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 4. out= 处理 + 别名检测（ADR L2-5）
    let (out_data_ref, out_ptr) = match out {
        Some(o) => {
            if o.data() == a.data() {
                return Err(MemoryError::AliasDetected.into());
            }
            (o.data().clone(), o.data().buffer().ptr())
        }
        None => {
            let buffer = Buffer::alloc(nbytes, device.clone(), &out_stream)?;
            let buffer_arc = Arc::new(buffer);
            let data_ref = BufferRef::new(buffer_arc);
            let ptr = data_ref.buffer().ptr();
            (data_ref, ptr)
        }
    };

    // 5. 自动 stream wait（ADR L1-8）
    a.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：Kernel launch（可重放，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = a.data().buffer().ptr();

    if out_size > 0 {
        let a_strides: Vec<isize> = a.layout().strides.iter().map(|&s| s as isize).collect();
        let ndim = out_shape.len() as i32;
        let stream_raw = out_stream.raw();

        match &device {
            Device::Cpu => {
                cpu_clamp_elementwise(
                    a_ptr, out_ptr, &out_shape, &a_strides, dtype, lo, hi,
                );
            }
            // lo/hi 按 dtype 转换（f64 → f32 时截断，与 kernel 签名一致）
            Device::Musa(_) => match dtype {
                Dtype::Float32 => launch_clamp!(musapy_clamp_f32_v2, a_ptr, out_ptr, lo as f32, hi as f32, ndim, out_shape, a_strides, stream_raw, "clamp_f32_v2"),
                Dtype::Float64 => launch_clamp!(musapy_clamp_f64_v2, a_ptr, out_ptr, lo, hi, ndim, out_shape, a_strides, stream_raw, "clamp_f64_v2"),
                _ => unreachable!("dtype already validated as float32/float64"),
            },
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理
    // ═══════════════════════════════════════════════════════════════

    // 6. 事件记录（ADR L3-10）
    a.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    // 7. OpContext 记录（ADR L3-2）
    let mut ctx = OpContext::new(
        op_name,
        vec![a.shape().clone()],
        vec![a.device().clone()],
        vec![a.dtype()],
        out_shape.clone(),
        out_stream.id(),
    );
    if musapy_core::debug::is_debug() {
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
    }
    out_stream.record_op(ctx);

    // 8. 构造输出 Array（连续布局，shape = 输入 shape）
    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

// ── Cast 骨架（Phase 2）───────────────────────────────────────

/// 校验 cast 类型对是否受 Phase 2 kernel 支持。
///
/// 支持矩阵（见 kernels.rs `musapy_cast_<src>_<dst>_v2`）：
/// - dst ∈ {float32, float64}（计算白名单）
/// - src ∈ {int8..int64, uint8..uint64, float32, float64}，且 src != dst
/// - bool/float16/bfloat16/complex 尚无 cast kernel（后续 Phase）
fn validate_cast_pair(src: Dtype, dst: Dtype) -> Result<()> {
    match dst {
        Dtype::Float32 | Dtype::Float64 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "cast: target dtype {} not supported (only float32/float64)",
                dst
            ))
            .into());
        }
    }
    match src {
        Dtype::Int8
        | Dtype::Int16
        | Dtype::Int32
        | Dtype::Int64
        | Dtype::Uint8
        | Dtype::Uint16
        | Dtype::Uint32
        | Dtype::Uint64
        | Dtype::Float32
        | Dtype::Float64 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "cast: source dtype {} not supported (bool/float16/bfloat16/complex not yet implemented)",
                src
            ))
            .into());
        }
    }
    if src == dst {
        return Err(DtypeError::Unsupported(format!(
            "cast: source and target dtype are identical ({})",
            src
        ))
        .into());
    }
    Ok(())
}

/// Cast kernel 分派（CPU fallback 或 GPU _v2 stride-aware）。
///
/// 前置条件：调用者已通过 `validate_cast_pair`。
/// 输出为连续布局，输入 strides 由调用者提供（元素单位）。
fn launch_cast_kernel(
    a_ptr: Option<NonNull<u8>>,
    c_ptr: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    src: Dtype,
    dst: Dtype,
    device: &Device,
    stream: &Arc<Stream>,
) -> Result<()> {
    let n: usize = shape.iter().product();
    if n == 0 {
        return Ok(());
    }

    match device {
        Device::Cpu => {
            cpu_cast(a_ptr, c_ptr, shape, a_strides, src, dst);
            Ok(())
        }
        Device::Musa(_) => {
            let ndim = shape.len() as i32;
            let stream_raw = stream.raw();
            match dst {
                Dtype::Float32 => match src {
                    Dtype::Int8 => launch_cast!(musapy_cast_i8_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i8_f32_v2"),
                    Dtype::Int16 => launch_cast!(musapy_cast_i16_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i16_f32_v2"),
                    Dtype::Int32 => launch_cast!(musapy_cast_i32_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i32_f32_v2"),
                    Dtype::Int64 => launch_cast!(musapy_cast_i64_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i64_f32_v2"),
                    Dtype::Uint8 => launch_cast!(musapy_cast_u8_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u8_f32_v2"),
                    Dtype::Uint16 => launch_cast!(musapy_cast_u16_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u16_f32_v2"),
                    Dtype::Uint32 => launch_cast!(musapy_cast_u32_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u32_f32_v2"),
                    Dtype::Uint64 => launch_cast!(musapy_cast_u64_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u64_f32_v2"),
                    Dtype::Float64 => launch_cast!(musapy_cast_f64_f32_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_f64_f32_v2"),
                    _ => unreachable!("cast pair already validated"),
                },
                Dtype::Float64 => match src {
                    Dtype::Int8 => launch_cast!(musapy_cast_i8_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i8_f64_v2"),
                    Dtype::Int16 => launch_cast!(musapy_cast_i16_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i16_f64_v2"),
                    Dtype::Int32 => launch_cast!(musapy_cast_i32_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i32_f64_v2"),
                    Dtype::Int64 => launch_cast!(musapy_cast_i64_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i64_f64_v2"),
                    Dtype::Uint8 => launch_cast!(musapy_cast_u8_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u8_f64_v2"),
                    Dtype::Uint16 => launch_cast!(musapy_cast_u16_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u16_f64_v2"),
                    Dtype::Uint32 => launch_cast!(musapy_cast_u32_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u32_f64_v2"),
                    Dtype::Uint64 => launch_cast!(musapy_cast_u64_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u64_f64_v2"),
                    Dtype::Float32 => launch_cast!(musapy_cast_f32_f64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_f32_f64_v2"),
                    _ => unreachable!("cast pair already validated"),
                },
                _ => unreachable!("cast target already validated as float32/float64"),
            }
            Ok(())
        }
    }
}

/// 内部 cast 助手：分配新 Buffer，执行 dtype 转换，返回新 Array。
///
/// 用于 binary 骨架的类型提升路径（输入 dtype != 提升结果 dtype）：
/// - GPU：launch `musapy_cast_<src>_<dst>_v2` kernel（stride-aware）
/// - CPU：`as` 转换循环
///
/// 事件语义：在 `stream` 上等待输入的最后写入，转换后记录输入读 + 输出写。
/// 调用者保证 `a.dtype() != target_dtype`（相同 dtype 无需转换）。
pub(crate) fn cast_array(a: &Array, target_dtype: Dtype, stream: &Arc<Stream>) -> Result<Array> {
    let src = a.dtype();
    validate_cast_pair(src, target_dtype)?;

    let device = a.device().clone();
    let shape = a.shape().clone();
    let out_size: usize = shape.iter().product();
    let nbytes = out_size * target_dtype.element_size();

    // 分配输出 Buffer
    let buffer = Buffer::alloc(nbytes, device.clone(), stream)?;
    let buffer_arc = Arc::new(buffer);
    let out_data_ref = BufferRef::new(buffer_arc);
    let out_ptr = out_data_ref.buffer().ptr();

    // 自动 stream wait（ADR L1-8）
    a.data().buffer().wait_last_write_on(stream)?;

    // Kernel launch（stride-aware，输入布局 strides）
    let a_strides: Vec<isize> = a.layout().strides.iter().map(|&s| s as isize).collect();
    launch_cast_kernel(
        a.data().buffer().ptr(),
        out_ptr,
        &shape,
        &a_strides,
        src,
        target_dtype,
        &device,
        stream,
    )?;

    // 事件记录（ADR L3-10）
    a.data().buffer().record_read(stream);
    out_data_ref.buffer().record_write(stream);

    // OpContext 记录（ADR L3-2）
    let mut ctx = OpContext::new(
        "cast",
        vec![shape.clone()],
        vec![device.clone()],
        vec![src],
        shape.clone(),
        stream.id(),
    );
    if musapy_core::debug::is_debug() {
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
    }
    stream.record_op(ctx);

    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(shape),
        target_dtype,
        Arc::clone(stream),
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(target_dtype, ResolutionSource::InputArray),
    ))
}

/// `astype` 骨架：显式 dtype 转换（公开路径，支持 out=）。
///
/// Phase A：cast 类型对校验 → out= 校验 → Stream 选择 →
///          Buffer 分配 + alias 检测 → Stream wait
/// Phase B：
///   - src != dst：cast kernel（`launch_cast_kernel`）
///   - src == dst：深拷贝（要求连续布局；GPU 用 musaMemcpy D2D，
///     CPU 用 copy_nonoverlapping。非连续同 dtype 拷贝后续 Phase 支持）
/// Phase C：事件记录 + OpContext + 构造输出 Array
pub(crate) fn astype_op(a: &Array, dtype: Dtype, out: Option<&Array>) -> Result<Array> {
    let op_name = "astype";
    let src = a.dtype();

    // ═══════════════════════════════════════════════════════════════
    // Phase A：参数解析（一次性，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    // 1. Cast 类型对校验（同 dtype 走 Phase B 深拷贝分支）
    if src != dtype {
        validate_cast_pair(src, dtype)?;
    }

    let device = a.device().clone();
    let out_shape = a.shape().clone();
    let out_size: usize = out_shape.iter().product();
    let nbytes = out_size * dtype.element_size();

    // 2. out= 参数校验（若提供）
    if let Some(o) = out {
        if o.shape() != &out_shape {
            return Err(ShapeError::Mismatch(format!(
                "{}: out shape {:?} != input shape {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != dtype {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != target dtype {}",
                op_name,
                o.dtype(),
                dtype
            ))
            .into());
        }
        if o.device() != a.device() {
            return Err(DeviceError::Mismatch(format!(
                "{}: out device {} != input device {}",
                op_name,
                o.device(),
                a.device()
            ))
            .into());
        }
    }

    // 3. Stream 选择（ADR L1-8）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 4. out= 处理 + 别名检测（ADR L2-5）
    let (out_data_ref, out_ptr) = match out {
        Some(o) => {
            if o.data() == a.data() {
                return Err(MemoryError::AliasDetected.into());
            }
            (o.data().clone(), o.data().buffer().ptr())
        }
        None => {
            let buffer = Buffer::alloc(nbytes, device.clone(), &out_stream)?;
            let buffer_arc = Arc::new(buffer);
            let data_ref = BufferRef::new(buffer_arc);
            let ptr = data_ref.buffer().ptr();
            (data_ref, ptr)
        }
    };

    // 5. 自动 stream wait（ADR L1-8）
    a.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：Kernel launch（可重放，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = a.data().buffer().ptr();

    if out_size > 0 {
        if src == dtype {
            // 同 dtype 深拷贝（连续布局，按字节拷贝）
            if !a.is_contiguous() {
                return Err(DtypeError::Unsupported(
                    "astype: same-dtype copy requires contiguous input (Phase 2 limitation)"
                        .to_string(),
                )
                .into());
            }
            match &device {
                Device::Cpu => {
                    if let (Some(ap), Some(op)) = (a_ptr, out_ptr) {
                        unsafe {
                            std::ptr::copy_nonoverlapping(ap.as_ptr(), op.as_ptr(), nbytes);
                        }
                    }
                }
                Device::Musa(_) => {
                    if let (Some(ap), Some(op)) = (a_ptr, out_ptr) {
                        unsafe {
                            musa_ffi::check_musa(
                                musa_ffi::musaMemcpy(
                                    op.as_ptr() as *mut std::ffi::c_void,
                                    ap.as_ptr() as *const std::ffi::c_void,
                                    nbytes,
                                    musa_ffi::musaMemcpyKind::DeviceToDevice,
                                ),
                                "musaMemcpy(D2D astype same-dtype)",
                            )?;
                        }
                    }
                }
            }
        } else {
            // 异 dtype：cast kernel（stride-aware）
            let a_strides: Vec<isize> = a.layout().strides.iter().map(|&s| s as isize).collect();
            launch_cast_kernel(
                a_ptr, out_ptr, &out_shape, &a_strides, src, dtype, &device, &out_stream,
            )?;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理
    // ═══════════════════════════════════════════════════════════════

    // 6. 事件记录（ADR L3-10）
    a.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    // 7. OpContext 记录（ADR L3-2）
    let mut ctx = OpContext::new(
        op_name,
        vec![a.shape().clone()],
        vec![a.device().clone()],
        vec![src],
        out_shape.clone(),
        out_stream.id(),
    );
    if musapy_core::debug::is_debug() {
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
    }
    out_stream.record_op(ctx);

    // 8. 构造输出 Array（连续布局，shape = 输入 shape，dtype = 目标 dtype）
    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::Arg),
    ))
}

// ── 通用 comparison elementwise 骨架（Phase 3）──────────────

/// 具体 comparison kernel 标识（用于骨架分派）。
#[derive(Clone, Copy, Debug)]
pub(crate) enum CompareKernel {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

impl CompareKernel {
    fn name(&self) -> &'static str {
        match self {
            CompareKernel::Eq => "eq",
            CompareKernel::Ne => "ne",
            CompareKernel::Lt => "lt",
            CompareKernel::Gt => "gt",
            CompareKernel::Le => "le",
            CompareKernel::Ge => "ge",
        }
    }
}

/// 通用 comparison elementwise 3-phase 骨架（Phase 3, ADR-002 Phase 3）。
///
/// 与 `binary_elementwise` 同构，关键差异：
/// - 输出 dtype 恒为 `Dtype::Bool`（提升后的输入 dtype 仅用于 kernel 分派）
/// - `nbytes = out_size * 1`（bool element_size = 1）
/// - `out=` 校验要求 `o.dtype() == Dtype::Bool`
///
/// Phase A（参数解析，capture-safe）：
///   1. Device 匹配 → 2. Broadcast shape → 3. 类型提升（promote）→
///   4. 计算白名单（f32/f64）→ 5. out= 校验（shape + dtype=Bool + device）→
///   6. Stream 选择 → 7. 内部 cast（输入 dtype != 提升 dtype）→
///   8. Buffer 分配 + alias 检测 → 9. Stream wait
///
/// Phase B（kernel launch，可重放）：
///   CPU fallback 或 GPU comparison kernel（_v2 stride-aware，输出 u8/bool）
///
/// Phase C（后处理）：
///   事件记录 + OpContext + 构造输出 Array（dtype = Bool）
pub(crate) fn comparison_elementwise(
    a: &Array,
    b: &Array,
    out: Option<&Array>,
    kernel: CompareKernel,
) -> Result<Array> {
    let op_name = kernel.name();

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

    // 2. Broadcast shape 计算（ADR-002-D2, NumPy 规则）
    let out_shape = broadcast::broadcast_shape(&[a.shape(), b.shape()])?;

    // 3. 类型提升（仅用于 kernel 分派；输出恒为 bool）
    //    all_gpu：输入全在 MUSA 设备时用 GPU narrow 策略（性能优先）
    let all_gpu = matches!(device, Device::Musa(_));
    let dtype = promote(a.dtype(), b.dtype(), all_gpu)?;

    // 4. 计算白名单（仅 f32/f64，其他计算 dtype 后续 Phase 添加）
    match dtype {
        Dtype::Float32 | Dtype::Float64 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "{}: promoted dtype {} not supported (compute whitelist: float32/float64)",
                op_name, dtype
            ))
            .into());
        }
    }

    let out_size: usize = out_shape.iter().product();
    // 输出 dtype 恒为 bool（element_size = 1）
    let nbytes = out_size * Dtype::Bool.element_size();

    // 5. out= 参数校验（若提供）：shape + dtype=Bool + device
    if let Some(o) = out {
        if o.shape() != &out_shape {
            return Err(ShapeError::Mismatch(format!(
                "{}: out shape {:?} != broadcast output shape {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != Dtype::Bool {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != bool (comparison output dtype)",
                op_name,
                o.dtype()
            ))
            .into());
        }
        if o.device() != a.device() {
            return Err(DeviceError::Mismatch(format!(
                "{}: out device {} != input device {}",
                op_name,
                o.device(),
                a.device()
            ))
            .into());
        }
    }

    // 6. Stream 选择（ADR L1-8）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 7. 内部 cast（输入 dtype != 提升结果 dtype 时）
    //    cast_array 分配新 Buffer 并在 out_stream 上执行转换 kernel；
    //    无需转换时借用原输入（零拷贝）。
    let a_cast = (a.dtype() != dtype)
        .then(|| cast_array(a, dtype, &out_stream))
        .transpose()?;
    let b_cast = (b.dtype() != dtype)
        .then(|| cast_array(b, dtype, &out_stream))
        .transpose()?;
    let a_work: &Array = a_cast.as_ref().unwrap_or(a);
    let b_work: &Array = b_cast.as_ref().unwrap_or(b);

    // 8. out= 处理 + 别名检测（ADR L2-5，对实际参与 kernel 的 work 数组检测）
    let (out_data_ref, out_ptr) = match out {
        Some(o) => {
            if o.data() == a_work.data() || o.data() == b_work.data() {
                return Err(MemoryError::AliasDetected.into());
            }
            (o.data().clone(), o.data().buffer().ptr())
        }
        None => {
            let buffer = Buffer::alloc(nbytes, device.clone(), &out_stream)?;
            let buffer_arc = Arc::new(buffer);
            let data_ref = BufferRef::new(buffer_arc);
            let ptr = data_ref.buffer().ptr();
            (data_ref, ptr)
        }
    };

    // 9. 自动 stream wait（ADR L1-8）
    a_work.data().buffer().wait_last_write_on(&out_stream)?;
    b_work.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：Kernel launch（可重放，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = a_work.data().buffer().ptr();
    let b_ptr = b_work.data().buffer().ptr();

    if out_size > 0 {
        // 计算每个输入的广播 strides（组合输入自身 strides）
        let a_strides = broadcast::broadcast_strides(a_work.layout(), &out_shape);
        let b_strides = broadcast::broadcast_strides(b_work.layout(), &out_shape);
        let ndim = out_shape.len() as i32;
        let stream_raw = out_stream.raw();

        match &device {
            Device::Cpu => {
                cpu_comparison_elementwise(
                    a_ptr, b_ptr, out_ptr, &out_shape, &a_strides, &b_strides, dtype, &kernel,
                );
            }
            Device::Musa(_) => match (&kernel, dtype) {
                (CompareKernel::Eq, Dtype::Float32) => launch_compare!(musapy_eq_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "eq_f32_v2"),
                (CompareKernel::Eq, Dtype::Float64) => launch_compare!(musapy_eq_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "eq_f64_v2"),
                (CompareKernel::Ne, Dtype::Float32) => launch_compare!(musapy_ne_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "ne_f32_v2"),
                (CompareKernel::Ne, Dtype::Float64) => launch_compare!(musapy_ne_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "ne_f64_v2"),
                (CompareKernel::Lt, Dtype::Float32) => launch_compare!(musapy_lt_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "lt_f32_v2"),
                (CompareKernel::Lt, Dtype::Float64) => launch_compare!(musapy_lt_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "lt_f64_v2"),
                (CompareKernel::Gt, Dtype::Float32) => launch_compare!(musapy_gt_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "gt_f32_v2"),
                (CompareKernel::Gt, Dtype::Float64) => launch_compare!(musapy_gt_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "gt_f64_v2"),
                (CompareKernel::Le, Dtype::Float32) => launch_compare!(musapy_le_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "le_f32_v2"),
                (CompareKernel::Le, Dtype::Float64) => launch_compare!(musapy_le_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "le_f64_v2"),
                (CompareKernel::Ge, Dtype::Float32) => launch_compare!(musapy_ge_f32_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "ge_f32_v2"),
                (CompareKernel::Ge, Dtype::Float64) => launch_compare!(musapy_ge_f64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "ge_f64_v2"),
                _ => unreachable!("dtype already validated as float32/float64"),
            },
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理
    // ═══════════════════════════════════════════════════════════════

    // 10. 事件记录（ADR L3-10）
    a_work.data().buffer().record_read(&out_stream);
    b_work.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    // 11. OpContext 记录（ADR L3-2，记录用户视角的原始 dtype）
    let mut ctx = OpContext::new(
        op_name,
        vec![a.shape().clone(), b.shape().clone()],
        vec![a.device().clone(), b.device().clone()],
        vec![a.dtype(), b.dtype()],
        out_shape.clone(),
        out_stream.id(),
    );
    if musapy_core::debug::is_debug() {
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
    }
    out_stream.record_op(ctx);

    // 12. 构造输出 Array（连续布局，shape = broadcast output，dtype = Bool）
    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        Dtype::Bool,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(Dtype::Bool, ResolutionSource::InputArray),
    ))
}

// ── CPU fallback（stride-aware）──────────────────────────────

/// CPU 端 N 维偏移计算（与 common.h offset_nd 逻辑一致）。
fn cpu_offset_nd(linear_idx: usize, shape: &[usize], strides: &[isize]) -> usize {
    let mut offset = 0usize;
    let mut idx = linear_idx;
    for i in (0..shape.len()).rev() {
        let coord = idx % shape[i];
        idx /= shape[i];
        offset = (offset as isize + coord as isize * strides[i]) as usize;
    }
    offset
}

// ── CPU binary ──

/// 泛型浮点算术能力（f32/f64 共用；powf 是固有方法，需 trait 桥接）。
trait BinaryFloat:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn powf(self, exp: Self) -> Self;
}

impl BinaryFloat for f32 {
    fn powf(self, exp: f32) -> f32 {
        f32::powf(self, exp)
    }
}

impl BinaryFloat for f64 {
    fn powf(self, exp: f64) -> f64 {
        f64::powf(self, exp)
    }
}

/// 按 kernel 选择二元运算（CPU fallback 用）。
fn cpu_binary_op<T: BinaryFloat>(a_val: T, b_val: T, kernel: &BinaryKernel) -> T {
    match kernel {
        BinaryKernel::Add => a_val + b_val,
        BinaryKernel::Sub => a_val - b_val,
        BinaryKernel::Mul => a_val * b_val,
        BinaryKernel::Div => a_val / b_val,
        BinaryKernel::Pow => a_val.powf(b_val),
    }
}

/// 按 dtype 分派 CPU binary elementwise（stride-aware）。
fn cpu_binary_elementwise(
    a: Option<NonNull<u8>>,
    b: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    dtype: Dtype,
    kernel: &BinaryKernel,
) {
    match dtype {
        Dtype::Float32 => cpu_binary_typed::<f32>(a, b, c, shape, a_strides, b_strides, kernel),
        Dtype::Float64 => cpu_binary_typed::<f64>(a, b, c, shape, a_strides, b_strides, kernel),
        _ => unreachable!("dtype already validated as float32/float64"),
    }
}

/// 泛型 CPU binary elementwise（stride-aware，全 BinaryKernel 变体）。
fn cpu_binary_typed<T: BinaryFloat>(
    a: Option<NonNull<u8>>,
    b: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    kernel: &BinaryKernel,
) {
    let n: usize = shape.iter().product();
    if n == 0 {
        return;
    }
    let (Some(ap), Some(bp), Some(cp)) = (a, b, c) else {
        return;
    };
    unsafe {
        let base_a = ap.as_ptr() as *const T;
        let base_b = bp.as_ptr() as *const T;
        let base_c = cp.as_ptr() as *mut T;
        for idx in 0..n {
            let a_off = cpu_offset_nd(idx, shape, a_strides);
            let b_off = cpu_offset_nd(idx, shape, b_strides);
            *base_c.add(idx) = cpu_binary_op(*base_a.add(a_off), *base_b.add(b_off), kernel);
        }
    }
}

// ── CPU comparison ──

/// 按 dtype 分派 CPU comparison elementwise（stride-aware，输出 u8/bool）。
fn cpu_comparison_elementwise(
    a: Option<NonNull<u8>>,
    b: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    dtype: Dtype,
    kernel: &CompareKernel,
) {
    match dtype {
        Dtype::Float32 => cpu_compare_typed::<f32>(a, b, c, shape, a_strides, b_strides, kernel),
        Dtype::Float64 => cpu_compare_typed::<f64>(a, b, c, shape, a_strides, b_strides, kernel),
        _ => unreachable!("dtype already validated as float32/float64"),
    }
}

/// 泛型 CPU comparison elementwise（stride-aware，输出 0/1 字节）。
///
/// `av == bv` 等比较对 f32/f64 使用 PartialEq/PartialOrd（NaN != NaN 语义正确）。
fn cpu_compare_typed<T: Copy + PartialOrd>(
    a: Option<NonNull<u8>>,
    b: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    kernel: &CompareKernel,
) {
    let n: usize = shape.iter().product();
    if n == 0 {
        return;
    }
    let (Some(ap), Some(bp), Some(cp)) = (a, b, c) else {
        return;
    };
    unsafe {
        let base_a = ap.as_ptr() as *const T;
        let base_b = bp.as_ptr() as *const T;
        let base_c = cp.as_ptr() as *mut u8;
        for idx in 0..n {
            let a_off = cpu_offset_nd(idx, shape, a_strides);
            let b_off = cpu_offset_nd(idx, shape, b_strides);
            let av = *base_a.add(a_off);
            let bv = *base_b.add(b_off);
            let result = match kernel {
                CompareKernel::Eq => av == bv,
                CompareKernel::Ne => av != bv,
                CompareKernel::Lt => av < bv,
                CompareKernel::Gt => av > bv,
                CompareKernel::Le => av <= bv,
                CompareKernel::Ge => av >= bv,
            };
            *base_c.add(idx) = if result { 1 } else { 0 };
        }
    }
}

// ── CPU unary ──

/// 按 dtype 分派 CPU unary elementwise（stride-aware）。
fn cpu_unary_elementwise(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    dtype: Dtype,
    kernel: &UnaryKernel,
) {
    /// 按 kernel 选择单目函数（sin/cos/exp/ln/abs 是固有方法，sign 内联）。
    macro_rules! dispatch_unary_cpu {
        ($t:ty, $a:expr, $c:expr, $shape:expr, $strides:expr, $kernel:expr) => {
            match $kernel {
                UnaryKernel::Sin => cpu_unary_typed::<$t>($a, $c, $shape, $strides, <$t>::sin),
                UnaryKernel::Cos => cpu_unary_typed::<$t>($a, $c, $shape, $strides, <$t>::cos),
                UnaryKernel::Exp => cpu_unary_typed::<$t>($a, $c, $shape, $strides, <$t>::exp),
                UnaryKernel::Log => cpu_unary_typed::<$t>($a, $c, $shape, $strides, <$t>::ln),
                UnaryKernel::Abs => cpu_unary_typed::<$t>($a, $c, $shape, $strides, <$t>::abs),
                UnaryKernel::Sign => cpu_unary_typed::<$t>($a, $c, $shape, $strides, |v: $t| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }),
                UnaryKernel::Neg => cpu_unary_typed::<$t>($a, $c, $shape, $strides, |v: $t| -v),
            }
        };
    }

    match dtype {
        Dtype::Float32 => dispatch_unary_cpu!(f32, a, c, shape, a_strides, kernel),
        Dtype::Float64 => dispatch_unary_cpu!(f64, a, c, shape, a_strides, kernel),
        _ => unreachable!("dtype already validated as float32/float64"),
    }
}

/// 泛型 CPU unary elementwise（stride-aware，闭包分派具体运算）。
fn cpu_unary_typed<T: Copy>(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    f: impl Fn(T) -> T,
) {
    let n: usize = shape.iter().product();
    if n == 0 {
        return;
    }
    let (Some(ap), Some(cp)) = (a, c) else {
        return;
    };
    unsafe {
        let base_a = ap.as_ptr() as *const T;
        let base_c = cp.as_ptr() as *mut T;
        for idx in 0..n {
            let a_off = cpu_offset_nd(idx, shape, a_strides);
            *base_c.add(idx) = f(*base_a.add(a_off));
        }
    }
}

// ── CPU clamp ──

/// 按 dtype 分派 CPU clamp elementwise（stride-aware，lo/hi 按 dtype 转换）。
fn cpu_clamp_elementwise(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    dtype: Dtype,
    lo: f64,
    hi: f64,
) {
    match dtype {
        Dtype::Float32 => cpu_clamp_typed::<f32>(a, c, shape, a_strides, lo as f32, hi as f32),
        Dtype::Float64 => cpu_clamp_typed::<f64>(a, c, shape, a_strides, lo, hi),
        _ => unreachable!("dtype already validated as float32/float64"),
    }
}

/// 泛型 CPU clamp elementwise（stride-aware）。
///
/// 语义与 kernel 一致：`min(max(v, lo), hi)`（比较实现，NaN 透传）。
fn cpu_clamp_typed<T: Copy + PartialOrd>(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    lo: T,
    hi: T,
) {
    let n: usize = shape.iter().product();
    if n == 0 {
        return;
    }
    let (Some(ap), Some(cp)) = (a, c) else {
        return;
    };
    unsafe {
        let base_a = ap.as_ptr() as *const T;
        let base_c = cp.as_ptr() as *mut T;
        for idx in 0..n {
            let a_off = cpu_offset_nd(idx, shape, a_strides);
            let v = *base_a.add(a_off);
            *base_c.add(idx) = if v < lo {
                lo
            } else if v > hi {
                hi
            } else {
                v
            };
        }
    }
}

// ── CPU cast ──

/// 按 (src, dst) 分派 CPU cast（stride-aware `as` 转换循环）。
fn cpu_cast(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
    src: Dtype,
    dst: Dtype,
) {
    /// 固定 src 类型后按 dst 分派。
    macro_rules! dispatch_cast_dst {
        ($src_t:ty, $a:expr, $c:expr, $shape:expr, $strides:expr, $dst:expr) => {
            match $dst {
                Dtype::Float32 => cpu_cast_pair!($src_t, f32, $a, $c, $shape, $strides),
                Dtype::Float64 => cpu_cast_pair!($src_t, f64, $a, $c, $shape, $strides),
                _ => unreachable!("cast target already validated as float32/float64"),
            }
        };
    }

    match src {
        Dtype::Int8 => dispatch_cast_dst!(i8, a, c, shape, a_strides, dst),
        Dtype::Int16 => dispatch_cast_dst!(i16, a, c, shape, a_strides, dst),
        Dtype::Int32 => dispatch_cast_dst!(i32, a, c, shape, a_strides, dst),
        Dtype::Int64 => dispatch_cast_dst!(i64, a, c, shape, a_strides, dst),
        Dtype::Uint8 => dispatch_cast_dst!(u8, a, c, shape, a_strides, dst),
        Dtype::Uint16 => dispatch_cast_dst!(u16, a, c, shape, a_strides, dst),
        Dtype::Uint32 => dispatch_cast_dst!(u32, a, c, shape, a_strides, dst),
        Dtype::Uint64 => dispatch_cast_dst!(u64, a, c, shape, a_strides, dst),
        Dtype::Float32 => dispatch_cast_dst!(f32, a, c, shape, a_strides, dst),
        Dtype::Float64 => dispatch_cast_dst!(f64, a, c, shape, a_strides, dst),
        _ => unreachable!("cast source already validated"),
    }
}
