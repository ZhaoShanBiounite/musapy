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

/// Reduction kernel launch（_v2 stride-aware，沿轴缩减）。
macro_rules! launch_reduce {
    ($fn:ident, $ap:expr, $op:expr, $ndim:expr, $in_shape:expr, $in_strides:expr,
     $axis:expr, $axis_len:expr, $out_size:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(op)) = ($ap, $op) {
            unsafe {
                kernels::$fn(ap.as_ptr() as _, op.as_ptr() as _,
                    $ndim, $in_shape.as_ptr(), $in_strides.as_ptr(),
                    $axis, $axis_len, $out_size, $stream);
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Argmax/Argmin kernel launch（输入 T，输出 i64 索引）。
macro_rules! launch_argreduce {
    ($fn:ident, $ap:expr, $op:expr, $ndim:expr, $in_shape:expr, $in_strides:expr,
     $axis:expr, $axis_len:expr, $out_size:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(op)) = ($ap, $op) {
            unsafe {
                kernels::$fn(ap.as_ptr() as _, op.as_ptr() as _,
                    $ndim, $in_shape.as_ptr(), $in_strides.as_ptr(),
                    $axis, $axis_len, $out_size, $stream);
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Cumsum v3 launch（work-efficient 三阶段，含 scratch buffer）。
macro_rules! launch_cumsum_v3 {
    ($fn:ident, $ap:expr, $op:expr, $tp:expr, $ndim:expr, $in_shape:expr, $in_strides:expr,
     $axis:expr, $axis_len:expr, $out_size:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(op)) = ($ap, $op) {
            // tmp 可能为 null（blocks_per_row==1 时 kernel 不用），用 dangling 指针表达
            let tp_ptr = match $tp { Some(t) => t.as_ptr() as _, None => std::ptr::null_mut() };
            unsafe {
                kernels::$fn(ap.as_ptr() as _, op.as_ptr() as _, tp_ptr,
                    $ndim, $in_shape.as_ptr(), $in_strides.as_ptr(),
                    $axis, $axis_len, $out_size, $stream);
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Parallel reduction partial launch（Phase 1）。
macro_rules! launch_reduce_partial {
    ($fn:ident, $ap:expr, $pp:expr, $ndim:expr, $in_shape:expr, $in_strides:expr,
     $axis:expr, $axis_len:expr, $out_size:expr, $tiles:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(pp)) = ($ap, $pp) {
            unsafe {
                kernels::$fn(ap.as_ptr() as _, pp.as_ptr() as _,
                    $ndim, $in_shape.as_ptr(), $in_strides.as_ptr(),
                    $axis, $axis_len, $out_size, $tiles, $stream);
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Parallel reduction final launch（Phase 2）。
macro_rules! launch_reduce_final {
    ($fn:ident, $pp:expr, $op:expr, $num_partials:expr, $out_size:expr, $stream:expr, $label:expr) => {
        if let (Some(pp), Some(op)) = ($pp, $op) {
            unsafe {
                kernels::$fn(pp.as_ptr() as _, op.as_ptr() as _,
                    $num_partials, $out_size, $stream);
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Mean final launch（额外 axis_len 参数）。
macro_rules! launch_mean_final {
    ($fn:ident, $pp:expr, $op:expr, $num_partials:expr, $out_size:expr, $axis_len:expr, $stream:expr, $label:expr) => {
        if let (Some(pp), Some(op)) = ($pp, $op) {
            unsafe {
                kernels::$fn(pp.as_ptr() as _, op.as_ptr() as _,
                    $num_partials, $out_size, $axis_len, $stream);
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Argmax/Argmin parallel partial launch（Phase 1）。
macro_rules! launch_argreduce_partial {
    ($fn:ident, $ap:expr, $vp:expr, $ip:expr, $ndim:expr, $in_shape:expr, $in_strides:expr,
     $axis:expr, $axis_len:expr, $out_size:expr, $tiles:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(vp), Some(ip)) = ($ap, $vp, $ip) {
            unsafe {
                kernels::$fn(ap.as_ptr() as _, vp.as_ptr() as _, ip.as_ptr() as _,
                    $ndim, $in_shape.as_ptr(), $in_strides.as_ptr(),
                    $axis, $axis_len, $out_size, $tiles, $stream);
            }
            musa_ffi::check_last_kernel_launch($label)?;
        }
    };
}

/// Argmax/Argmin parallel final launch（Phase 2）。
macro_rules! launch_argreduce_final {
    ($fn:ident, $vp:expr, $ip:expr, $op:expr, $num_partials:expr, $out_size:expr, $stream:expr, $label:expr) => {
        if let (Some(vp), Some(ip), Some(op)) = ($vp, $ip, $op) {
            unsafe {
                kernels::$fn(vp.as_ptr() as _, ip.as_ptr() as _, op.as_ptr() as _,
                    $num_partials, $out_size, $stream);
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
                        *base_c.add(idx) = *base_a.offset(off) as $dst_t;
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
        // Phase C-lite：同 shape 时直接用 layout strides，跳过 broadcast 逻辑
        let a_strides: Vec<isize> = if a_work.shape() == &out_shape {
            a_work.layout().strides.clone()
        } else {
            broadcast::broadcast_strides(a_work.layout(), &out_shape)
        };
        let b_strides: Vec<isize> = if b_work.shape() == &out_shape {
            b_work.layout().strides.clone()
        } else {
            broadcast::broadcast_strides(b_work.layout(), &out_shape)
        };
        let ndim = out_shape.len() as i32;
        let stream_raw = out_stream.raw();

        // 调整指针以包含 layout offset（视图支持：flip/slice/index_select）
        let elem_size = dtype.element_size();
        let a_ptr_adj = adjust_ptr_offset(a_ptr, a_work.layout().offset, elem_size);
        let b_ptr_adj = adjust_ptr_offset(b_ptr, b_work.layout().offset, elem_size);

        match &device {
            Device::Cpu => {
                cpu_binary_elementwise(
                    a_ptr_adj, b_ptr_adj, out_ptr, &out_shape, &a_strides, &b_strides, dtype, &kernel,
                );
            }
            Device::Musa(_) => match (&kernel, dtype) {
                (BinaryKernel::Add, Dtype::Float32) => launch_binary!(musapy_add_f32_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "add_f32_v2"),
                (BinaryKernel::Add, Dtype::Float64) => launch_binary!(musapy_add_f64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "add_f64_v2"),
                (BinaryKernel::Sub, Dtype::Float32) => launch_binary!(musapy_sub_f32_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "sub_f32_v2"),
                (BinaryKernel::Sub, Dtype::Float64) => launch_binary!(musapy_sub_f64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "sub_f64_v2"),
                (BinaryKernel::Mul, Dtype::Float32) => launch_binary!(musapy_mul_f32_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "mul_f32_v2"),
                (BinaryKernel::Mul, Dtype::Float64) => launch_binary!(musapy_mul_f64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "mul_f64_v2"),
                (BinaryKernel::Div, Dtype::Float32) => launch_binary!(musapy_div_f32_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "div_f32_v2"),
                (BinaryKernel::Div, Dtype::Float64) => launch_binary!(musapy_div_f64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "div_f64_v2"),
                (BinaryKernel::Pow, Dtype::Float32) => launch_binary!(musapy_pow_f32_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "pow_f32_v2"),
                (BinaryKernel::Pow, Dtype::Float64) => launch_binary!(musapy_pow_f64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "pow_f64_v2"),
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
    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            op_name,
            vec![a.shape().clone(), b.shape().clone()],
            vec![a.device().clone(), b.device().clone()],
            vec![a.dtype(), b.dtype()],
            out_shape.clone(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }

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

    // 调整指针以包含 layout offset（视图支持）
    let a_ptr = adjust_ptr_offset(
        a.data().buffer().ptr(),
        a.layout().offset,
        dtype.element_size(),
    );

    if out_size > 0 {
        // 直接使用输入布局的 strides（无广播，stride-aware）
        let a_strides: Vec<isize> = a.layout().strides.clone();
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
    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            op_name,
            vec![a.shape().clone()],
            vec![a.device().clone()],
            vec![a.dtype()],
            out_shape.clone(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }

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

    // 调整指针以包含 layout offset（视图支持）
    let a_ptr = adjust_ptr_offset(
        a.data().buffer().ptr(),
        a.layout().offset,
        dtype.element_size(),
    );

    if out_size > 0 {
        let a_strides: Vec<isize> = a.layout().strides.clone();
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
    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            op_name,
            vec![a.shape().clone()],
            vec![a.device().clone()],
            vec![a.dtype()],
            out_shape.clone(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }

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
        Dtype::Float32 | Dtype::Float64 | Dtype::Int64 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "cast: target dtype {} not supported (only float32/float64/int64)",
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
                Dtype::Int64 => match src {
                    Dtype::Int8 => launch_cast!(musapy_cast_i8_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i8_i64_v2"),
                    Dtype::Int16 => launch_cast!(musapy_cast_i16_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i16_i64_v2"),
                    Dtype::Int32 => launch_cast!(musapy_cast_i32_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_i32_i64_v2"),
                    Dtype::Uint8 => launch_cast!(musapy_cast_u8_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u8_i64_v2"),
                    Dtype::Uint16 => launch_cast!(musapy_cast_u16_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u16_i64_v2"),
                    Dtype::Uint32 => launch_cast!(musapy_cast_u32_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u32_i64_v2"),
                    Dtype::Uint64 => launch_cast!(musapy_cast_u64_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_u64_i64_v2"),
                    _ => unreachable!("cast pair already validated"),
                },
                _ => unreachable!("cast target already validated as float32/float64/int64"),
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

    // Kernel launch（stride-aware，输入布局 strides + offset 调整）
    let a_strides: Vec<isize> = a.layout().strides.clone();
    let a_ptr = adjust_ptr_offset(
        a.data().buffer().ptr(),
        a.layout().offset,
        src.element_size(),
    );
    launch_cast_kernel(
        a_ptr,
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
    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            "cast",
            vec![shape.clone()],
            vec![device.clone()],
            vec![src],
            shape.clone(),
            stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        stream.record_op(ctx);
    }

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

    // 调整指针以包含 layout offset（视图支持）
    let a_ptr = adjust_ptr_offset(
        a.data().buffer().ptr(),
        a.layout().offset,
        src.element_size(),
    );

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
            let a_strides: Vec<isize> = a.layout().strides.clone();
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
    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            op_name,
            vec![a.shape().clone()],
            vec![a.device().clone()],
            vec![src],
            out_shape.clone(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }

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

    // 调整指针以包含 layout offset（视图支持）
    let a_ptr = adjust_ptr_offset(
        a_work.data().buffer().ptr(),
        a_work.layout().offset,
        dtype.element_size(),
    );
    let b_ptr = adjust_ptr_offset(
        b_work.data().buffer().ptr(),
        b_work.layout().offset,
        dtype.element_size(),
    );

    if out_size > 0 {
        // Phase C-lite：同 shape 时直接用 layout strides，跳过 broadcast 逻辑
        let a_strides: Vec<isize> = if a_work.shape() == &out_shape {
            a_work.layout().strides.clone()
        } else {
            broadcast::broadcast_strides(a_work.layout(), &out_shape)
        };
        let b_strides: Vec<isize> = if b_work.shape() == &out_shape {
            b_work.layout().strides.clone()
        } else {
            broadcast::broadcast_strides(b_work.layout(), &out_shape)
        };
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
    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            op_name,
            vec![a.shape().clone(), b.shape().clone()],
            vec![a.device().clone(), b.device().clone()],
            vec![a.dtype(), b.dtype()],
            out_shape.clone(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }

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

/// 按 layout offset 调整基指针（视图支持：flip/slice/index_select）。
///
/// 视图的逻辑首元素不一定在 buffer 起始处；所有 kernel（CPU/GPU）
/// 均以"调整后的基指针 + strides 相对偏移"访问数据。
/// 负 stride 依赖无符号回绕（mod 2^64），与 common.h offset_nd 语义一致。
pub(crate) fn adjust_ptr_offset(
    ptr: Option<NonNull<u8>>,
    offset: usize,
    elem_size: usize,
) -> Option<NonNull<u8>> {
    if offset == 0 {
        return ptr;
    }
    ptr.map(|p| NonNull::new(unsafe { p.as_ptr().add(offset * elem_size) }).unwrap())
}

/// CPU 端 N 维偏移计算（与 common.h offset_nd 逻辑一致）。
///
/// 返回 isize：负 stride（flip 视图）时 Σ coord*stride 可为负，
/// 由调用方与预调整的基址（`adjust_ptr_offset`）合成最终地址。
pub(crate) fn cpu_offset_nd(linear_idx: usize, shape: &[usize], strides: &[isize]) -> isize {
    let mut offset = 0isize;
    let mut idx = linear_idx;
    for i in (0..shape.len()).rev() {
        let coord = idx % shape[i];
        idx /= shape[i];
        offset += coord as isize * strides[i];
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
            *base_c.add(idx) = cpu_binary_op(*base_a.offset(a_off), *base_b.offset(b_off), kernel);
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
            let av = *base_a.offset(a_off);
            let bv = *base_b.offset(b_off);
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
            *base_c.add(idx) = f(*base_a.offset(a_off));
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
            let v = *base_a.offset(a_off);
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
                Dtype::Int64 => cpu_cast_pair!($src_t, i64, $a, $c, $shape, $strides),
                _ => unreachable!("cast target already validated as float32/float64/int64"),
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

// ── Reduction（Phase 4, ADR-002-D3）──────────────────────────

/// 具体 reduction kernel 标识（用于骨架分派）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ReduceKernel {
    Sum,
    Prod,
    Max,
    Min,
    Mean,
    Argmax,
    Argmin,
}

impl ReduceKernel {
    fn name(&self) -> &'static str {
        match self {
            ReduceKernel::Sum => "sum",
            ReduceKernel::Prod => "prod",
            ReduceKernel::Max => "max",
            ReduceKernel::Min => "min",
            ReduceKernel::Mean => "mean",
            ReduceKernel::Argmax => "argmax",
            ReduceKernel::Argmin => "argmin",
        }
    }

    /// 输出 dtype 是否恒为 i64（argmax/argmin）。
    fn output_is_index(&self) -> bool {
        matches!(self, ReduceKernel::Argmax | ReduceKernel::Argmin)
    }
}

/// 决定 reduction 的 compute dtype（ADR-002-D3 累加规则）。
///
/// - sum/prod/max/min/cumsum：int → i64，float 保持
/// - mean：int → f64，float 保持
/// - argmax/argmin：int → i64，float 保持（输出恒 i64，但 kernel 输入需要 compute dtype）
fn reduction_compute_dtype(input_dtype: Dtype, kernel: &ReduceKernel) -> Dtype {
    match input_dtype {
        Dtype::Float32 => Dtype::Float32,
        Dtype::Float64 => Dtype::Float64,
        // 所有整数类型
        _ => match kernel {
            ReduceKernel::Mean => Dtype::Float64,
            _ => Dtype::Int64,
        },
    }
}

/// 通用 reduction 3-phase 骨架（Phase 4, ADR-002-D3）。
///
/// 支持 sum/prod/max/min/mean/argmax/argmin。
/// axis: 已归一化的轴（None = 全局缩减）。
/// keepdims: 输出是否保留被缩减维（长度 1）。
///
/// Phase A：参数解析 → Phase B：kernel launch → Phase C：后处理
pub(crate) fn reduction_axis(
    a: &Array,
    axis: Option<usize>,
    keepdims: bool,
    out: Option<&Array>,
    kernel: ReduceKernel,
) -> Result<Array> {
    let op_name = kernel.name();

    // ═══════════════════════════════════════════════════════════════
    // Phase A：参数解析（一次性，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    // 1. Device
    let device = a.device().clone();

    // 2. 输出 shape 推导
    let in_shape = a.shape().clone();
    let ndim = in_shape.len();
    let out_shape: Vec<usize> = match axis {
        None => {
            // 全局缩减 → 0-dim（或 keepdims → [1; ndim]）
            if keepdims {
                vec![1; ndim]
            } else {
                vec![]
            }
        }
        Some(ax) => {
            let mut s = Vec::with_capacity(ndim - 1 + if keepdims { 1 } else { 0 });
            for (i, &dim) in in_shape.iter().enumerate() {
                if i == ax {
                    if keepdims {
                        s.push(1);
                    }
                    // else: skip this dim
                } else {
                    s.push(dim);
                }
            }
            s
        }
    };

    // 3. Compute dtype（ADR-002-D3 累加规则）
    let compute_dtype = reduction_compute_dtype(a.dtype(), &kernel);

    // 4. 输出 dtype
    let out_dtype = if kernel.output_is_index() {
        Dtype::Int64
    } else {
        compute_dtype
    };

    // 5. 计算 kernel 参数
    // axis=None → 视为 1D 全缩减：kernel_ndim=1, kernel_shape=[total], kernel_axis=0
    let total_size: usize = in_shape.iter().product();
    let (kernel_ndim, kernel_shape, kernel_axis, axis_len): (i32, Vec<usize>, i32, usize) =
        match axis {
            None => (1, vec![total_size], 0, total_size),
            Some(ax) => (
                ndim as i32,
                in_shape.clone(),
                ax as i32,
                in_shape[ax],
            ),
        };

    let out_size: usize = out_shape.iter().product::<usize>().max(1); // 0-dim scalar → size 1
    let nbytes = out_size * out_dtype.element_size();

    // 6. out= 参数校验
    if let Some(o) = out {
        if o.shape() != &out_shape {
            return Err(ShapeError::Mismatch(format!(
                "{}: out shape {:?} != expected output shape {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != out_dtype {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != expected {}",
                op_name,
                o.dtype(),
                out_dtype
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

    // 7. Stream 选择（ADR L1-8）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 8. 内部 cast（输入 dtype != compute dtype 时）
    let a_cast = (a.dtype() != compute_dtype)
        .then(|| cast_array(a, compute_dtype, &out_stream))
        .transpose()?;
    let a_work: &Array = a_cast.as_ref().unwrap_or(a);

    // 9. out= 处理 + 别名检测（ADR L2-5）
    let (out_data_ref, out_ptr) = match out {
        Some(o) => {
            if o.data() == a_work.data() {
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

    // 10. 自动 stream wait（ADR L1-8）
    a_work.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B：Kernel launch（可重放，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    // axis=None → flatten 缩减；非连续 strides 视图需先物化
    // （连续 strides + offset 的视图由下方指针调整处理）
    let a_flat_holder;
    let a_work: &Array = if axis.is_none() && !a_work.layout().has_contiguous_strides() {
        a_flat_holder = crate::indexing::contiguous(a_work)?;
        a_flat_holder.data().buffer().wait_last_write_on(&out_stream)?;
        &a_flat_holder
    } else {
        a_work
    };

    // 调整指针以包含 layout offset（视图支持）
    let a_ptr = adjust_ptr_offset(
        a_work.data().buffer().ptr(),
        a_work.layout().offset,
        compute_dtype.element_size(),
    );

    if out_size > 0 && axis_len > 0 {
        // 输入 strides（元素单位）
        // axis=None → 视为 1D contiguous（stride=[1]）
        let in_strides: Vec<isize> = match axis {
            None => vec![1],
            Some(_) => a_work
                .layout()
                .strides
                .clone(),
        };
        let stream_raw = out_stream.raw();

        match &device {
            Device::Cpu => {
                cpu_reduction_axis(
                    a_ptr,
                    out_ptr,
                    &kernel_shape,
                    &in_strides,
                    kernel_axis as usize,
                    axis_len,
                    out_size,
                    compute_dtype,
                    &kernel,
                );
            }
            Device::Musa(_) => {
                // 两阶段并行阈值：axis_len > 此值时使用 block-cooperative reduction
                const PARALLEL_REDUCE_THRESHOLD: usize = 1024;

                if axis_len > PARALLEL_REDUCE_THRESHOLD {
                    // ═══ 两阶段并行路径 ═══
                    let tiles_per_output = (axis_len + 255) / 256;
                    let num_partials = tiles_per_output; // per output element
                    let elem_size = compute_dtype.element_size();

                    if kernel.output_is_index() {
                        // argmax/argmin：需要 partials_val + partials_idx 两个 buffer
                        let partial_val_nbytes = out_size * tiles_per_output * elem_size;
                        let partial_idx_nbytes = out_size * tiles_per_output * 8; // i64
                        let partial_val_buf = Buffer::alloc(partial_val_nbytes, device.clone(), &out_stream)?;
                        let partial_idx_buf = Buffer::alloc(partial_idx_nbytes, device.clone(), &out_stream)?;
                        let pv_ptr = partial_val_buf.ptr();
                        let pi_ptr = partial_idx_buf.ptr();

                        match (&kernel, compute_dtype) {
                            (ReduceKernel::Argmax, Dtype::Int64) => {
                                launch_argreduce_partial!(musapy_argmax_partial_i64_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmax_partial_i64_v2");
                                launch_argreduce_final!(musapy_argmax_final_i64_v2, pv_ptr, pi_ptr, out_ptr, num_partials, out_size, stream_raw, "argmax_final_i64_v2");
                            }
                            (ReduceKernel::Argmax, Dtype::Float32) => {
                                launch_argreduce_partial!(musapy_argmax_partial_f32_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmax_partial_f32_v2");
                                launch_argreduce_final!(musapy_argmax_final_f32_v2, pv_ptr, pi_ptr, out_ptr, num_partials, out_size, stream_raw, "argmax_final_f32_v2");
                            }
                            (ReduceKernel::Argmax, Dtype::Float64) => {
                                launch_argreduce_partial!(musapy_argmax_partial_f64_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmax_partial_f64_v2");
                                launch_argreduce_final!(musapy_argmax_final_f64_v2, pv_ptr, pi_ptr, out_ptr, num_partials, out_size, stream_raw, "argmax_final_f64_v2");
                            }
                            (ReduceKernel::Argmin, Dtype::Int64) => {
                                launch_argreduce_partial!(musapy_argmin_partial_i64_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmin_partial_i64_v2");
                                launch_argreduce_final!(musapy_argmin_final_i64_v2, pv_ptr, pi_ptr, out_ptr, num_partials, out_size, stream_raw, "argmin_final_i64_v2");
                            }
                            (ReduceKernel::Argmin, Dtype::Float32) => {
                                launch_argreduce_partial!(musapy_argmin_partial_f32_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmin_partial_f32_v2");
                                launch_argreduce_final!(musapy_argmin_final_f32_v2, pv_ptr, pi_ptr, out_ptr, num_partials, out_size, stream_raw, "argmin_final_f32_v2");
                            }
                            (ReduceKernel::Argmin, Dtype::Float64) => {
                                launch_argreduce_partial!(musapy_argmin_partial_f64_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmin_partial_f64_v2");
                                launch_argreduce_final!(musapy_argmin_final_f64_v2, pv_ptr, pi_ptr, out_ptr, num_partials, out_size, stream_raw, "argmin_final_f64_v2");
                            }
                            _ => unreachable!(),
                        }
                    } else if kernel == ReduceKernel::Mean {
                        // mean：partial 做 sum，final 除以 axis_len
                        let partial_nbytes = out_size * tiles_per_output * elem_size;
                        let partial_buf = Buffer::alloc(partial_nbytes, device.clone(), &out_stream)?;
                        let pp_ptr = partial_buf.ptr();

                        match compute_dtype {
                            Dtype::Float32 => {
                                launch_reduce_partial!(musapy_mean_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "mean_partial_f32_v2");
                                launch_mean_final!(musapy_mean_final_f32_v2, pp_ptr, out_ptr, num_partials, out_size, axis_len, stream_raw, "mean_final_f32_v2");
                            }
                            Dtype::Float64 => {
                                launch_reduce_partial!(musapy_mean_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "mean_partial_f64_v2");
                                launch_mean_final!(musapy_mean_final_f64_v2, pp_ptr, out_ptr, num_partials, out_size, axis_len, stream_raw, "mean_final_f64_v2");
                            }
                            _ => unreachable!("mean only supports float compute dtype"),
                        }
                    } else {
                        // sum/prod/max/min
                        let partial_nbytes = out_size * tiles_per_output * elem_size;
                        let partial_buf = Buffer::alloc(partial_nbytes, device.clone(), &out_stream)?;
                        let pp_ptr = partial_buf.ptr();

                        match (&kernel, compute_dtype) {
                            (ReduceKernel::Sum, Dtype::Int64) => {
                                launch_reduce_partial!(musapy_sum_partial_i64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "sum_partial_i64_v2");
                                launch_reduce_final!(musapy_sum_final_i64_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "sum_final_i64_v2");
                            }
                            (ReduceKernel::Sum, Dtype::Float32) => {
                                launch_reduce_partial!(musapy_sum_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "sum_partial_f32_v2");
                                launch_reduce_final!(musapy_sum_final_f32_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "sum_final_f32_v2");
                            }
                            (ReduceKernel::Sum, Dtype::Float64) => {
                                launch_reduce_partial!(musapy_sum_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "sum_partial_f64_v2");
                                launch_reduce_final!(musapy_sum_final_f64_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "sum_final_f64_v2");
                            }
                            (ReduceKernel::Prod, Dtype::Int64) => {
                                launch_reduce_partial!(musapy_prod_partial_i64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "prod_partial_i64_v2");
                                launch_reduce_final!(musapy_prod_final_i64_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "prod_final_i64_v2");
                            }
                            (ReduceKernel::Prod, Dtype::Float32) => {
                                launch_reduce_partial!(musapy_prod_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "prod_partial_f32_v2");
                                launch_reduce_final!(musapy_prod_final_f32_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "prod_final_f32_v2");
                            }
                            (ReduceKernel::Prod, Dtype::Float64) => {
                                launch_reduce_partial!(musapy_prod_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "prod_partial_f64_v2");
                                launch_reduce_final!(musapy_prod_final_f64_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "prod_final_f64_v2");
                            }
                            (ReduceKernel::Max, Dtype::Int64) => {
                                launch_reduce_partial!(musapy_max_partial_i64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "max_partial_i64_v2");
                                launch_reduce_final!(musapy_max_final_i64_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "max_final_i64_v2");
                            }
                            (ReduceKernel::Max, Dtype::Float32) => {
                                launch_reduce_partial!(musapy_max_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "max_partial_f32_v2");
                                launch_reduce_final!(musapy_max_final_f32_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "max_final_f32_v2");
                            }
                            (ReduceKernel::Max, Dtype::Float64) => {
                                launch_reduce_partial!(musapy_max_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "max_partial_f64_v2");
                                launch_reduce_final!(musapy_max_final_f64_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "max_final_f64_v2");
                            }
                            (ReduceKernel::Min, Dtype::Int64) => {
                                launch_reduce_partial!(musapy_min_partial_i64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "min_partial_i64_v2");
                                launch_reduce_final!(musapy_min_final_i64_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "min_final_i64_v2");
                            }
                            (ReduceKernel::Min, Dtype::Float32) => {
                                launch_reduce_partial!(musapy_min_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "min_partial_f32_v2");
                                launch_reduce_final!(musapy_min_final_f32_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "min_final_f32_v2");
                            }
                            (ReduceKernel::Min, Dtype::Float64) => {
                                launch_reduce_partial!(musapy_min_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "min_partial_f64_v2");
                                launch_reduce_final!(musapy_min_final_f64_v2, pp_ptr, out_ptr, num_partials, out_size, stream_raw, "min_final_f64_v2");
                            }
                            _ => unreachable!(),
                        }
                    }
                } else {
                    // ═══ 原始 naive 路径（axis_len ≤ 阈值）═══
                    if kernel.output_is_index() {
                        // argmax/argmin：输入 compute_dtype，输出 i64
                        match (&kernel, compute_dtype) {
                            (ReduceKernel::Argmax, Dtype::Int64) => launch_argreduce!(musapy_argmax_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "argmax_i64_v2"),
                            (ReduceKernel::Argmax, Dtype::Float32) => launch_argreduce!(musapy_argmax_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "argmax_f32_v2"),
                            (ReduceKernel::Argmax, Dtype::Float64) => launch_argreduce!(musapy_argmax_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "argmax_f64_v2"),
                            (ReduceKernel::Argmin, Dtype::Int64) => launch_argreduce!(musapy_argmin_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "argmin_i64_v2"),
                            (ReduceKernel::Argmin, Dtype::Float32) => launch_argreduce!(musapy_argmin_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "argmin_f32_v2"),
                            (ReduceKernel::Argmin, Dtype::Float64) => launch_argreduce!(musapy_argmin_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "argmin_f64_v2"),
                            _ => unreachable!(),
                        }
                    } else {
                        // sum/prod/max/min/mean：输入输出同 compute_dtype
                        match (&kernel, compute_dtype) {
                            (ReduceKernel::Sum, Dtype::Int64) => launch_reduce!(musapy_sum_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "sum_i64_v2"),
                            (ReduceKernel::Sum, Dtype::Float32) => launch_reduce!(musapy_sum_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "sum_f32_v2"),
                            (ReduceKernel::Sum, Dtype::Float64) => launch_reduce!(musapy_sum_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "sum_f64_v2"),
                            (ReduceKernel::Prod, Dtype::Int64) => launch_reduce!(musapy_prod_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "prod_i64_v2"),
                            (ReduceKernel::Prod, Dtype::Float32) => launch_reduce!(musapy_prod_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "prod_f32_v2"),
                            (ReduceKernel::Prod, Dtype::Float64) => launch_reduce!(musapy_prod_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "prod_f64_v2"),
                            (ReduceKernel::Max, Dtype::Int64) => launch_reduce!(musapy_max_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "max_i64_v2"),
                            (ReduceKernel::Max, Dtype::Float32) => launch_reduce!(musapy_max_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "max_f32_v2"),
                            (ReduceKernel::Max, Dtype::Float64) => launch_reduce!(musapy_max_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "max_f64_v2"),
                            (ReduceKernel::Min, Dtype::Int64) => launch_reduce!(musapy_min_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "min_i64_v2"),
                            (ReduceKernel::Min, Dtype::Float32) => launch_reduce!(musapy_min_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "min_f32_v2"),
                            (ReduceKernel::Min, Dtype::Float64) => launch_reduce!(musapy_min_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "min_f64_v2"),
                            (ReduceKernel::Mean, Dtype::Float32) => launch_reduce!(musapy_mean_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "mean_f32_v2"),
                            (ReduceKernel::Mean, Dtype::Float64) => launch_reduce!(musapy_mean_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "mean_f64_v2"),
                            _ => unreachable!("mean only supports float compute dtype"),
                        }
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C：后处理
    // ═══════════════════════════════════════════════════════════════

    // 事件记录（ADR L3-10）
    a_work.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    // OpContext 记录（ADR L3-2）
    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            op_name,
            vec![a.shape().clone()],
            vec![a.device().clone()],
            vec![a.dtype()],
            out_shape.clone(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }

    // 构造输出 Array
    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        out_dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(out_dtype, ResolutionSource::InputArray),
    ))
}

/// Cumsum 骨架（Phase 4, ADR-002-D3）。
///
/// 与 reduction_axis 不同：输出 shape = 输入 shape（无维度缩减），无 keepdims。
pub(crate) fn cumsum_op(
    a: &Array,
    axis: Option<usize>,
    out: Option<&Array>,
) -> Result<Array> {
    let op_name = "cumsum";

    // ═══════════════════════════════════════════════════════════════
    // Phase A
    // ═══════════════════════════════════════════════════════════════

    let device = a.device().clone();
    let in_shape = a.shape().clone();
    let ndim = in_shape.len();
    let total_size: usize = in_shape.iter().product();

    // axis=None → flatten 后 cumsum（输出 1D）
    let (out_shape, kernel_ndim, kernel_shape, kernel_axis): (Vec<usize>, i32, Vec<usize>, i32) =
        match axis {
            None => (vec![total_size], 1, vec![total_size], 0),
            Some(ax) => (in_shape.clone(), ndim as i32, in_shape.clone(), ax as i32),
        };

    // Compute dtype：int → i64，float 保持
    let compute_dtype = match a.dtype() {
        Dtype::Float32 => Dtype::Float32,
        Dtype::Float64 => Dtype::Float64,
        _ => Dtype::Int64,
    };

    let out_size: usize = out_shape.iter().product();
    let nbytes = out_size * compute_dtype.element_size();

    // out= 校验
    if let Some(o) = out {
        if o.shape() != &out_shape {
            return Err(ShapeError::Mismatch(format!(
                "{}: out shape {:?} != expected {:?}",
                op_name,
                o.shape(),
                out_shape
            ))
            .into());
        }
        if o.dtype() != compute_dtype {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != expected {}",
                op_name,
                o.dtype(),
                compute_dtype
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

    // Stream 选择
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 内部 cast
    let a_cast = (a.dtype() != compute_dtype)
        .then(|| cast_array(a, compute_dtype, &out_stream))
        .transpose()?;
    let a_work: &Array = a_cast.as_ref().unwrap_or(a);

    // Buffer 分配 + alias 检测
    let (out_data_ref, out_ptr) = match out {
        Some(o) => {
            if o.data() == a_work.data() {
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

    a_work.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Phase B
    // ═══════════════════════════════════════════════════════════════

    // axis=None → flatten scan；非连续 strides 视图需先物化
    // （连续 strides + offset 的视图由下方指针调整处理）
    let a_flat_holder;
    let a_work: &Array = if axis.is_none() && !a_work.layout().has_contiguous_strides() {
        a_flat_holder = crate::indexing::contiguous(a_work)?;
        a_flat_holder.data().buffer().wait_last_write_on(&out_stream)?;
        &a_flat_holder
    } else {
        a_work
    };

    // 调整指针以包含 layout offset（视图支持）
    let a_ptr = adjust_ptr_offset(
        a_work.data().buffer().ptr(),
        a_work.layout().offset,
        compute_dtype.element_size(),
    );

    if out_size > 0 {
        // axis=None → 视为 1D contiguous（stride=[1]）
        let in_strides: Vec<isize> = match axis {
            None => vec![1],
            Some(_) => a_work
                .layout()
                .strides
                .clone(),
        };
        let stream_raw = out_stream.raw();

        // axis_len：被累加的轴长度（kernel_shape 已把 axis=None 视为 1D）
        let axis_len: usize = kernel_shape[kernel_axis as usize];

        match &device {
            Device::Cpu => {
                cpu_cumsum(
                    a_ptr,
                    out_ptr,
                    &kernel_shape,
                    &in_strides,
                    kernel_axis as usize,
                    out_size,
                    compute_dtype,
                );
            }
            Device::Musa(_) => {
                // 分层 work-efficient scan：按需分配 scratch buffer。
                // num_rows = out_size / axis_len（每行独立 scan）
                // blocks_per_row = ceil(axis_len / 256)；> 1 时才需要 scratch。
                let blocks_per_row = (axis_len + 255) / 256;
                // 分层扫描容量：Phase 2 为两级 256 宽的 tile scan，
                // blocks_per_row ≤ 256×256 = 65536 → axis_len ≤ 256^3。
                if blocks_per_row > 65536 {
                    return Err(ShapeError::Mismatch(format!(
                        "{}: axis_len {} exceeds max supported length 16777216 (256^3)",
                        op_name, axis_len
                    ))
                    .into());
                }
                let num_rows = out_size / axis_len.max(1);
                // scratch 布局（与 kernel wrapper 约定一致）：
                // block_sums 区（num_rows × blocks_per_row）；
                // blocks_per_row > 256 时其后紧跟 tile_sums 区
                // （num_rows × tiles_per_row，tiles_per_row = ceil(bpr/256)）。
                let scratch_elems = if blocks_per_row > 256 {
                    let tiles_per_row = (blocks_per_row + 255) / 256;
                    num_rows * (blocks_per_row + tiles_per_row)
                } else {
                    num_rows * blocks_per_row
                };
                let tmp_nbytes = scratch_elems * compute_dtype.element_size();
                let tmp_buf = if blocks_per_row > 1 && tmp_nbytes > 0 {
                    Some(Buffer::alloc(tmp_nbytes, device.clone(), &out_stream)?)
                } else {
                    None
                };
                // Buffer::ptr() 返回 Option<NonNull<u8>>，展平为单个 Option
                let tmp_ptr: Option<NonNull<u8>> = tmp_buf.as_ref().and_then(|b| b.ptr());

                match compute_dtype {
                    Dtype::Int64 => launch_cumsum_v3!(musapy_cumsum_i64_v3, a_ptr, out_ptr, tmp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "cumsum_i64_v3"),
                    Dtype::Float32 => launch_cumsum_v3!(musapy_cumsum_f32_v3, a_ptr, out_ptr, tmp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "cumsum_f32_v3"),
                    Dtype::Float64 => launch_cumsum_v3!(musapy_cumsum_f64_v3, a_ptr, out_ptr, tmp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "cumsum_f64_v3"),
                    _ => unreachable!("cumsum compute dtype already validated"),
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase C
    // ═══════════════════════════════════════════════════════════════

    a_work.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            op_name,
            vec![a.shape().clone()],
            vec![a.device().clone()],
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
        out_data_ref,
        Layout::from_shape(out_shape),
        compute_dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(compute_dtype, ResolutionSource::InputArray),
    ))
}

// ── CPU reduction fallback ───────────────────────────────────

/// CPU 端 reduce_input_offset（与 common.h 逻辑一致）。
fn cpu_reduce_offset(out_idx: usize, in_shape: &[usize], in_strides: &[isize], axis: usize, k: usize) -> usize {
    let ndim = in_shape.len();
    let mut coords = [0usize; 32];
    let mut ci = 0;
    let mut tmp = out_idx;
    for i in (0..ndim).rev() {
        if i == axis {
            continue;
        }
        coords[ci] = tmp % in_shape[i];
        tmp /= in_shape[i];
        ci += 1;
    }
    let mut offset = 0isize;
    ci = 0;
    for i in (0..ndim).rev() {
        let coord = if i == axis {
            k
        } else {
            let c = coords[ci];
            ci += 1;
            c
        };
        offset += coord as isize * in_strides[i];
    }
    offset as usize
}

/// 按 dtype 分派 CPU reduction。
fn cpu_reduction_axis(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    in_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    axis_len: usize,
    out_size: usize,
    dtype: Dtype,
    kernel: &ReduceKernel,
) {
    // mean 单独处理（需要除法，只有 float）
    if matches!(kernel, ReduceKernel::Mean) {
        match dtype {
            Dtype::Float32 => cpu_mean_typed::<f32>(a, c, in_shape, in_strides, axis, axis_len, out_size),
            Dtype::Float64 => cpu_mean_typed::<f64>(a, c, in_shape, in_strides, axis, axis_len, out_size),
            _ => unreachable!("mean compute dtype is always float"),
        }
        return;
    }
    match dtype {
        Dtype::Int64 => cpu_reduce_typed::<i64>(a, c, in_shape, in_strides, axis, axis_len, out_size, kernel),
        Dtype::Float32 => cpu_reduce_typed::<f32>(a, c, in_shape, in_strides, axis, axis_len, out_size, kernel),
        Dtype::Float64 => cpu_reduce_typed::<f64>(a, c, in_shape, in_strides, axis, axis_len, out_size, kernel),
        _ => unreachable!("reduction compute dtype already validated"),
    }
}

/// 泛型 CPU reduction（stride-aware，per-output-element 循环累加）。
/// 处理 sum/prod/max/min/argmax/argmin（不含 mean）。
fn cpu_reduce_typed<T>(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    in_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    axis_len: usize,
    out_size: usize,
    kernel: &ReduceKernel,
) where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + From<i8>,
{
    let (Some(ap), Some(cp)) = (a, c) else {
        return;
    };
    if out_size == 0 || axis_len == 0 {
        return;
    }
    unsafe {
        let base_a = ap.as_ptr() as *const T;
        match kernel {
            ReduceKernel::Sum => {
                let base_c = cp.as_ptr() as *mut T;
                for idx in 0..out_size {
                    let base = cpu_reduce_offset(idx, in_shape, in_strides, axis, 0);
                    let axis_stride = in_strides[axis];
                    let mut acc = T::from(0i8);
                    for k in 0..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        acc = acc + *base_a.add(off);
                    }
                    *base_c.add(idx) = acc;
                }
            }
            ReduceKernel::Prod => {
                let base_c = cp.as_ptr() as *mut T;
                for idx in 0..out_size {
                    let base = cpu_reduce_offset(idx, in_shape, in_strides, axis, 0);
                    let axis_stride = in_strides[axis];
                    let mut acc = T::from(1i8);
                    for k in 0..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        acc = acc * *base_a.add(off);
                    }
                    *base_c.add(idx) = acc;
                }
            }
            ReduceKernel::Max | ReduceKernel::Min => {
                let base_c = cp.as_ptr() as *mut T;
                let want_max = matches!(kernel, ReduceKernel::Max);
                for idx in 0..out_size {
                    let base = cpu_reduce_offset(idx, in_shape, in_strides, axis, 0);
                    let axis_stride = in_strides[axis];
                    let mut acc = *base_a.add(base);
                    for k in 1..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        let val = *base_a.add(off);
                        if (want_max && val > acc) || (!want_max && val < acc) {
                            acc = val;
                        }
                    }
                    *base_c.add(idx) = acc;
                }
            }
            ReduceKernel::Argmax | ReduceKernel::Argmin => {
                let base_c = cp.as_ptr() as *mut i64;
                let want_max = matches!(kernel, ReduceKernel::Argmax);
                for idx in 0..out_size {
                    let base = cpu_reduce_offset(idx, in_shape, in_strides, axis, 0);
                    let axis_stride = in_strides[axis];
                    let mut best_val = *base_a.add(base);
                    let mut best_idx: i64 = 0;
                    for k in 1..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        let val = *base_a.add(off);
                        if (want_max && val > best_val) || (!want_max && val < best_val) {
                            best_val = val;
                            best_idx = k as i64;
                        }
                    }
                    *base_c.add(idx) = best_idx;
                }
            }
            ReduceKernel::Mean => unreachable!("mean handled separately"),
        }
    }
}

/// CPU mean（sum / axis_len，只有 float compute dtype）。
fn cpu_mean_typed<T: Copy + std::ops::Add<Output = T> + std::ops::Div<Output = T> + From<i8>>(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    in_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    axis_len: usize,
    out_size: usize,
) {
    let (Some(ap), Some(cp)) = (a, c) else {
        return;
    };
    if out_size == 0 || axis_len == 0 {
        return;
    }
    unsafe {
        let base_a = ap.as_ptr() as *const T;
        let base_c = cp.as_ptr() as *mut T;
        for idx in 0..out_size {
            let base = cpu_reduce_offset(idx, in_shape, in_strides, axis, 0);
            let axis_stride = in_strides[axis];
            let mut acc = T::from(0i8);
            for k in 0..axis_len {
                let off = (base as isize + k as isize * axis_stride) as usize;
                acc = acc + *base_a.add(off);
            }
            // 除以 count：构造 axis_len 的 T 值
            let mut count = T::from(0i8);
            for _ in 0..axis_len {
                count = count + T::from(1i8);
            }
            *base_c.add(idx) = acc / count;
        }
    }
}

/// CPU cumsum fallback。
fn cpu_cumsum(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    in_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    out_size: usize,
    dtype: Dtype,
) {
    match dtype {
        Dtype::Int64 => cpu_cumsum_typed::<i64>(a, c, in_shape, in_strides, axis, out_size),
        Dtype::Float32 => cpu_cumsum_typed::<f32>(a, c, in_shape, in_strides, axis, out_size),
        Dtype::Float64 => cpu_cumsum_typed::<f64>(a, c, in_shape, in_strides, axis, out_size),
        _ => unreachable!("cumsum compute dtype already validated"),
    }
}

/// 泛型 CPU cumsum（prefix sum along axis）。
fn cpu_cumsum_typed<T: Copy + std::ops::Add<Output = T> + From<i8>>(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    in_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    out_size: usize,
) {
    let (Some(ap), Some(cp)) = (a, c) else {
        return;
    };
    if out_size == 0 {
        return;
    }
    let ndim = in_shape.len();
    unsafe {
        let base_a = ap.as_ptr() as *const T;
        let base_c = cp.as_ptr() as *mut T;
        for idx in 0..out_size {
            // 展开 idx 得到 axis 坐标
            let mut tmp = idx;
            let mut axis_coord = 0usize;
            for i in (0..ndim).rev() {
                let coord = tmp % in_shape[i];
                tmp /= in_shape[i];
                if i == axis {
                    axis_coord = coord;
                }
            }
            // 计算 axis=0 时的 base offset
            let mut base_off = 0isize;
            tmp = idx;
            for i in (0..ndim).rev() {
                let coord = tmp % in_shape[i];
                tmp /= in_shape[i];
                if i != axis {
                    base_off += coord as isize * in_strides[i];
                }
            }
            let axis_stride = in_strides[axis];
            let mut acc = T::from(0i8);
            for k in 0..=axis_coord {
                let off = (base_off + k as isize * axis_stride) as usize;
                acc = acc + *base_a.add(off);
            }
            *base_c.add(idx) = acc;
        }
    }
}
