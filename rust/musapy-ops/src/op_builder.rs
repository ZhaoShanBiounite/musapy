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
use musapy_core::musa_x_ffi::{muComplex, muDoubleComplex};
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

/// 小 axis 并行 reduction launch（P2，每输出 group_size 线程）。
macro_rules! launch_reduce_small_axis {
    ($fn:ident, $ap:expr, $op:expr, $ndim:expr, $in_shape:expr, $in_strides:expr,
     $axis:expr, $axis_len:expr, $out_size:expr, $group:expr, $stream:expr, $label:expr) => {
        if let (Some(ap), Some(op)) = ($ap, $op) {
            unsafe {
                kernels::$fn(ap.as_ptr() as _, op.as_ptr() as _,
                    $ndim, $in_shape.as_ptr(), $in_strides.as_ptr(),
                    $axis, $axis_len, $out_size, $group, $stream);
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
/// Mean final launch（额外 axis_len 参数）。
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

    // 4. 计算白名单（f32/f64 + complex64/128，Phase 5；pow 对 complex 不实例化）
    match (&dtype, &kernel) {
        (Dtype::Float32 | Dtype::Float64, _) => {}
        (Dtype::Complex64 | Dtype::Complex128, BinaryKernel::Pow) => {
            return Err(DtypeError::Unsupported(format!(
                "{}: pow not supported for complex dtype {} (Phase 5 complex scope: add/sub/mul/div)",
                op_name, dtype
            ))
            .into());
        }
        (Dtype::Complex64 | Dtype::Complex128, _) => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "{}: promoted dtype {} not supported (compute whitelist: float32/float64/complex64/complex128)",
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
                // complex（Phase 5，ADR-003 003-D5：add/sub/mul/div；pow 白名单已拒）
                (BinaryKernel::Add, Dtype::Complex64) => launch_binary!(musapy_add_c64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "add_c64_v2"),
                (BinaryKernel::Add, Dtype::Complex128) => launch_binary!(musapy_add_c128_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "add_c128_v2"),
                (BinaryKernel::Sub, Dtype::Complex64) => launch_binary!(musapy_sub_c64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "sub_c64_v2"),
                (BinaryKernel::Sub, Dtype::Complex128) => launch_binary!(musapy_sub_c128_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "sub_c128_v2"),
                (BinaryKernel::Mul, Dtype::Complex64) => launch_binary!(musapy_mul_c64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "mul_c64_v2"),
                (BinaryKernel::Mul, Dtype::Complex128) => launch_binary!(musapy_mul_c128_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "mul_c128_v2"),
                (BinaryKernel::Div, Dtype::Complex64) => launch_binary!(musapy_div_c64_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "div_c64_v2"),
                (BinaryKernel::Div, Dtype::Complex128) => launch_binary!(musapy_div_c128_v2, a_ptr_adj, b_ptr_adj, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "div_c128_v2"),
                _ => unreachable!("dtype already validated as float32/float64/complex64/complex128"),
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

    // 1. Dtype 白名单（f32/f64 + complex64/128；complex 仅 neg/abs，Phase 5）
    let dtype = a.dtype();
    match dtype {
        Dtype::Float32 | Dtype::Float64 => {}
        Dtype::Complex64 | Dtype::Complex128 => match kernel {
            UnaryKernel::Neg | UnaryKernel::Abs => {}
            _ => {
                return Err(DtypeError::Unsupported(format!(
                    "{}: {} not supported for complex dtype {} (Phase 5 complex scope: neg/abs)",
                    op_name, kernel.name(), dtype
                ))
                .into());
            }
        },
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "{}: dtype {} not supported (only float32/float64/complex64/complex128)",
                op_name, dtype
            ))
            .into());
        }
    }

    // abs(complex) 输出 real（NumPy：np.abs(complex) → float）；其余输出同输入 dtype。
    let out_dtype = match (dtype, kernel) {
        (Dtype::Complex64, UnaryKernel::Abs) => Dtype::Float32,
        (Dtype::Complex128, UnaryKernel::Abs) => Dtype::Float64,
        _ => dtype,
    };

    let device = a.device().clone();
    let out_shape = a.shape().clone();
    let out_size: usize = out_shape.iter().product();
    let nbytes = out_size * out_dtype.element_size();

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
        if o.dtype() != out_dtype {
            return Err(DtypeError::Unsupported(format!(
                "{}: out dtype {} != expected output dtype {}",
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
                // complex（Phase 5，ADR-003 003-D5：neg/abs；abs 输出 real）
                (UnaryKernel::Neg, Dtype::Complex64) => launch_unary!(musapy_neg_c64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "neg_c64_v2"),
                (UnaryKernel::Neg, Dtype::Complex128) => launch_unary!(musapy_neg_c128_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "neg_c128_v2"),
                (UnaryKernel::Abs, Dtype::Complex64) => launch_unary!(musapy_abs_c64_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "abs_c64_v2"),
                (UnaryKernel::Abs, Dtype::Complex128) => launch_unary!(musapy_abs_c128_v2, a_ptr, out_ptr, ndim, out_shape, a_strides, stream_raw, "abs_c128_v2"),
                _ => unreachable!("dtype already validated as float32/float64/complex64/complex128"),
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

    // 8. 构造输出 Array（连续布局，shape = 输入 shape；abs(complex) 输出 real）
    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(out_shape),
        out_dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(out_dtype, ResolutionSource::InputArray),
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

/// 校验 cast 类型对是否受 Phase 2/5 kernel 支持。
///
/// 支持矩阵（见 kernels.rs `musapy_cast_<src>_<dst>_v2`）：
/// - dst ∈ {float32, float64}（计算白名单），src ∈ {int8..int64, uint8..uint64, float32, float64}
/// - dst ∈ {int64}（reduction 整数累加 + 显式 astype），src ∈ 同上 + {float32, float64}
/// - dst ∈ {complex64, complex128}（Phase 5：real→complex，re=src, im=0；
///   f32/f64 → c64/c128 + c64 → c128 宽度提升）
/// - complex → real、bool/float16/bfloat16 任何方向 尚无 cast kernel（后续 Phase，
///   显式拒绝避免 dispatch 命中 unreachable）
fn validate_cast_pair(src: Dtype, dst: Dtype) -> Result<()> {
    // 源侧先拒绝：complex→real / bool / f16 / bf16 全拒
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
        Dtype::Complex64 | Dtype::Complex128 => {
            // 仅允许 complex 宽度提升 c64 → c128；complex→real 无 kernel，显式拒绝
            if dst == Dtype::Complex128 && src == Dtype::Complex64 {
                return Ok(());
            }
            return Err(DtypeError::Unsupported(format!(
                "cast: {} → {} not supported (complex→real 无 cast kernel，后续 Phase)",
                src, dst
            ))
            .into());
        }
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "cast: source dtype {} not supported (bool/float16/bfloat16 not yet implemented)",
                src
            ))
            .into());
        }
    }
    match dst {
        Dtype::Float32 | Dtype::Float64 | Dtype::Int64 | Dtype::Complex64 | Dtype::Complex128 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "cast: target dtype {} not supported (only float32/float64/int64/complex64/complex128)",
                dst
            ))
            .into());
        }
    }
    // real 源 + complex 目标：仅 f32/f64 → c64/c128
    if matches!(dst, Dtype::Complex64 | Dtype::Complex128) && !matches!(src, Dtype::Float32 | Dtype::Float64) {
        return Err(DtypeError::Unsupported(format!(
            "cast: {} → {} not supported (Phase 5 cast scope: real→complex + c64→c128)",
            src, dst
        ))
        .into());
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
                    Dtype::Float32 => launch_cast!(musapy_cast_f32_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_f32_i64_v2"),
                    Dtype::Float64 => launch_cast!(musapy_cast_f64_i64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_f64_i64_v2"),
                    _ => unreachable!("cast pair already validated"),
                },
                // real → complex（Phase 5，ADR-003 003-D5；re=src, im=0）
                Dtype::Complex64 => match src {
                    Dtype::Float32 => launch_cast!(musapy_cast_f32_c64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_f32_c64_v2"),
                    Dtype::Float64 => launch_cast!(musapy_cast_f64_c64_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_f64_c64_v2"),
                    _ => unreachable!("cast pair already validated"),
                },
                Dtype::Complex128 => match src {
                    Dtype::Float32 => launch_cast!(musapy_cast_f32_c128_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_f32_c128_v2"),
                    Dtype::Float64 => launch_cast!(musapy_cast_f64_c128_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_f64_c128_v2"),
                    Dtype::Complex64 => launch_cast!(musapy_cast_c64_c128_v2, a_ptr, c_ptr, ndim, shape, a_strides, stream_raw, "cast_c64_c128_v2"),
                    _ => unreachable!("cast pair already validated"),
                },
                _ => unreachable!("cast target already validated as float32/float64/int64/complex64/complex128"),
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

    // 4. 计算白名单（f32/f64 + complex eq/ne，Phase 5；lt/gt/le/ge 对 complex 拒绝）
    match (&dtype, &kernel) {
        (Dtype::Float32 | Dtype::Float64, _) => {}
        (Dtype::Complex64 | Dtype::Complex128, CompareKernel::Eq | CompareKernel::Ne) => {}
        (Dtype::Complex64 | Dtype::Complex128, _) => {
            return Err(DtypeError::Unsupported(format!(
                "{}: {} not supported for complex dtype {} (complex has no total order, ADR-003 003-D5; only eq/ne)",
                op_name, kernel.name(), dtype
            ))
            .into());
        }
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "{}: promoted dtype {} not supported (compute whitelist: float32/float64, complex eq/ne)",
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
                // complex eq/ne（Phase 5，ADR-003 003-D5；lt/gt/le/ge 白名单已拒）
                (CompareKernel::Eq, Dtype::Complex64) => launch_compare!(musapy_eq_c64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "eq_c64_v2"),
                (CompareKernel::Eq, Dtype::Complex128) => launch_compare!(musapy_eq_c128_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "eq_c128_v2"),
                (CompareKernel::Ne, Dtype::Complex64) => launch_compare!(musapy_ne_c64_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "ne_c64_v2"),
                (CompareKernel::Ne, Dtype::Complex128) => launch_compare!(musapy_ne_c128_v2, a_ptr, b_ptr, out_ptr, ndim, out_shape, a_strides, b_strides, stream_raw, "ne_c128_v2"),
                _ => unreachable!("dtype already validated as float32/float64/complex64/complex128"),
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
        Dtype::Complex64 => {
            cpu_binary_cplx::<muComplex>(a, b, c, shape, a_strides, b_strides, kernel)
        }
        Dtype::Complex128 => {
            cpu_binary_cplx::<muDoubleComplex>(a, b, c, shape, a_strides, b_strides, kernel)
        }
        _ => unreachable!("dtype already validated as float32/float64/complex64/complex128"),
    }
}

// ── CPU complex binary（Phase 5，ADR-003 003-D5）──────────────

/// 复数分量 trait：让 muComplex/muDoubleComplex 共用一套 CPU 复算。
trait CplxScalar: Copy {
    fn cplx_re(&self) -> f64;
    fn cplx_im(&self) -> f64;
    fn cplx_from_parts(re: f64, im: f64) -> Self;
}

impl CplxScalar for muComplex {
    fn cplx_re(&self) -> f64 {
        self.re as f64
    }
    fn cplx_im(&self) -> f64 {
        self.im as f64
    }
    fn cplx_from_parts(re: f64, im: f64) -> Self {
        muComplex {
            re: re as f32,
            im: im as f32,
        }
    }
}

impl CplxScalar for muDoubleComplex {
    fn cplx_re(&self) -> f64 {
        self.re
    }
    fn cplx_im(&self) -> f64 {
        self.im
    }
    fn cplx_from_parts(re: f64, im: f64) -> Self {
        muDoubleComplex { re, im }
    }
}

/// 泛型 CPU complex binary（stride-aware，re/im 分量公式与 kernel 一致）。
fn cpu_binary_cplx<T: CplxScalar>(
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
            let av = *base_a.offset(a_off);
            let bv = *base_b.offset(b_off);
            let ar = av.cplx_re();
            let ai = av.cplx_im();
            let br = bv.cplx_re();
            let bi = bv.cplx_im();
            let (re, im) = match kernel {
                BinaryKernel::Add => (ar + br, ai + bi),
                BinaryKernel::Sub => (ar - br, ai - bi),
                BinaryKernel::Mul => (ar * br - ai * bi, ar * bi + ai * br),
                BinaryKernel::Div => {
                    let den = br * br + bi * bi;
                    ((ar * br + ai * bi) / den, (ai * br - ar * bi) / den)
                }
                BinaryKernel::Pow => unreachable!("pow rejected for complex by whitelist"),
            };
            *base_c.add(idx) = T::cplx_from_parts(re, im);
        }
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
        // complex（Phase 5：仅 eq/ne，re 与 im 全等才相等）
        Dtype::Complex64 => {
            cpu_compare_cplx::<muComplex>(a, b, c, shape, a_strides, b_strides, kernel)
        }
        Dtype::Complex128 => {
            cpu_compare_cplx::<muDoubleComplex>(a, b, c, shape, a_strides, b_strides, kernel)
        }
        _ => unreachable!("dtype already validated as float32/float64/complex64/complex128"),
    }
}

/// 泛型 CPU complex comparison（仅 eq/ne；lt/gt/le/ge 由白名单拦截）。
fn cpu_compare_cplx<T: CplxScalar>(
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
                CompareKernel::Eq => av.cplx_re() == bv.cplx_re() && av.cplx_im() == bv.cplx_im(),
                CompareKernel::Ne => av.cplx_re() != bv.cplx_re() || av.cplx_im() != bv.cplx_im(),
                _ => unreachable!("ordering comparisons rejected for complex by whitelist"),
            };
            *base_c.add(idx) = if result { 1 } else { 0 };
        }
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
        // complex（Phase 5，ADR-003 003-D5：neg/abs；abs 输出 real）
        Dtype::Complex64 => match kernel {
            UnaryKernel::Neg => cpu_unary_typed::<muComplex>(a, c, shape, a_strides, |v| {
                muComplex { re: -v.re, im: -v.im }
            }),
            UnaryKernel::Abs => cpu_unary_cplx_abs::<muComplex, f32>(a, c, shape, a_strides),
            _ => unreachable!("complex unary whitelist: neg/abs only"),
        },
        Dtype::Complex128 => match kernel {
            UnaryKernel::Neg => {
                cpu_unary_typed::<muDoubleComplex>(a, c, shape, a_strides, |v| {
                    muDoubleComplex { re: -v.re, im: -v.im }
                })
            }
            UnaryKernel::Abs => {
                cpu_unary_cplx_abs::<muDoubleComplex, f64>(a, c, shape, a_strides)
            }
            _ => unreachable!("complex unary whitelist: neg/abs only"),
        },
        _ => unreachable!("dtype already validated as float32/float64/complex64/complex128"),
    }
}

/// 实数标量（abs(complex) 输出类型：c64→f32 / c128→f64）。
trait CplxReal: Copy {
    fn from_f64(x: f64) -> Self;
}

impl CplxReal for f32 {
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
}

impl CplxReal for f64 {
    fn from_f64(x: f64) -> f64 {
        x
    }
}

/// 泛型 CPU complex abs（输出 real：c64→f32 / c128→f64）。
fn cpu_unary_cplx_abs<T: CplxScalar, R: CplxReal>(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
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
        let base_c = cp.as_ptr() as *mut R;
        for idx in 0..n {
            let a_off = cpu_offset_nd(idx, shape, a_strides);
            let v = *base_a.offset(a_off);
            let re = v.cplx_re();
            let im = v.cplx_im();
            *base_c.add(idx) = R::from_f64((re * re + im * im).sqrt());
        }
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
        Dtype::Float32 => match dst {
            Dtype::Complex64 => cpu_cast_cplx::<f32, muComplex>(a, c, shape, a_strides),
            Dtype::Complex128 => cpu_cast_cplx::<f32, muDoubleComplex>(a, c, shape, a_strides),
            _ => dispatch_cast_dst!(f32, a, c, shape, a_strides, dst),
        },
        Dtype::Float64 => match dst {
            Dtype::Complex64 => cpu_cast_cplx::<f64, muComplex>(a, c, shape, a_strides),
            Dtype::Complex128 => cpu_cast_cplx::<f64, muDoubleComplex>(a, c, shape, a_strides),
            _ => dispatch_cast_dst!(f64, a, c, shape, a_strides, dst),
        },
        Dtype::Complex64 => match dst {
            Dtype::Complex128 => cpu_cast_c64_c128(a, c, shape, a_strides),
            _ => unreachable!("cast source already validated"),
        },
        _ => unreachable!("cast source already validated"),
    }
}

/// CPU c64 → c128 宽度提升（re/im 各 f32→f64，无精度损失）。
fn cpu_cast_c64_c128(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
) {
    let n: usize = shape.iter().product();
    if n == 0 {
        return;
    }
    let (Some(ap), Some(cp)) = (a, c) else {
        return;
    };
    unsafe {
        let base_a = ap.as_ptr() as *const muComplex;
        let base_c = cp.as_ptr() as *mut muDoubleComplex;
        for idx in 0..n {
            let off = cpu_offset_nd(idx, shape, a_strides);
            let v = *base_a.offset(off);
            *base_c.add(idx) = muDoubleComplex {
                re: v.re as f64,
                im: v.im as f64,
            };
        }
    }
}

/// real 标量 → f64（cpu_cast_cplx 的源标量统一入口）。
trait CplxCastReal: Copy {
    fn as_f64(self) -> f64;
}

impl CplxCastReal for f32 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}

impl CplxCastReal for f64 {
    fn as_f64(self) -> f64 {
        self
    }
}

/// 泛型 CPU real→complex cast（Phase 5：re=src, im=0；`as` 不支持 struct 目标）。
fn cpu_cast_cplx<S: CplxCastReal, C: CplxScalar>(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    shape: &[usize],
    a_strides: &[isize],
) {
    let n: usize = shape.iter().product();
    if n == 0 {
        return;
    }
    let (Some(ap), Some(cp)) = (a, c) else {
        return;
    };
    unsafe {
        let base_a = ap.as_ptr() as *const S;
        let base_c = cp.as_ptr() as *mut C;
        for idx in 0..n {
            let off = cpu_offset_nd(idx, shape, a_strides);
            let v = (*base_a.offset(off)).as_f64();
            *base_c.add(idx) = C::cplx_from_parts(v, 0.0);
        }
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
/// - complex（Phase 7 P7.2）：sum/mean/prod 保持自身；max/min/arg* 在
///   `reduction_axis` 入口拒绝（复数无全序，ADR-003 003-D5）
fn reduction_compute_dtype(input_dtype: Dtype, kernel: &ReduceKernel) -> Dtype {
    match input_dtype {
        Dtype::Float32 => Dtype::Float32,
        Dtype::Float64 => Dtype::Float64,
        Dtype::Complex64 | Dtype::Complex128 => match kernel {
            ReduceKernel::Max | ReduceKernel::Min | ReduceKernel::Argmax | ReduceKernel::Argmin => {
                unreachable!("complex ordering reduction rejected in reduction_axis")
            }
            _ => input_dtype, // sum/prod/mean 保持 complex
        },
        // 所有整数类型
        _ => match kernel {
            ReduceKernel::Mean => Dtype::Float64,
            _ => Dtype::Int64,
        },
    }
}

/// P2b（2026-08-08）：多级 partial 归约收尾。
///
/// 调用方已生成第一级 partials（`tiles` = ceil(axis_len/1024) 个/输出）。
/// 把 partials 以 (out_size, tiles) 布局逐级喂给 partial kernel 递归归约
/// （每级 ÷1024），直到 tiles ≤ MULTI_STAGE_MIN_TILES——触发阈值即收尾
/// 条件（32K 蕴含 final 扫 ≤256）。中间缓冲在函数内分配，随作用域释放
/// （实际级数 ≤ 2）。
///
/// `launch_partial(src, dst, out_size, tiles)`：把 src（每输出 tiles 个
/// 元素，紧凑布局 [out_size, tiles]）归约到 dst（每输出
/// ceil(tiles/1024) 个）；`launch_final(partials, num_partials)` 收尾。
#[allow(clippy::too_many_arguments)]
fn multi_stage_reduce_tail(
    device: &Device,
    out_stream: &Arc<Stream>,
    out_size: usize,
    elem_size: usize,
    mut src: *mut u8,
    mut tiles: usize,
    launch_partial: impl Fn(*mut u8, *mut u8, usize, usize) -> Result<()>,
    launch_final: impl Fn(*mut u8, usize) -> Result<()>,
) -> Result<()> {
    // 触发阈值（P2b 实测校准，2026-08-08）：tiles > 32K（axis_len > 32M）
    // 才走多级——final 串行扫 partials 的代价是每线程逐次依赖 load
    // （64M 时 ~90µs），而每级多级化要付 ~45µs launch 地板；1M/16M
    // 规模多级化净亏（1M sum 0.085→0.121ms 实测），故设此下限。
    // （MULTI_STAGE_MIN_TILES=32K 已蕴含 final 扫 ≤256 的收尾条件）
    const MULTI_STAGE_MIN_TILES: usize = 32768;
    let mut stage_bufs: Vec<Buffer> = Vec::new();
    while tiles > MULTI_STAGE_MIN_TILES {
        let next_tiles = (tiles + 1023) / 1024;
        let buf = Buffer::alloc(out_size * next_tiles * elem_size, device.clone(), out_stream)?;
        let dst = buf.ptr().expect("multi-stage partial buf").as_ptr();
        launch_partial(src, dst, out_size, tiles)?;
        stage_bufs.push(buf);
        src = dst;
        tiles = next_tiles;
    }
    launch_final(src, tiles)
}

/// P2b（2026-08-08）：argmax/argmin 多级 partial 收尾（val/idx 双缓冲）。
///
/// 与 `multi_stage_reduce_tail` 同构，但 val/idx 两路 partials 同步递归
/// （argmid kernel 沿袭输入对的 idx，不能复用 argreduce_partial——后者
/// 只读值数组、idx 按轴内 k 重新计算）。每级 ÷1024，直到
/// tiles ≤ MULTI_STAGE_MIN_TILES 再走 arg final（32K 已蕴含 final 扫 ≤256）。
#[allow(clippy::too_many_arguments)]
fn multi_stage_arg_reduce_tail(
    device: &Device,
    out_stream: &Arc<Stream>,
    out_size: usize,
    elem_size: usize,
    mut src_val: *mut u8,
    mut src_idx: *mut u8,
    mut tiles: usize,
    launch_partial: impl Fn(*mut u8, *mut u8, *mut u8, *mut u8, usize, usize) -> Result<()>,
    launch_final: impl Fn(*mut u8, *mut u8, usize) -> Result<()>,
) -> Result<()> {
    const MULTI_STAGE_MIN_TILES: usize = 32768;
    let mut stage_bufs: Vec<(Buffer, Buffer)> = Vec::new();
    while tiles > MULTI_STAGE_MIN_TILES {
        let next_tiles = (tiles + 1023) / 1024;
        let vbuf = Buffer::alloc(out_size * next_tiles * elem_size, device.clone(), out_stream)?;
        let ibuf = Buffer::alloc(out_size * next_tiles * 8, device.clone(), out_stream)?;
        let dst_val = vbuf.ptr().expect("multi-stage arg val buf").as_ptr();
        let dst_idx = ibuf.ptr().expect("multi-stage arg idx buf").as_ptr();
        launch_partial(src_val, src_idx, dst_val, dst_idx, out_size, tiles)?;
        stage_bufs.push((vbuf, ibuf));
        src_val = dst_val;
        src_idx = dst_idx;
        tiles = next_tiles;
    }
    launch_final(src_val, src_idx, tiles)
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

    // 1b. complex 拒绝（Phase 7 P7.2，ADR-003 003-D5）：max/min/argmax/argmin
    //     对复数无全序，显式抛 DtypeError（替代 Phase 7 前的隐式 cast 失败）
    if matches!(a.dtype(), Dtype::Complex64 | Dtype::Complex128)
        && matches!(
            kernel,
            ReduceKernel::Max | ReduceKernel::Min | ReduceKernel::Argmax | ReduceKernel::Argmin
        )
    {
        return Err(DtypeError::Unsupported(format!(
            "{}: ordering reduction not supported for complex dtype {} (complex has no total order, ADR-003 003-D5)",
            op_name,
            a.dtype()
        ))
        .into());
    }

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
                // 路径选择（P2：按 axis_len 双维选择）：
                //   axis_len > 1024           → 两阶段并行（partial 每线程 4 元素 + final）
                //   16 < axis_len ≤ 1024      → 小 axis 并行（每输出 32..256 线程组）
                //   axis_len ≤ 16 / arg* 等   → naive one-thread-per-output
                const PARALLEL_REDUCE_THRESHOLD: usize = 1024;
                const SMALL_AXIS_MIN: usize = 16;

                // Phase 7 P7.2：complex 归约支持三路并行（2026-08-08 优化：
                // 分量 re/im 各一路 shuffle 的 small_axis/partial/final kernel）

                if axis_len > PARALLEL_REDUCE_THRESHOLD {
                    // ═══ 两阶段并行路径 ═══
                    // P2：partial kernel 每线程 REDUCE_ITEMS=4 元素，
                    // 一个 tile（256 线程）覆盖 1024 个元素
                    let tiles_per_output = (axis_len + 1023) / 1024;
                    let elem_size = compute_dtype.element_size();

                    if kernel.output_is_index() {
                        // argmax/argmin：需要 partials_val + partials_idx 两个 buffer。
                        // P2b：tiles > 256 时经 argmid kernel 逐级 ÷1024
                        // （(val, idx) 对归约，idx 沿袭输入），final 只扫 ≤256 对。
                        let partial_val_nbytes = out_size * tiles_per_output * elem_size;
                        let partial_idx_nbytes = out_size * tiles_per_output * 8; // i64
                        let partial_val_buf = Buffer::alloc(partial_val_nbytes, device.clone(), &out_stream)?;
                        let partial_idx_buf = Buffer::alloc(partial_idx_nbytes, device.clone(), &out_stream)?;
                        let pv_ptr = partial_val_buf.ptr();
                        let pi_ptr = partial_idx_buf.ptr();

                        // 中间级 arg 归约 launch（读 (val, idx) 对，写缩小后的对）
                        macro_rules! launch_argmid {
                            ($fn:ident, $sv:expr, $si:expr, $dv:expr, $di:expr, $os:expr, $tiles:expr, $axis_len:expr, $label:expr) => {{
                                unsafe {
                                    kernels::$fn(
                                        $sv as *const _,
                                        $si as *const _,
                                        $dv as *mut _,
                                        $di as *mut _,
                                        $os,
                                        $tiles,
                                        $axis_len,
                                        stream_raw,
                                    );
                                }
                                musa_ffi::check_last_kernel_launch($label)
                            }};
                        }
                        // final：num_partials ≤ FINAL_TILES_THRESHOLD
                        macro_rules! launch_argfinal_tail {
                            ($fn:ident, $vp:expr, $ip:expr, $np:expr, $label:expr) => {{
                                if let Some(op) = out_ptr {
                                    unsafe {
                                        kernels::$fn(
                                            $vp as *const _,
                                            $ip as *const _,
                                            op.as_ptr() as _,
                                            $np,
                                            out_size,
                                            stream_raw,
                                        );
                                    }
                                    musa_ffi::check_last_kernel_launch($label)
                                } else {
                                    Ok(())
                                }
                            }};
                        }

                        match (&kernel, compute_dtype) {
                            (ReduceKernel::Argmax, Dtype::Int64) => {
                                launch_argreduce_partial!(musapy_argmax_partial_i64_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmax_partial_i64_v2");
                                multi_stage_arg_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pv_ptr.expect("arg partial val buf").as_ptr(), pi_ptr.expect("arg partial idx buf").as_ptr(), tiles_per_output,
                                    |sv, si, dv, di, os, tiles| {
                                        launch_argmid!(musapy_argmax_mid_i64_v2, sv, si, dv, di, os, (tiles + 1023) / 1024, tiles, "argmax_mid_i64_v2")
                                    },
                                    |vp, ip, np| {
                                        launch_argfinal_tail!(musapy_argmax_final_i64_v2, vp, ip, np, "argmax_final_i64_v2")
                                    },
                                )?;
                            }
                            (ReduceKernel::Argmax, Dtype::Float32) => {
                                launch_argreduce_partial!(musapy_argmax_partial_f32_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmax_partial_f32_v2");
                                multi_stage_arg_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pv_ptr.expect("arg partial val buf").as_ptr(), pi_ptr.expect("arg partial idx buf").as_ptr(), tiles_per_output,
                                    |sv, si, dv, di, os, tiles| {
                                        launch_argmid!(musapy_argmax_mid_f32_v2, sv, si, dv, di, os, (tiles + 1023) / 1024, tiles, "argmax_mid_f32_v2")
                                    },
                                    |vp, ip, np| {
                                        launch_argfinal_tail!(musapy_argmax_final_f32_v2, vp, ip, np, "argmax_final_f32_v2")
                                    },
                                )?;
                            }
                            (ReduceKernel::Argmax, Dtype::Float64) => {
                                launch_argreduce_partial!(musapy_argmax_partial_f64_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmax_partial_f64_v2");
                                multi_stage_arg_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pv_ptr.expect("arg partial val buf").as_ptr(), pi_ptr.expect("arg partial idx buf").as_ptr(), tiles_per_output,
                                    |sv, si, dv, di, os, tiles| {
                                        launch_argmid!(musapy_argmax_mid_f64_v2, sv, si, dv, di, os, (tiles + 1023) / 1024, tiles, "argmax_mid_f64_v2")
                                    },
                                    |vp, ip, np| {
                                        launch_argfinal_tail!(musapy_argmax_final_f64_v2, vp, ip, np, "argmax_final_f64_v2")
                                    },
                                )?;
                            }
                            (ReduceKernel::Argmin, Dtype::Int64) => {
                                launch_argreduce_partial!(musapy_argmin_partial_i64_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmin_partial_i64_v2");
                                multi_stage_arg_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pv_ptr.expect("arg partial val buf").as_ptr(), pi_ptr.expect("arg partial idx buf").as_ptr(), tiles_per_output,
                                    |sv, si, dv, di, os, tiles| {
                                        launch_argmid!(musapy_argmin_mid_i64_v2, sv, si, dv, di, os, (tiles + 1023) / 1024, tiles, "argmin_mid_i64_v2")
                                    },
                                    |vp, ip, np| {
                                        launch_argfinal_tail!(musapy_argmin_final_i64_v2, vp, ip, np, "argmin_final_i64_v2")
                                    },
                                )?;
                            }
                            (ReduceKernel::Argmin, Dtype::Float32) => {
                                launch_argreduce_partial!(musapy_argmin_partial_f32_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmin_partial_f32_v2");
                                multi_stage_arg_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pv_ptr.expect("arg partial val buf").as_ptr(), pi_ptr.expect("arg partial idx buf").as_ptr(), tiles_per_output,
                                    |sv, si, dv, di, os, tiles| {
                                        launch_argmid!(musapy_argmin_mid_f32_v2, sv, si, dv, di, os, (tiles + 1023) / 1024, tiles, "argmin_mid_f32_v2")
                                    },
                                    |vp, ip, np| {
                                        launch_argfinal_tail!(musapy_argmin_final_f32_v2, vp, ip, np, "argmin_final_f32_v2")
                                    },
                                )?;
                            }
                            (ReduceKernel::Argmin, Dtype::Float64) => {
                                launch_argreduce_partial!(musapy_argmin_partial_f64_v2, a_ptr, pv_ptr, pi_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "argmin_partial_f64_v2");
                                multi_stage_arg_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pv_ptr.expect("arg partial val buf").as_ptr(), pi_ptr.expect("arg partial idx buf").as_ptr(), tiles_per_output,
                                    |sv, si, dv, di, os, tiles| {
                                        launch_argmid!(musapy_argmin_mid_f64_v2, sv, si, dv, di, os, (tiles + 1023) / 1024, tiles, "argmin_mid_f64_v2")
                                    },
                                    |vp, ip, np| {
                                        launch_argfinal_tail!(musapy_argmin_final_f64_v2, vp, ip, np, "argmin_final_f64_v2")
                                    },
                                )?;
                            }
                            _ => unreachable!(),
                        }
                    } else if kernel == ReduceKernel::Mean {
                        // mean：partial 做 sum，final 除以 axis_len。
                        // P2b：中间级用 sum partial（与 mean_partial 同计算），
                        // 最终 mean_final 带原始 axis_len。
                        let partial_nbytes = out_size * tiles_per_output * elem_size;
                        let partial_buf = Buffer::alloc(partial_nbytes, device.clone(), &out_stream)?;
                        let pp_ptr = partial_buf.ptr();

                        match compute_dtype {
                            Dtype::Float32 => {
                                launch_reduce_partial!(musapy_mean_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "mean_partial_f32_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| {
                                        let shape = [os, tiles];
                                        let strides = [tiles as isize, 1];
                                        unsafe {
                                            kernels::musapy_sum_partial_f32_v2(
                                                src as *const f32, dst as *mut f32, 2,
                                                shape.as_ptr(), strides.as_ptr(), 1, tiles, os,
                                                (tiles + 1023) / 1024, stream_raw,
                                            );
                                        }
                                        musa_ffi::check_last_kernel_launch("sum_partial_f32_v2_mid")
                                    },
                                    |pp, np| {
                                        if let Some(op) = out_ptr {
                                            unsafe {
                                                kernels::musapy_mean_final_f32_v2(
                                                    pp as *const f32, op.as_ptr() as *mut f32,
                                                    np, out_size, axis_len, stream_raw,
                                                );
                                            }
                                            musa_ffi::check_last_kernel_launch("mean_final_f32_v2")
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )?;
                            }
                            Dtype::Float64 => {
                                launch_reduce_partial!(musapy_mean_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "mean_partial_f64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| {
                                        let shape = [os, tiles];
                                        let strides = [tiles as isize, 1];
                                        unsafe {
                                            kernels::musapy_sum_partial_f64_v2(
                                                src as *const f64, dst as *mut f64, 2,
                                                shape.as_ptr(), strides.as_ptr(), 1, tiles, os,
                                                (tiles + 1023) / 1024, stream_raw,
                                            );
                                        }
                                        musa_ffi::check_last_kernel_launch("sum_partial_f64_v2_mid")
                                    },
                                    |pp, np| {
                                        if let Some(op) = out_ptr {
                                            unsafe {
                                                kernels::musapy_mean_final_f64_v2(
                                                    pp as *const f64, op.as_ptr() as *mut f64,
                                                    np, out_size, axis_len, stream_raw,
                                                );
                                            }
                                            musa_ffi::check_last_kernel_launch("mean_final_f64_v2")
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )?;
                            }
                            Dtype::Complex64 => {
                                launch_reduce_partial!(musapy_mean_partial_c64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "mean_partial_c64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| {
                                        let shape = [os, tiles];
                                        let strides = [tiles as isize, 1];
                                        unsafe {
                                            kernels::musapy_sum_partial_c64_v2(
                                                src as *const muComplex,
                                                dst as *mut muComplex, 2,
                                                shape.as_ptr(), strides.as_ptr(), 1, tiles, os,
                                                (tiles + 1023) / 1024, stream_raw,
                                            );
                                        }
                                        musa_ffi::check_last_kernel_launch("sum_partial_c64_v2_mid")
                                    },
                                    |pp, np| {
                                        if let Some(op) = out_ptr {
                                            unsafe {
                                                kernels::musapy_mean_final_c64_v2(
                                                    pp as *const muComplex,
                                                    op.as_ptr() as *mut muComplex,
                                                    np, out_size, axis_len, stream_raw,
                                                );
                                            }
                                            musa_ffi::check_last_kernel_launch("mean_final_c64_v2")
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )?;
                            }
                            Dtype::Complex128 => {
                                launch_reduce_partial!(musapy_mean_partial_c128_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "mean_partial_c128_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| {
                                        let shape = [os, tiles];
                                        let strides = [tiles as isize, 1];
                                        unsafe {
                                            kernels::musapy_sum_partial_c128_v2(
                                                src as *const muDoubleComplex,
                                                dst as *mut muDoubleComplex, 2,
                                                shape.as_ptr(), strides.as_ptr(), 1, tiles, os,
                                                (tiles + 1023) / 1024, stream_raw,
                                            );
                                        }
                                        musa_ffi::check_last_kernel_launch("sum_partial_c128_v2_mid")
                                    },
                                    |pp, np| {
                                        if let Some(op) = out_ptr {
                                            unsafe {
                                                kernels::musapy_mean_final_c128_v2(
                                                    pp as *const muDoubleComplex,
                                                    op.as_ptr() as *mut muDoubleComplex,
                                                    np, out_size, axis_len, stream_raw,
                                                );
                                            }
                                            musa_ffi::check_last_kernel_launch("mean_final_c128_v2")
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )?;
                            }
                            _ => unreachable!("mean only supports float compute dtype"),
                        }
                    } else {
                        // sum/prod/max/min
                        let partial_nbytes = out_size * tiles_per_output * elem_size;
                        let partial_buf = Buffer::alloc(partial_nbytes, device.clone(), &out_stream)?;
                        let pp_ptr = partial_buf.ptr();

                        // 中间级 partial launch（partials 以 [out_size, tiles] 布局递归）
                        macro_rules! launch_partial_mid {
                            ($fn:ident, $src:expr, $dst:expr, $os:expr, $tiles:expr, $label:expr) => {{
                                let shape = [$os, $tiles];
                                let strides = [$tiles as isize, 1];
                                unsafe {
                                    kernels::$fn(
                                        $src as *const _,
                                        $dst as *mut _,
                                        2,
                                        shape.as_ptr(),
                                        strides.as_ptr(),
                                        1,
                                        $tiles,
                                        $os,
                                        ($tiles + 1023) / 1024,
                                        stream_raw,
                                    );
                                }
                                musa_ffi::check_last_kernel_launch($label)
                            }};
                        }
                        // final（num_partials ≤ FINAL_TILES_THRESHOLD）
                        macro_rules! launch_final_tail {
                            ($fn:ident, $pp:expr, $np:expr, $label:expr) => {{
                                if let Some(op) = out_ptr {
                                    unsafe {
                                        kernels::$fn(
                                            $pp as *const _,
                                            op.as_ptr() as _,
                                            $np,
                                            out_size,
                                            stream_raw,
                                        );
                                    }
                                    musa_ffi::check_last_kernel_launch($label)
                                } else {
                                    Ok(())
                                }
                            }};
                        }

                        match (&kernel, compute_dtype) {
                            (ReduceKernel::Sum, Dtype::Int64) => {
                                launch_reduce_partial!(musapy_sum_partial_i64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "sum_partial_i64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_sum_partial_i64_v2, src, dst, os, tiles, "sum_partial_i64_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_sum_final_i64_v2, pp, np, "sum_final_i64_v2"),
                                )?;
                            }
                            (ReduceKernel::Sum, Dtype::Float32) => {
                                launch_reduce_partial!(musapy_sum_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "sum_partial_f32_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_sum_partial_f32_v2, src, dst, os, tiles, "sum_partial_f32_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_sum_final_f32_v2, pp, np, "sum_final_f32_v2"),
                                )?;
                            }
                            (ReduceKernel::Sum, Dtype::Float64) => {
                                launch_reduce_partial!(musapy_sum_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "sum_partial_f64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_sum_partial_f64_v2, src, dst, os, tiles, "sum_partial_f64_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_sum_final_f64_v2, pp, np, "sum_final_f64_v2"),
                                )?;
                            }
                            // complex sum（Phase 7 优化：分量 partial/final）
                            (ReduceKernel::Sum, Dtype::Complex64) => {
                                launch_reduce_partial!(musapy_sum_partial_c64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "sum_partial_c64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_sum_partial_c64_v2, src, dst, os, tiles, "sum_partial_c64_v2_mid"),
                                    |pp, np| {
                                        if let Some(op) = out_ptr {
                                            unsafe {
                                                kernels::musapy_sum_final_c64_v2(
                                                    pp as *const muComplex,
                                                    op.as_ptr() as *mut muComplex,
                                                    np, out_size, axis_len, stream_raw,
                                                );
                                            }
                                            musa_ffi::check_last_kernel_launch("sum_final_c64_v2")
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )?;
                            }
                            (ReduceKernel::Sum, Dtype::Complex128) => {
                                launch_reduce_partial!(musapy_sum_partial_c128_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "sum_partial_c128_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_sum_partial_c128_v2, src, dst, os, tiles, "sum_partial_c128_v2_mid"),
                                    |pp, np| {
                                        if let Some(op) = out_ptr {
                                            unsafe {
                                                kernels::musapy_sum_final_c128_v2(
                                                    pp as *const muDoubleComplex,
                                                    op.as_ptr() as *mut muDoubleComplex,
                                                    np, out_size, axis_len, stream_raw,
                                                );
                                            }
                                            musa_ffi::check_last_kernel_launch("sum_final_c128_v2")
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )?;
                            }
                            (ReduceKernel::Prod, Dtype::Int64) => {
                                launch_reduce_partial!(musapy_prod_partial_i64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "prod_partial_i64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_prod_partial_i64_v2, src, dst, os, tiles, "prod_partial_i64_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_prod_final_i64_v2, pp, np, "prod_final_i64_v2"),
                                )?;
                            }
                            (ReduceKernel::Prod, Dtype::Float32) => {
                                launch_reduce_partial!(musapy_prod_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "prod_partial_f32_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_prod_partial_f32_v2, src, dst, os, tiles, "prod_partial_f32_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_prod_final_f32_v2, pp, np, "prod_final_f32_v2"),
                                )?;
                            }
                            (ReduceKernel::Prod, Dtype::Float64) => {
                                launch_reduce_partial!(musapy_prod_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "prod_partial_f64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_prod_partial_f64_v2, src, dst, os, tiles, "prod_partial_f64_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_prod_final_f64_v2, pp, np, "prod_final_f64_v2"),
                                )?;
                            }
                            // complex prod（Phase 7 优化：分量 partial/final）
                            (ReduceKernel::Prod, Dtype::Complex64) => {
                                launch_reduce_partial!(musapy_prod_partial_c64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "prod_partial_c64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_prod_partial_c64_v2, src, dst, os, tiles, "prod_partial_c64_v2_mid"),
                                    |pp, np| {
                                        if let Some(op) = out_ptr {
                                            unsafe {
                                                kernels::musapy_prod_final_c64_v2(
                                                    pp as *const muComplex,
                                                    op.as_ptr() as *mut muComplex,
                                                    np, out_size, axis_len, stream_raw,
                                                );
                                            }
                                            musa_ffi::check_last_kernel_launch("prod_final_c64_v2")
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )?;
                            }
                            (ReduceKernel::Prod, Dtype::Complex128) => {
                                launch_reduce_partial!(musapy_prod_partial_c128_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "prod_partial_c128_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_prod_partial_c128_v2, src, dst, os, tiles, "prod_partial_c128_v2_mid"),
                                    |pp, np| {
                                        if let Some(op) = out_ptr {
                                            unsafe {
                                                kernels::musapy_prod_final_c128_v2(
                                                    pp as *const muDoubleComplex,
                                                    op.as_ptr() as *mut muDoubleComplex,
                                                    np, out_size, axis_len, stream_raw,
                                                );
                                            }
                                            musa_ffi::check_last_kernel_launch("prod_final_c128_v2")
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )?;
                            }
                            (ReduceKernel::Max, Dtype::Int64) => {
                                launch_reduce_partial!(musapy_max_partial_i64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "max_partial_i64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_max_partial_i64_v2, src, dst, os, tiles, "max_partial_i64_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_max_final_i64_v2, pp, np, "max_final_i64_v2"),
                                )?;
                            }
                            (ReduceKernel::Max, Dtype::Float32) => {
                                launch_reduce_partial!(musapy_max_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "max_partial_f32_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_max_partial_f32_v2, src, dst, os, tiles, "max_partial_f32_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_max_final_f32_v2, pp, np, "max_final_f32_v2"),
                                )?;
                            }
                            (ReduceKernel::Max, Dtype::Float64) => {
                                launch_reduce_partial!(musapy_max_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "max_partial_f64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_max_partial_f64_v2, src, dst, os, tiles, "max_partial_f64_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_max_final_f64_v2, pp, np, "max_final_f64_v2"),
                                )?;
                            }
                            (ReduceKernel::Min, Dtype::Int64) => {
                                launch_reduce_partial!(musapy_min_partial_i64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "min_partial_i64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_min_partial_i64_v2, src, dst, os, tiles, "min_partial_i64_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_min_final_i64_v2, pp, np, "min_final_i64_v2"),
                                )?;
                            }
                            (ReduceKernel::Min, Dtype::Float32) => {
                                launch_reduce_partial!(musapy_min_partial_f32_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "min_partial_f32_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_min_partial_f32_v2, src, dst, os, tiles, "min_partial_f32_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_min_final_f32_v2, pp, np, "min_final_f32_v2"),
                                )?;
                            }
                            (ReduceKernel::Min, Dtype::Float64) => {
                                launch_reduce_partial!(musapy_min_partial_f64_v2, a_ptr, pp_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, tiles_per_output, stream_raw, "min_partial_f64_v2");
                                multi_stage_reduce_tail(
                                    &device, &out_stream, out_size, elem_size, pp_ptr.expect("reduce partial buf").as_ptr(), tiles_per_output,
                                    |src, dst, os, tiles| launch_partial_mid!(musapy_min_partial_f64_v2, src, dst, os, tiles, "min_partial_f64_v2_mid"),
                                    |pp, np| launch_final_tail!(musapy_min_final_f64_v2, pp, np, "min_final_f64_v2"),
                                )?;
                            }
                            _ => unreachable!(),
                        }
                    }
                } else if axis_len > SMALL_AXIS_MIN && !kernel.output_is_index() {
                    // ═══ 小 axis 并行路径（P2）═══
                    // 每输出配 group_size 线程（≥ axis_len 向上取 2 的幂，
                    // 上限 256），修 naive 在 out_size 小时并行度不足的问题
                    let group_size: i32 = if axis_len <= 32 {
                        32
                    } else if axis_len <= 64 {
                        64
                    } else if axis_len <= 128 {
                        128
                    } else {
                        256
                    };
                    match (&kernel, compute_dtype) {
                        (ReduceKernel::Sum, Dtype::Int64) => launch_reduce_small_axis!(musapy_sum_small_axis_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "sum_small_axis_i64_v2"),
                        (ReduceKernel::Sum, Dtype::Float32) => launch_reduce_small_axis!(musapy_sum_small_axis_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "sum_small_axis_f32_v2"),
                        (ReduceKernel::Sum, Dtype::Float64) => launch_reduce_small_axis!(musapy_sum_small_axis_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "sum_small_axis_f64_v2"),
                        (ReduceKernel::Prod, Dtype::Int64) => launch_reduce_small_axis!(musapy_prod_small_axis_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "prod_small_axis_i64_v2"),
                        (ReduceKernel::Prod, Dtype::Float32) => launch_reduce_small_axis!(musapy_prod_small_axis_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "prod_small_axis_f32_v2"),
                        (ReduceKernel::Prod, Dtype::Float64) => launch_reduce_small_axis!(musapy_prod_small_axis_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "prod_small_axis_f64_v2"),
                        (ReduceKernel::Max, Dtype::Int64) => launch_reduce_small_axis!(musapy_max_small_axis_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "max_small_axis_i64_v2"),
                        (ReduceKernel::Max, Dtype::Float32) => launch_reduce_small_axis!(musapy_max_small_axis_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "max_small_axis_f32_v2"),
                        (ReduceKernel::Max, Dtype::Float64) => launch_reduce_small_axis!(musapy_max_small_axis_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "max_small_axis_f64_v2"),
                        (ReduceKernel::Min, Dtype::Int64) => launch_reduce_small_axis!(musapy_min_small_axis_i64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "min_small_axis_i64_v2"),
                        (ReduceKernel::Min, Dtype::Float32) => launch_reduce_small_axis!(musapy_min_small_axis_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "min_small_axis_f32_v2"),
                        (ReduceKernel::Min, Dtype::Float64) => launch_reduce_small_axis!(musapy_min_small_axis_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "min_small_axis_f64_v2"),
                        (ReduceKernel::Mean, Dtype::Float32) => launch_reduce_small_axis!(musapy_mean_small_axis_f32_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "mean_small_axis_f32_v2"),
                        (ReduceKernel::Mean, Dtype::Float64) => launch_reduce_small_axis!(musapy_mean_small_axis_f64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "mean_small_axis_f64_v2"),
                        // complex（Phase 7 优化：分量 small_axis）
                        (ReduceKernel::Sum, Dtype::Complex64) => launch_reduce_small_axis!(musapy_sum_small_axis_c64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "sum_small_axis_c64_v2"),
                        (ReduceKernel::Sum, Dtype::Complex128) => launch_reduce_small_axis!(musapy_sum_small_axis_c128_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "sum_small_axis_c128_v2"),
                        (ReduceKernel::Prod, Dtype::Complex64) => launch_reduce_small_axis!(musapy_prod_small_axis_c64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "prod_small_axis_c64_v2"),
                        (ReduceKernel::Prod, Dtype::Complex128) => launch_reduce_small_axis!(musapy_prod_small_axis_c128_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "prod_small_axis_c128_v2"),
                        (ReduceKernel::Mean, Dtype::Complex64) => launch_reduce_small_axis!(musapy_mean_small_axis_c64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "mean_small_axis_c64_v2"),
                        (ReduceKernel::Mean, Dtype::Complex128) => launch_reduce_small_axis!(musapy_mean_small_axis_c128_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, group_size, stream_raw, "mean_small_axis_c128_v2"),
                        _ => unreachable!(),
                    }
                } else {
                    // ═══ 原始 naive 路径（axis_len ≤ 16 或 argmax/argmin）═══
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
                            // complex（Phase 7 P7.2：sum/prod/mean；naive 路径）
                            (ReduceKernel::Sum, Dtype::Complex64) => launch_reduce!(musapy_sum_c64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "sum_c64_v2"),
                            (ReduceKernel::Sum, Dtype::Complex128) => launch_reduce!(musapy_sum_c128_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "sum_c128_v2"),
                            (ReduceKernel::Prod, Dtype::Complex64) => launch_reduce!(musapy_prod_c64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "prod_c64_v2"),
                            (ReduceKernel::Prod, Dtype::Complex128) => launch_reduce!(musapy_prod_c128_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "prod_c128_v2"),
                            (ReduceKernel::Mean, Dtype::Complex64) => launch_reduce!(musapy_mean_c64_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "mean_c64_v2"),
                            (ReduceKernel::Mean, Dtype::Complex128) => launch_reduce!(musapy_mean_c128_v2, a_ptr, out_ptr, kernel_ndim, kernel_shape, in_strides, kernel_axis, axis_len, out_size, stream_raw, "mean_c128_v2"),
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
                // out_size > 0 保证 num_rows ≥ 1，blocks_per_row > 1 时
                // tmp_nbytes 恒 > 0（P6 简化：原多余判断）
                let tmp_buf = if blocks_per_row > 1 {
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
    // complex（Phase 7 P7.2：sum/prod/mean；max/min/arg* 已在 reduction_axis 拒绝）
    if matches!(dtype, Dtype::Complex64 | Dtype::Complex128) {
        match dtype {
            Dtype::Complex64 => {
                cpu_reduce_cplx::<muComplex>(a, c, in_shape, in_strides, axis, axis_len, out_size, kernel)
            }
            Dtype::Complex128 => {
                cpu_reduce_cplx::<muDoubleComplex>(a, c, in_shape, in_strides, axis, axis_len, out_size, kernel)
            }
            _ => unreachable!(),
        }
        return;
    }
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

/// 泛型 CPU complex 归约（Phase 7 P7.2：sum/prod/mean；窄化由 CplxScalar 内完成）。
fn cpu_reduce_cplx<T: CplxScalar>(
    a: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    in_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    axis_len: usize,
    out_size: usize,
    kernel: &ReduceKernel,
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
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            match kernel {
                ReduceKernel::Sum | ReduceKernel::Mean => {
                    for k in 0..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        let v = *base_a.add(off);
                        re += v.cplx_re();
                        im += v.cplx_im();
                    }
                }
                ReduceKernel::Prod => {
                    re = 1.0;
                    for k in 0..axis_len {
                        let off = (base as isize + k as isize * axis_stride) as usize;
                        let v = *base_a.add(off);
                        let (br, bi) = (v.cplx_re(), v.cplx_im());
                        let (ar, ai) = (re, im);
                        re = ar * br - ai * bi;
                        im = ar * bi + ai * br;
                    }
                }
                _ => unreachable!("complex ordering reduction rejected"),
            }
            if matches!(kernel, ReduceKernel::Mean) {
                re /= axis_len as f64;
                im /= axis_len as f64;
            }
            *base_c.add(idx) = T::cplx_from_parts(re, im);
        }
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
