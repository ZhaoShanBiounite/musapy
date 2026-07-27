//! OpBuilder — 算子构建器（ADR L1-12, L1-16, L2-4, L2-5）
//!
//! Capture-safe 设计：参数解析（一次性）与 kernel launch（可重放）分离，
//! 为未来 MUSA Graphs capture 保留 lazy hook。
//!
//! 约束（ADR L2-4）：执行阶段不读 host 侧可变状态。

use crate::kernels;
use musapy_core::error::{DeviceError, DtypeError, MemoryError, ShapeError};
use musapy_core::musa_ffi;
use musapy_core::resolution;
use musapy_core::{
    Array, Buffer, BufferRef, Device, DeviceResolution, Dtype, DtypeResolution, Layout, OpContext,
    ResolutionSource, Result, Stream,
};
use std::ptr::NonNull;
use std::sync::Arc;

/// `ms.add(a, b, out=None)` — 逐元素加法。
///
/// 流程：参数解析 → 自动 stream wait → kernel launch → 事件记录 → OpContext。
///
/// - 无 `out=`：分配新 Buffer，在 a 的 stream（或 stream context）上执行
/// - 有 `out=`：写入 out 的 Buffer，在 out 的 stream 上执行（ADR L1-8）
///
/// 别名检测（ADR L2-5）：out 不能同时是输入。
pub fn add(a: &Array, b: &Array, out: Option<&Array>) -> Result<Array> {
    // ═══════════════════════════════════════════════════════════════
    // 参数解析阶段（一次性，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    // 1. Shape 校验（无广播，v0.1-alpha）
    if a.shape() != b.shape() {
        return Err(ShapeError::Mismatch(format!(
            "add: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        ))
        .into());
    }

    // 2. Device 校验
    if a.device() != b.device() {
        return Err(DeviceError::Mismatch(format!(
            "add: device mismatch {} vs {}",
            a.device(),
            b.device()
        ))
        .into());
    }

    // 3. Dtype 校验
    let dtype = a.dtype();
    if dtype != b.dtype() {
        return Err(DtypeError::Unsupported(format!(
            "add: dtype mismatch {} vs {}",
            dtype,
            b.dtype()
        ))
        .into());
    }

    // 4. Dtype 支持（仅 f32/f64，其他 dtype 后续添加）
    match dtype {
        Dtype::Float32 | Dtype::Float64 => {}
        _ => {
            return Err(DtypeError::Unsupported(format!(
                "add: dtype {} not supported (only float32/float64)",
                dtype
            ))
            .into())
        }
    }

    let device = a.device().clone();
    let n = a.size();
    let nbytes = n * dtype.element_size();

    // 5. out= 参数校验（若提供）
    if let Some(o) = out {
        if o.shape() != a.shape() {
            return Err(ShapeError::Mismatch(format!(
                "add: out shape {:?} != input shape {:?}",
                o.shape(),
                a.shape()
            ))
            .into());
        }
        if o.dtype() != dtype {
            return Err(DtypeError::Unsupported(format!(
                "add: out dtype {} != input dtype {}",
                o.dtype(),
                dtype
            ))
            .into());
        }
        if o.device() != a.device() {
            return Err(DeviceError::Mismatch(format!(
                "add: out device {} != input device {}",
                o.device(),
                a.device()
            ))
            .into());
        }
    }

    // 6. Stream 选择（ADR L1-8）
    //    out= → out 的 stream
    //    否则 → stream context（`with ms.stream(s)`）
    //    否则 → a 的 stream（第一个输入）
    let out_stream: Arc<Stream> = match out {
        Some(o) => Arc::clone(o.stream()),
        None => resolution::get_current_stream().unwrap_or_else(|| Arc::clone(a.stream())),
    };

    // 7. out= 处理 + 别名检测（ADR L2-5）
    let (out_data_ref, out_ptr) = match out {
        Some(o) => {
            // 别名检测：out 的 BufferRef 不能与任一输入相同
            if o.data() == a.data() || o.data() == b.data() {
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

    // 8. 自动 stream wait（ADR L1-8）
    //    输入 buffer 可能在另一个 stream 上写入，
    //    让 out_stream 等待输入的写操作完成。
    a.data().buffer().wait_last_write_on(&out_stream)?;
    b.data().buffer().wait_last_write_on(&out_stream)?;

    // ═══════════════════════════════════════════════════════════════
    // Kernel launch 阶段（可重放，capture-safe）
    // ═══════════════════════════════════════════════════════════════

    let a_ptr = a.data().buffer().ptr();
    let b_ptr = b.data().buffer().ptr();

    if n > 0 {
        match &device {
            Device::Cpu => {
                cpu_add(a_ptr, b_ptr, out_ptr, n, dtype);
            }
            Device::Musa(_) => match dtype {
                Dtype::Float32 => {
                    if let (Some(ap), Some(bp), Some(op)) = (a_ptr, b_ptr, out_ptr) {
                        unsafe {
                            kernels::musapy_add_f32_v1(
                                ap.as_ptr() as *const f32,
                                bp.as_ptr() as *const f32,
                                op.as_ptr() as *mut f32,
                                n,
                                out_stream.raw(),
                            );
                        }
                        // P6.10: 即时 launch 错误检查（ADR L3-1）
                        musa_ffi::check_last_kernel_launch("add_f32")?;
                    }
                }
                Dtype::Float64 => {
                    if let (Some(ap), Some(bp), Some(op)) = (a_ptr, b_ptr, out_ptr) {
                        unsafe {
                            kernels::musapy_add_f64_v1(
                                ap.as_ptr() as *const f64,
                                bp.as_ptr() as *const f64,
                                op.as_ptr() as *mut f64,
                                n,
                                out_stream.raw(),
                            );
                        }
                        musa_ffi::check_last_kernel_launch("add_f64")?;
                    }
                }
                _ => unreachable!("dtype already validated as float32/float64"),
            },
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 后处理
    // ═══════════════════════════════════════════════════════════════

    // 9. 事件记录（ADR L3-10）
    a.data().buffer().record_read(&out_stream);
    b.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    // 10. OpContext 记录（ADR L3-2）
    let ctx = OpContext::new(
        "add",
        vec![a.shape().clone(), b.shape().clone()],
        vec![a.device().clone(), b.device().clone()],
        vec![a.dtype(), b.dtype()],
        a.shape().clone(), // 输出 shape = 输入 shape（逐元素）
        out_stream.id(),
    );
    out_stream.record_op(ctx);

    // 11. 构造输出 Array
    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(a.shape().clone()),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

// ── CPU 逐元素加法（不调用 MUSA kernel）──────────────────────

/// 按 dtype 分派 CPU 加法。
fn cpu_add(
    a: Option<NonNull<u8>>,
    b: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    n: usize,
    dtype: Dtype,
) {
    match dtype {
        Dtype::Float32 => cpu_add_typed::<f32>(a, b, c, n),
        Dtype::Float64 => cpu_add_typed::<f64>(a, b, c, n),
        _ => unreachable!("dtype already validated as float32/float64"),
    }
}

/// 泛型 CPU 逐元素加法。
fn cpu_add_typed<T: Copy + std::ops::Add<Output = T>>(
    a: Option<NonNull<u8>>,
    b: Option<NonNull<u8>>,
    c: Option<NonNull<u8>>,
    n: usize,
) {
    if n == 0 {
        return;
    }
    let (Some(ap), Some(bp), Some(cp)) = (a, b, c) else {
        return;
    };
    unsafe {
        let sa = std::slice::from_raw_parts(ap.as_ptr() as *const T, n);
        let sb = std::slice::from_raw_parts(bp.as_ptr() as *const T, n);
        let sc = std::slice::from_raw_parts_mut(cp.as_ptr() as *mut T, n);
        for i in 0..n {
            sc[i] = sa[i] + sb[i];
        }
    }
}
