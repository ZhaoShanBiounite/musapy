//! Creation 算子公开 API（v0.2-alpha, Phase 5）
//!
//! 创建算子无输入 Array，输出始终 C-contiguous。
//! 遵循 3-phase 骨架（ADR L1-12, L2-4）：
//!   Phase A: device/dtype resolution + buffer 分配
//!   Phase B: kernel launch（GPU）或 Rust 循环（CPU）
//!   Phase C: event 记录 + Array 构造
//!
//! dtype 推断规则（ADR-002-D5）：
//!   - zeros/ones/full/eye: resolve_dtype(arg, &[]) → float32 兜底
//!   - arange: 全整数参数 → int64；含浮点 → float64；显式 dtype 覆盖
//!   - linspace: 默认 float64；显式 dtype 覆盖
//!   - zeros_like/ones_like: 继承输入 Array 的 dtype/device（跳过 resolution chain）

use crate::kernels;
use musapy_core::error::ShapeError;
use musapy_core::musa_ffi;
use musapy_core::resolution;
use musapy_core::{
    Array, Buffer, BufferRef, Device, DeviceResolution, Dtype, DtypeResolution, Layout,
    ResolutionSource, Result, Stream,
};
use std::sync::Arc;

// ── FillAction：描述填充方式 ─────────────────────────────────

/// 创建算子的填充动作（内部使用）。
enum FillAction {
    /// 写入常量值（zeros/ones/full）
    Fill(f64),
    /// out[i] = start + i * step（arange）
    Arange { start: f64, step: f64 },
    /// out[i] = start + i * (stop - start) / (n - 1)（linspace）
    Linspace { start: f64, stop: f64 },
    /// eye: out[row*m + col] = (col - row == k) ? 1 : 0
    Eye { m: usize, k: i32 },
}

// ── 骨架函数 ─────────────────────────────────────────────────

/// 创建算子通用骨架：解析 device/dtype → 分配 buffer → 填充 → 构造 Array。
///
/// `n_rows` 仅用于 Eye（shape = [n_rows, m]）；其他算子忽略。
fn creation_skeleton(
    shape: &[usize],
    device_arg: Option<Device>,
    dtype_arg: Option<Dtype>,
    action: &FillAction,
) -> Result<Array> {
    // ── Phase A: 参数解析（capture-safe）──

    // 1. Device resolution（无输入，Level 3 跳过）
    let dev_res = resolution::resolve_device(device_arg, &[])?;
    let device = dev_res.device.clone();

    // 2. Dtype resolution（无输入，Level 3 跳过）
    let dtype_res = resolution::resolve_dtype(dtype_arg, &[])?;
    let dtype = dtype_res.dtype;

    // 3. Layout + nbytes
    let layout = Layout::from_shape(shape.to_vec());
    let n = layout.size();
    let nbytes = n * dtype.element_size();

    // 4. Stream 选择
    let stream: Arc<Stream> = resolution::get_current_stream()
        .unwrap_or_else(|| Arc::new(Stream::new(device.clone(), 0).unwrap()));

    // 5. Buffer 分配（自动走 buffer pool）
    let buffer = Buffer::alloc(nbytes.max(1), device.clone(), &stream)?;
    let buf_arc = Arc::new(buffer);
    let data_ref = BufferRef::new(buf_arc);

    // ── Phase B: 填充（kernel launch 或 CPU 循环）──

    if n > 0 {
        match &device {
            Device::Cpu => {
                cpu_fill(data_ref.buffer().ptr(), dtype, n, action, shape)?;
            }
            Device::Musa(_) => {
                let stream_raw = stream.raw();
                let out_ptr = data_ref.buffer().ptr();
                if let Some(ptr) = out_ptr {
                    let dispatched =
                        gpu_dispatch(ptr.as_ptr(), dtype, n, action, shape, stream_raw)?;
                    if !dispatched {
                        // 未实例化 kernel 的 dtype：CPU 填充 + H2D memcpy
                        cpu_fill_to_device(ptr.as_ptr(), dtype, n, action, shape, &device)?;
                    }
                }
            }
        }
    }

    // ── Phase C: 后处理 ──

    data_ref.buffer().record_write(&stream);

    Ok(Array::new(
        data_ref, layout, dtype, stream, dev_res, dtype_res,
    ))
}

/// 尝试 GPU kernel 分发。返回 true 表示已成功分发，false 表示需要 fallback。
fn gpu_dispatch(
    out_ptr: *mut u8,
    dtype: Dtype,
    n: usize,
    action: &FillAction,
    shape: &[usize],
    stream_raw: musa_ffi::musaStream_t,
) -> Result<bool> {
    match action {
        FillAction::Fill(value) => {
            let v = *value;
            match dtype {
                Dtype::Float32 => unsafe {
                    kernels::musapy_fill_f32(out_ptr as *mut f32, v as f32, n, stream_raw);
                },
                Dtype::Float64 => unsafe {
                    kernels::musapy_fill_f64(out_ptr as *mut f64, v, n, stream_raw);
                },
                Dtype::Int64 => unsafe {
                    kernels::musapy_fill_i64(out_ptr as *mut i64, v as i64, n, stream_raw);
                },
                Dtype::Int32 => unsafe {
                    kernels::musapy_fill_i32(out_ptr as *mut i32, v as i32, n, stream_raw);
                },
                Dtype::Int16 => unsafe {
                    kernels::musapy_fill_i16(out_ptr as *mut i16, v as i16, n, stream_raw);
                },
                Dtype::Int8 => unsafe {
                    kernels::musapy_fill_i8(out_ptr as *mut i8, v as i8, n, stream_raw);
                },
                Dtype::Uint64 => unsafe {
                    kernels::musapy_fill_u64(out_ptr as *mut u64, v as u64, n, stream_raw);
                },
                Dtype::Uint32 => unsafe {
                    kernels::musapy_fill_u32(out_ptr as *mut u32, v as u32, n, stream_raw);
                },
                Dtype::Uint16 => unsafe {
                    kernels::musapy_fill_u16(out_ptr as *mut u16, v as u16, n, stream_raw);
                },
                Dtype::Uint8 => unsafe {
                    kernels::musapy_fill_u8(out_ptr, v as u8, n, stream_raw);
                },
                _ => return Ok(false), // complex 等未实例化
            }
            musa_ffi::check_last_kernel_launch("fill")?;
            Ok(true)
        }
        FillAction::Arange { start, step } => {
            let (s, st) = (*start, *step);
            match dtype {
                Dtype::Float32 => unsafe {
                    kernels::musapy_arange_f32(
                        out_ptr as *mut f32,
                        s as f32,
                        st as f32,
                        n,
                        stream_raw,
                    );
                },
                Dtype::Float64 => unsafe {
                    kernels::musapy_arange_f64(out_ptr as *mut f64, s, st, n, stream_raw);
                },
                Dtype::Int64 => unsafe {
                    kernels::musapy_arange_i64(
                        out_ptr as *mut i64,
                        s as i64,
                        st as i64,
                        n,
                        stream_raw,
                    );
                },
                Dtype::Int32 => unsafe {
                    kernels::musapy_arange_i32(
                        out_ptr as *mut i32,
                        s as i32,
                        st as i32,
                        n,
                        stream_raw,
                    );
                },
                _ => return Ok(false),
            }
            musa_ffi::check_last_kernel_launch("arange")?;
            Ok(true)
        }
        FillAction::Linspace { start, stop } => {
            let (s, e) = (*start, *stop);
            match dtype {
                Dtype::Float32 => unsafe {
                    kernels::musapy_linspace_f32(
                        out_ptr as *mut f32,
                        s as f32,
                        e as f32,
                        n,
                        stream_raw,
                    );
                },
                Dtype::Float64 => unsafe {
                    kernels::musapy_linspace_f64(out_ptr as *mut f64, s, e, n, stream_raw);
                },
                _ => return Ok(false),
            }
            musa_ffi::check_last_kernel_launch("linspace")?;
            Ok(true)
        }
        FillAction::Eye { m, k } => {
            let (m, k) = (*m, *k);
            let n_rows = shape[0];
            match dtype {
                Dtype::Float32 => unsafe {
                    kernels::musapy_eye_f32(out_ptr as *mut f32, n_rows, m, k, stream_raw);
                },
                Dtype::Float64 => unsafe {
                    kernels::musapy_eye_f64(out_ptr as *mut f64, n_rows, m, k, stream_raw);
                },
                Dtype::Int64 => unsafe {
                    kernels::musapy_eye_i64(out_ptr as *mut i64, n_rows, m, k, stream_raw);
                },
                Dtype::Int32 => unsafe {
                    kernels::musapy_eye_i32(out_ptr as *mut i32, n_rows, m, k, stream_raw);
                },
                _ => return Ok(false),
            }
            musa_ffi::check_last_kernel_launch("eye")?;
            Ok(true)
        }
    }
}

/// CPU 填充（直接写入 buffer 内存）。
fn cpu_fill(
    ptr: Option<std::ptr::NonNull<u8>>,
    dtype: Dtype,
    n: usize,
    action: &FillAction,
    shape: &[usize],
) -> Result<()> {
    let ptr = match ptr {
        Some(p) => p.as_ptr(),
        None => return Ok(()),
    };
    unsafe { cpu_fill_ptr(ptr, dtype, n, action, shape) }
}

/// CPU 填充 + H2D memcpy（GPU 上未实例化 kernel 的 dtype fallback）。
fn cpu_fill_to_device(
    dev_ptr: *mut u8,
    dtype: Dtype,
    n: usize,
    action: &FillAction,
    shape: &[usize],
    device: &Device,
) -> Result<()> {
    let nbytes = n * dtype.element_size();
    let mut host_buf = vec![0u8; nbytes];
    unsafe {
        cpu_fill_ptr(host_buf.as_mut_ptr(), dtype, n, action, shape)?;
        // H2D memcpy
        musa_ffi::check_musa(
            musa_ffi::musaMemcpy(
                dev_ptr as *mut std::ffi::c_void,
                host_buf.as_ptr() as *const std::ffi::c_void,
                nbytes,
                musa_ffi::musaMemcpyKind::HostToDevice,
            ),
            "musaMemcpy(H2D creation fallback)",
        )?;
    }
    let _ = device;
    Ok(())
}

/// 向指定指针写入填充数据（unsafe，调用者保证指针有效）。
unsafe fn cpu_fill_ptr(
    ptr: *mut u8,
    dtype: Dtype,
    n: usize,
    action: &FillAction,
    shape: &[usize],
) -> Result<()> {
    unsafe {
        match action {
            FillAction::Fill(value) => {
                let v = *value;
                match dtype {
                    Dtype::Float32 => {
                        let p = ptr as *mut f32;
                        for i in 0..n {
                            *p.add(i) = v as f32;
                        }
                    }
                    Dtype::Float64 => {
                        let p = ptr as *mut f64;
                        for i in 0..n {
                            *p.add(i) = v;
                        }
                    }
                    Dtype::Int64 => {
                        let p = ptr as *mut i64;
                        for i in 0..n {
                            *p.add(i) = v as i64;
                        }
                    }
                    Dtype::Int32 => {
                        let p = ptr as *mut i32;
                        for i in 0..n {
                            *p.add(i) = v as i32;
                        }
                    }
                    Dtype::Int16 => {
                        let p = ptr as *mut i16;
                        for i in 0..n {
                            *p.add(i) = v as i16;
                        }
                    }
                    Dtype::Int8 => {
                        let p = ptr as *mut i8;
                        for i in 0..n {
                            *p.add(i) = v as i8;
                        }
                    }
                    Dtype::Uint64 => {
                        let p = ptr as *mut u64;
                        for i in 0..n {
                            *p.add(i) = v as u64;
                        }
                    }
                    Dtype::Uint32 => {
                        let p = ptr as *mut u32;
                        for i in 0..n {
                            *p.add(i) = v as u32;
                        }
                    }
                    Dtype::Uint16 => {
                        let p = ptr as *mut u16;
                        for i in 0..n {
                            *p.add(i) = v as u16;
                        }
                    }
                    Dtype::Uint8 => {
                        let p = ptr;
                        for i in 0..n {
                            *p.add(i) = v as u8;
                        }
                    }
                    Dtype::Bool => {
                        let p = ptr;
                        let bv: u8 = if v != 0.0 { 1 } else { 0 };
                        for i in 0..n {
                            *p.add(i) = bv;
                        }
                    }
                    _ => {
                        // Float16/Bfloat16/Complex: 暂不支持 CPU fill
                        return Err(musapy_core::error::DtypeError::Unsupported(format!(
                            "creation ops do not support dtype {:?}",
                            dtype
                        ))
                        .into());
                    }
                }
            }
            FillAction::Arange { start, step } => {
                let (s, st) = (*start, *step);
                match dtype {
                    Dtype::Float32 => {
                        let p = ptr as *mut f32;
                        for i in 0..n {
                            *p.add(i) = (s + i as f64 * st) as f32;
                        }
                    }
                    Dtype::Float64 => {
                        let p = ptr as *mut f64;
                        for i in 0..n {
                            *p.add(i) = s + i as f64 * st;
                        }
                    }
                    Dtype::Int64 => {
                        let p = ptr as *mut i64;
                        for i in 0..n {
                            *p.add(i) = (s + i as f64 * st) as i64;
                        }
                    }
                    Dtype::Int32 => {
                        let p = ptr as *mut i32;
                        for i in 0..n {
                            *p.add(i) = (s + i as f64 * st) as i32;
                        }
                    }
                    _ => {
                        return Err(musapy_core::error::DtypeError::Unsupported(format!(
                            "arange does not support dtype {:?}",
                            dtype
                        ))
                        .into());
                    }
                }
            }
            FillAction::Linspace { start, stop } => {
                let (s, e) = (*start, *stop);
                match dtype {
                    Dtype::Float32 => {
                        let p = ptr as *mut f32;
                        if n == 1 {
                            *p = s as f32;
                        } else {
                            let step = (e - s) / (n - 1) as f64;
                            for i in 0..n {
                                *p.add(i) = (s + i as f64 * step) as f32;
                            }
                        }
                    }
                    Dtype::Float64 => {
                        let p = ptr as *mut f64;
                        if n == 1 {
                            *p = s;
                        } else {
                            let step = (e - s) / (n - 1) as f64;
                            for i in 0..n {
                                *p.add(i) = s + i as f64 * step;
                            }
                        }
                    }
                    _ => {
                        return Err(musapy_core::error::DtypeError::Unsupported(format!(
                            "linspace does not support dtype {:?}",
                            dtype
                        ))
                        .into());
                    }
                }
            }
            FillAction::Eye { m, k } => {
                let (m, k) = (*m, *k);
                let n_rows = shape[0];
                match dtype {
                    Dtype::Float32 => {
                        let p = ptr as *mut f32;
                        for idx in 0..n {
                            let row = idx / m;
                            let col = idx % m;
                            *p.add(idx) = if (col as i32 - row as i32) == k {
                                1.0
                            } else {
                                0.0
                            };
                        }
                    }
                    Dtype::Float64 => {
                        let p = ptr as *mut f64;
                        for idx in 0..n {
                            let row = idx / m;
                            let col = idx % m;
                            *p.add(idx) = if (col as i32 - row as i32) == k {
                                1.0
                            } else {
                                0.0
                            };
                        }
                    }
                    Dtype::Int64 => {
                        let p = ptr as *mut i64;
                        for idx in 0..n {
                            let row = idx / m;
                            let col = idx % m;
                            *p.add(idx) = if (col as i32 - row as i32) == k { 1 } else { 0 };
                        }
                    }
                    Dtype::Int32 => {
                        let p = ptr as *mut i32;
                        for idx in 0..n {
                            let row = idx / m;
                            let col = idx % m;
                            *p.add(idx) = if (col as i32 - row as i32) == k { 1 } else { 0 };
                        }
                    }
                    _ => {
                        let _ = n_rows;
                        return Err(musapy_core::error::DtypeError::Unsupported(format!(
                            "eye does not support dtype {:?}",
                            dtype
                        ))
                        .into());
                    }
                }
            }
        }
    } // unsafe
    Ok(())
}

// ── 公开 API ─────────────────────────────────────────────────

/// `ms.zeros(shape, dtype=None, device=None)` — 创建全零数组。
///
/// dtype 默认 float32（L0-7 级 5 兜底）。
pub fn zeros(shape: &[usize], dtype: Option<Dtype>, device: Option<Device>) -> Result<Array> {
    creation_skeleton(shape, device, dtype, &FillAction::Fill(0.0))
}

/// `ms.ones(shape, dtype=None, device=None)` — 创建全一数组。
///
/// dtype 默认 float32（L0-7 级 5 兜底）。
pub fn ones(shape: &[usize], dtype: Option<Dtype>, device: Option<Device>) -> Result<Array> {
    creation_skeleton(shape, device, dtype, &FillAction::Fill(1.0))
}

/// `ms.full(shape, fill_value, dtype=None, device=None)` — 创建填充指定值的数组。
///
/// dtype 默认 float32（L0-7 级 5 兜底）。
pub fn full(
    shape: &[usize],
    value: f64,
    dtype: Option<Dtype>,
    device: Option<Device>,
) -> Result<Array> {
    creation_skeleton(shape, device, dtype, &FillAction::Fill(value))
}

/// `ms.eye(n, m=None, k=0, dtype=None, device=None)` — 创建单位矩阵。
///
/// - `n`: 行数
/// - `m`: 列数（默认 = n）
/// - `k`: 对角线偏移（0 = 主对角线，>0 上移，<0 下移）
///
/// dtype 默认 float32。
pub fn eye(
    n: usize,
    m: Option<usize>,
    k: i32,
    dtype: Option<Dtype>,
    device: Option<Device>,
) -> Result<Array> {
    let m = m.unwrap_or(n);
    let shape = [n, m];
    creation_skeleton(&shape, device, dtype, &FillAction::Eye { m, k })
}

/// `ms.arange(start, stop=None, step=1, dtype=None, device=None)` — 创建等差序列。
///
/// dtype 推断（NumPy 行为）：
/// - 全整数参数 → int64
/// - 含浮点参数 → float64
/// - 显式 dtype= 覆盖推断
pub fn arange(
    start: f64,
    stop: Option<f64>,
    step: f64,
    dtype: Option<Dtype>,
    device: Option<Device>,
) -> Result<Array> {
    // 处理单参数形式：arange(stop) → start=0, stop=arg
    let (start, stop) = match stop {
        Some(s) => (start, s),
        None => (0.0, start),
    };

    // step == 0 → 错误
    if step == 0.0 {
        return Err(ShapeError::Mismatch("arange: step must not be zero".to_string()).into());
    }

    // 计算长度
    let n = arange_len(start, stop, step);

    // dtype 推断（ADR-002-D5）
    let dtype_arg = match dtype {
        Some(d) => Some(d),
        None => {
            let all_integer = start.fract() == 0.0 && stop.fract() == 0.0 && step.fract() == 0.0;
            Some(if all_integer {
                Dtype::Int64
            } else {
                Dtype::Float64
            })
        }
    };

    creation_skeleton(&[n], device, dtype_arg, &FillAction::Arange { start, step })
}

/// `ms.linspace(start, stop, num=50, dtype=None, device=None)` — 创建等间隔序列。
///
/// 默认 dtype = float64（NumPy 行为）。
/// num=0 → 空数组；num=1 → [start]。
pub fn linspace(
    start: f64,
    stop: f64,
    num: usize,
    dtype: Option<Dtype>,
    device: Option<Device>,
) -> Result<Array> {
    // linspace 默认 float64（NumPy 行为）
    let dtype_arg = Some(dtype.unwrap_or(Dtype::Float64));
    creation_skeleton(
        &[num],
        device,
        dtype_arg,
        &FillAction::Linspace { start, stop },
    )
}

/// `ms.zeros_like(a)` — 创建与输入同 shape/dtype/device 的全零数组。
///
/// 完全绕过 resolution chain，直接继承输入属性（ADR L3-18）。
pub fn zeros_like(a: &Array) -> Result<Array> {
    let shape = a.shape().to_vec();
    let dtype = a.dtype();
    let device = a.device().clone();

    let layout = Layout::from_shape(shape);
    let n = layout.size();
    let nbytes = n * dtype.element_size();

    let stream: Arc<Stream> = resolution::get_current_stream()
        .unwrap_or_else(|| Arc::new(Stream::new(device.clone(), 0).unwrap()));

    let buffer = Buffer::alloc(nbytes.max(1), device.clone(), &stream)?;
    let buf_arc = Arc::new(buffer);
    let data_ref = BufferRef::new(buf_arc);

    if n > 0 {
        match &device {
            Device::Cpu => {
                cpu_fill(
                    data_ref.buffer().ptr(),
                    dtype,
                    n,
                    &FillAction::Fill(0.0),
                    a.shape(),
                )?;
            }
            Device::Musa(_) => {
                let stream_raw = stream.raw();
                if let Some(ptr) = data_ref.buffer().ptr() {
                    let dispatched = gpu_dispatch(
                        ptr.as_ptr(),
                        dtype,
                        n,
                        &FillAction::Fill(0.0),
                        a.shape(),
                        stream_raw,
                    )?;
                    if !dispatched {
                        cpu_fill_to_device(
                            ptr.as_ptr(),
                            dtype,
                            n,
                            &FillAction::Fill(0.0),
                            a.shape(),
                            &device,
                        )?;
                    }
                }
            }
        }
    }

    data_ref.buffer().record_write(&stream);

    // 继承输入的 resolution（source = InputArray）
    Ok(Array::new(
        data_ref,
        layout,
        dtype,
        stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

/// `ms.ones_like(a)` — 创建与输入同 shape/dtype/device 的全一数组。
///
/// 完全绕过 resolution chain，直接继承输入属性（ADR L3-18）。
pub fn ones_like(a: &Array) -> Result<Array> {
    let shape = a.shape().to_vec();
    let dtype = a.dtype();
    let device = a.device().clone();

    let layout = Layout::from_shape(shape);
    let n = layout.size();
    let nbytes = n * dtype.element_size();

    let stream: Arc<Stream> = resolution::get_current_stream()
        .unwrap_or_else(|| Arc::new(Stream::new(device.clone(), 0).unwrap()));

    let buffer = Buffer::alloc(nbytes.max(1), device.clone(), &stream)?;
    let buf_arc = Arc::new(buffer);
    let data_ref = BufferRef::new(buf_arc);

    if n > 0 {
        match &device {
            Device::Cpu => {
                cpu_fill(
                    data_ref.buffer().ptr(),
                    dtype,
                    n,
                    &FillAction::Fill(1.0),
                    a.shape(),
                )?;
            }
            Device::Musa(_) => {
                let stream_raw = stream.raw();
                if let Some(ptr) = data_ref.buffer().ptr() {
                    let dispatched = gpu_dispatch(
                        ptr.as_ptr(),
                        dtype,
                        n,
                        &FillAction::Fill(1.0),
                        a.shape(),
                        stream_raw,
                    )?;
                    if !dispatched {
                        cpu_fill_to_device(
                            ptr.as_ptr(),
                            dtype,
                            n,
                            &FillAction::Fill(1.0),
                            a.shape(),
                            &device,
                        )?;
                    }
                }
            }
        }
    }

    data_ref.buffer().record_write(&stream);

    Ok(Array::new(
        data_ref,
        layout,
        dtype,
        stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

// ── 辅助函数 ─────────────────────────────────────────────────

/// 计算 arange 输出长度（NumPy 语义）。
fn arange_len(start: f64, stop: f64, step: f64) -> usize {
    if step > 0.0 && start >= stop {
        return 0;
    }
    if step < 0.0 && start <= stop {
        return 0;
    }
    let len = ((stop - start) / step).ceil();
    if len <= 0.0 { 0 } else { len as usize }
}

// ============================================================
// 单元测试（CPU 路径）
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use musapy_core::Device;

    fn read_f32(a: &Array) -> Vec<f32> {
        let n = a.size();
        let mut out = vec![0f32; n];
        if let Some(ptr) = a.data().buffer().ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    n * 4,
                );
            }
        }
        out
    }

    fn read_f64(a: &Array) -> Vec<f64> {
        let n = a.size();
        let mut out = vec![0f64; n];
        if let Some(ptr) = a.data().buffer().ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    n * 8,
                );
            }
        }
        out
    }

    fn read_i64(a: &Array) -> Vec<i64> {
        let n = a.size();
        let mut out = vec![0i64; n];
        if let Some(ptr) = a.data().buffer().ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr() as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    n * 8,
                );
            }
        }
        out
    }

    // --- zeros ---

    #[test]
    fn zeros_basic_f32() {
        let a = zeros(&[2, 3], None, Some(Device::Cpu)).unwrap();
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_eq!(a.shape(), &vec![2, 3]);
        assert_eq!(read_f32(&a), vec![0.0; 6]);
    }

    #[test]
    fn zeros_explicit_dtype() {
        let a = zeros(&[4], Some(Dtype::Int64), Some(Device::Cpu)).unwrap();
        assert_eq!(a.dtype(), Dtype::Int64);
        assert_eq!(read_i64(&a), vec![0; 4]);
    }

    // --- ones ---

    #[test]
    fn ones_basic_f32() {
        let a = ones(&[3], None, Some(Device::Cpu)).unwrap();
        assert_eq!(read_f32(&a), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn ones_f64() {
        let a = ones(&[2], Some(Dtype::Float64), Some(Device::Cpu)).unwrap();
        assert_eq!(a.dtype(), Dtype::Float64);
        assert_eq!(read_f64(&a), vec![1.0, 1.0]);
    }

    // --- full ---

    #[test]
    fn full_f32() {
        let a = full(&[3], 3.14, None, Some(Device::Cpu)).unwrap();
        let vals = read_f32(&a);
        for v in &vals {
            assert!((v - 3.14).abs() < 1e-5);
        }
    }

    // --- eye ---

    #[test]
    fn eye_3x3() {
        let a = eye(3, None, 0, None, Some(Device::Cpu)).unwrap();
        assert_eq!(a.shape(), &vec![3, 3]);
        assert_eq!(
            read_f32(&a),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn eye_rectangular() {
        let a = eye(2, Some(3), 0, None, Some(Device::Cpu)).unwrap();
        assert_eq!(a.shape(), &vec![2, 3]);
        assert_eq!(read_f32(&a), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn eye_offset_k1() {
        let a = eye(3, Some(3), 1, None, Some(Device::Cpu)).unwrap();
        assert_eq!(
            read_f32(&a),
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn eye_offset_k_neg1() {
        let a = eye(3, Some(3), -1, None, Some(Device::Cpu)).unwrap();
        assert_eq!(
            read_f32(&a),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
    }

    // --- arange ---

    #[test]
    fn arange_int_inference() {
        let a = arange(5.0, None, 1.0, None, Some(Device::Cpu)).unwrap();
        assert_eq!(a.dtype(), Dtype::Int64);
        assert_eq!(read_i64(&a), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn arange_float_inference() {
        let a = arange(0.0, Some(1.0), 0.25, None, Some(Device::Cpu)).unwrap();
        assert_eq!(a.dtype(), Dtype::Float64);
        let vals = read_f64(&a);
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[3] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn arange_explicit_dtype() {
        let a = arange(0.0, Some(4.0), 1.0, Some(Dtype::Float32), Some(Device::Cpu)).unwrap();
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_eq!(read_f32(&a), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn arange_negative_step() {
        let a = arange(5.0, Some(0.0), -1.0, None, Some(Device::Cpu)).unwrap();
        assert_eq!(read_i64(&a), vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn arange_empty() {
        let a = arange(5.0, Some(0.0), 1.0, None, Some(Device::Cpu)).unwrap();
        assert_eq!(a.size(), 0);
    }

    #[test]
    fn arange_step_zero_errors() {
        assert!(arange(0.0, Some(5.0), 0.0, None, Some(Device::Cpu)).is_err());
    }

    // --- linspace ---

    #[test]
    fn linspace_basic() {
        let a = linspace(0.0, 1.0, 5, None, Some(Device::Cpu)).unwrap();
        assert_eq!(a.dtype(), Dtype::Float64);
        let vals = read_f64(&a);
        assert_eq!(vals.len(), 5);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 0.25).abs() < 1e-10);
        assert!((vals[2] - 0.5).abs() < 1e-10);
        assert!((vals[3] - 0.75).abs() < 1e-10);
        assert!((vals[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn linspace_num_1() {
        let a = linspace(3.0, 10.0, 1, None, Some(Device::Cpu)).unwrap();
        assert_eq!(read_f64(&a), vec![3.0]);
    }

    #[test]
    fn linspace_num_0() {
        let a = linspace(0.0, 1.0, 0, None, Some(Device::Cpu)).unwrap();
        assert_eq!(a.size(), 0);
    }

    #[test]
    fn linspace_explicit_f32() {
        let a = linspace(0.0, 1.0, 3, Some(Dtype::Float32), Some(Device::Cpu)).unwrap();
        assert_eq!(a.dtype(), Dtype::Float32);
        let vals = read_f32(&a);
        assert!((vals[1] - 0.5).abs() < 1e-6);
    }

    // --- zeros_like / ones_like ---

    #[test]
    fn zeros_like_inherits() {
        let a = ones(&[2, 3], Some(Dtype::Float64), Some(Device::Cpu)).unwrap();
        let z = zeros_like(&a).unwrap();
        assert_eq!(z.dtype(), Dtype::Float64);
        assert_eq!(z.shape(), &vec![2, 3]);
        assert_eq!(read_f64(&z), vec![0.0; 6]);
    }

    #[test]
    fn ones_like_inherits() {
        let a = zeros(&[4], Some(Dtype::Int64), Some(Device::Cpu)).unwrap();
        let o = ones_like(&a).unwrap();
        assert_eq!(o.dtype(), Dtype::Int64);
        assert_eq!(read_i64(&o), vec![1, 1, 1, 1]);
    }
}
