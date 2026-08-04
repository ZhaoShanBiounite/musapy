//! 索引算子：transpose / permute / flip / slice（view）+ gather / scatter（copy）
//!
//! 设计原则（ADR 002-D4）：
//!   - view 操作（transpose/permute/flip/slice）零拷贝，仅修改 Layout，共享 BufferRef
//!   - copy 操作（gather/scatter/contiguous）分配新 buffer，走 GPU kernel 或 CPU fallback
//!   - GPU kernel dtype 实例化：f32/f64/i32/i64；其余 dtype 走 D2H→host→H2D fallback
//!   - indices 固定 int64 1D；GPU 路径越界由 device 侧错误标志报告
//!     （P1：kernel 跳过越界元素并置标志，Stream::synchronize 时报错），
//!     CPU 路径与 via-host fallback 仍在 host 端同步校验
//!   - 高级索引（boolean mask / fancy indexing）推迟到 v0.3+

use crate::kernels;
use crate::op_builder::{adjust_ptr_offset, cpu_offset_nd};
use musapy_core::error::{DtypeError, Result, ShapeError};
use musapy_core::musa_ffi;
use musapy_core::resolution;
use musapy_core::{
    Array, Buffer, BufferRef, Device, DeviceResolution, Dtype, DtypeResolution, Layout,
    OpContext, ResolutionSource, Stream,
};
use std::ptr::NonNull;
use std::sync::Arc;

// ── View 操作（零拷贝）────────────────────────────────────────

/// `transpose(a, axes=None)` — 转置（零拷贝视图）。
///
/// `axes=None` 时完全反转维度顺序（等价 `np.transpose(a)`）。
/// 返回新 Array，共享底层 buffer。
pub fn transpose(a: &Array, axes: Option<&[usize]>) -> Result<Array> {
    let new_layout = a.layout().transposed(axes)?;
    Ok(Array::new_view(a, new_layout))
}

/// `permute(a, dims)` — 按指定维度排列（零拷贝视图）。
///
/// 等价于 `transpose(a, axes=dims)`。
pub fn permute(a: &Array, dims: &[usize]) -> Result<Array> {
    transpose(a, Some(dims))
}

/// `flip(a, axis)` — 翻转指定轴（零拷贝视图）。
///
/// stride 取负，offset 调整到该轴末尾。
pub fn flip(a: &Array, axis: usize) -> Result<Array> {
    let new_layout = a.layout().flipped(axis)?;
    Ok(Array::new_view(a, new_layout))
}

// ── Slice 操作（零拷贝）───────────────────────────────────────

/// 切片参数（单维度）。
#[derive(Clone, Debug)]
pub struct SliceSpec {
    pub start: usize,
    pub stop: usize,
    pub step: usize,
}

/// `slice(a, specs)` — 多维切片（零拷贝视图）。
///
/// `specs` 长度必须等于 `a.ndim()`。每维 step >= 1。
pub fn slice(a: &Array, specs: &[SliceSpec]) -> Result<Array> {
    let ranges: Vec<(usize, usize, usize)> = specs
        .iter()
        .map(|s| (s.start, s.stop, s.step))
        .collect();
    let new_layout = a.layout().sliced(&ranges)?;
    Ok(Array::new_view(a, new_layout))
}

/// 整数索引：选择某一维的单个索引，降维（零拷贝视图）。
///
/// 例如 shape [3, 4] 的 `a[1]` → shape [4]，offset += 1 * strides[0]。
pub fn index_select(a: &Array, axis: usize, index: usize) -> Result<Array> {
    let ndim = a.ndim();
    if axis >= ndim {
        return Err(ShapeError::Mismatch(format!(
            "index axis {} out of bounds for ndim {}",
            axis, ndim
        ))
        .into());
    }
    if index >= a.shape()[axis] {
        return Err(ShapeError::Mismatch(format!(
            "index {} out of bounds for dimension {} (size {})",
            index,
            axis,
            a.shape()[axis]
        ))
        .into());
    }

    let layout = a.layout();
    // 新 offset = 原 offset + index * strides[axis]
    let new_offset =
        (layout.offset as isize + index as isize * layout.strides[axis]) as usize;

    // 去掉 axis 维的 shape 和 strides
    let mut new_shape = Vec::with_capacity(ndim - 1);
    let mut new_strides = Vec::with_capacity(ndim - 1);
    for i in 0..ndim {
        if i != axis {
            new_shape.push(layout.shape[i]);
            new_strides.push(layout.strides[i]);
        }
    }

    let new_layout = Layout {
        shape: new_shape,
        strides: new_strides,
        offset: new_offset,
    };
    Ok(Array::new_view(a, new_layout))
}

// ── Copy 操作（gather / scatter / contiguous）────────────────

/// `contiguous(a)` — 物化为连续布局（C order）。
///
/// 已连续（含 offset=0）时零拷贝返回共享视图；否则分配新 buffer
/// 逐元素拷贝（GPU kernel 或 CPU byte-gather）。
pub fn contiguous(a: &Array) -> Result<Array> {
    if a.is_contiguous() {
        return Ok(Array::new_view(a, a.layout().clone()));
    }

    let device = a.device().clone();
    let shape = a.shape().clone();
    let dtype = a.dtype();
    let n: usize = shape.iter().product::<usize>().max(1);
    let nbytes = n * dtype.element_size();

    let out_stream: Arc<Stream> = resolution::get_current_stream()
        .unwrap_or_else(|| Arc::clone(a.stream()));

    let buffer = Buffer::alloc(nbytes.max(1), device.clone(), &out_stream)?;
    let out_data_ref = BufferRef::new(Arc::new(buffer));
    let out_ptr = out_data_ref.buffer().ptr();

    a.data().buffer().wait_last_write_on(&out_stream)?;
    copy_into(a, out_ptr, &out_stream)?;

    a.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            "contiguous",
            vec![a.shape().clone()],
            vec![a.device().clone()],
            vec![dtype],
            shape.clone(),
            out_stream.id(),
        );
        if let Some(frame) = musapy_core::debug::take_debug_frame() {
            ctx = ctx.with_frame(frame);
        }
        out_stream.record_op(ctx);
    }

    Ok(Array::new(
        out_data_ref,
        Layout::from_shape(shape),
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

/// `gather(a, indices, axis)` — 沿 axis 按 indices 取元素（copy 语义）。
///
/// 等价 `np.take(a, indices, axis=axis)`：
/// 输出 shape = a.shape 中 axis 维替换为 indices.size()。
/// indices 必须为 1D int64，值域 [0, a.shape[axis])。
///
/// **越界检查（P1）**：GPU 路径不做同步 host 校验，越界索引由 device 侧
/// 错误标志报告——kernel 跳过越界元素的读/写，错误在下一次流同步
/// （如 `.tolist()`/`.item()`/显式 sync）时抛出。CPU 路径仍在调用时立即报错。
pub fn gather(a: &Array, indices: &Array, axis: usize) -> Result<Array> {
    // ── Phase A：校验 ──
    let ndim = a.ndim();
    if axis >= ndim {
        return Err(ShapeError::Mismatch(format!(
            "gather: axis {} out of bounds for ndim {}",
            axis, ndim
        ))
        .into());
    }
    if indices.ndim() != 1 {
        return Err(ShapeError::Mismatch(format!(
            "gather: indices must be 1D, got ndim {}",
            indices.ndim()
        ))
        .into());
    }
    if indices.dtype() != Dtype::Int64 {
        return Err(DtypeError::Unsupported(format!(
            "gather: indices dtype must be int64, got {}",
            indices.dtype()
        ))
        .into());
    }

    let n_indices = indices.size();
    let axis_len = a.shape()[axis];

    let device = a.device().clone();
    let dtype = a.dtype();
    let mut out_shape = a.shape().clone();
    out_shape[axis] = n_indices;
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let nbytes = n_out * dtype.element_size();

    // ── Phase B：分配 + kernel launch ──
    let out_stream: Arc<Stream> = resolution::get_current_stream()
        .unwrap_or_else(|| Arc::clone(a.stream()));

    let buffer = Buffer::alloc(nbytes.max(1), device.clone(), &out_stream)?;
    let out_data_ref = BufferRef::new(Arc::new(buffer));
    let out_ptr = out_data_ref.buffer().ptr();

    a.data().buffer().wait_last_write_on(&out_stream)?;
    indices.data().buffer().wait_last_write_on(&out_stream)?;

    let in_ptr = adjust_ptr_offset(
        a.data().buffer().ptr(),
        a.layout().offset,
        dtype.element_size(),
    );
    let in_strides = a.layout().strides.clone();

    // kernel 读取的 indices buffer（物化/上传后的实际来源，Phase C 记录读事件用）
    let idx_read_buffer: Arc<Buffer>;

    match &device {
        Device::Cpu => {
            // CPU 路径：host 端同步校验（立即报错）
            let idx_host = read_indices_host(indices)?;
            check_indices_bounds(&idx_host, axis_len, "gather", axis)?;
            cpu_gather_bytes(
                in_ptr,
                out_ptr,
                &idx_host,
                &out_shape,
                &in_strides,
                axis,
                dtype.element_size(),
            );
            idx_read_buffer = Arc::clone(indices.data().arc());
        }
        Device::Musa(_) => {
            // mock 构建保留同步 host 校验（mock 无 sync drain 机制）
            #[cfg(musapy_mock_musa)]
            {
                let idx_host = read_indices_host(indices)?;
                check_indices_bounds(&idx_host, axis_len, "gather", axis)?;
            }
            // kernel 要求 indices 连续：非连续视图先物化（GPU copy kernel，异步）
            let idx_contig_holder;
            let idx_src: &Array = if indices.is_contiguous() {
                indices
            } else {
                idx_contig_holder = contiguous(indices)?;
                &idx_contig_holder
            };
            // indices 需与 a 同设备：CPU indices 直接上传原始字节（不做 host 校验）
            let idx_holder;
            let idx_dev_src: &Array = if idx_src.device() == &device {
                idx_src.data().buffer().wait_last_write_on(&out_stream)?;
                idx_src
            } else {
                idx_holder = upload_indices_bytes(idx_src, &device, &out_stream)?;
                &idx_holder
            };
            let indices_dev = adjust_ptr_offset(
                idx_dev_src.data().buffer().ptr(),
                idx_dev_src.layout().offset,
                8,
            );
            idx_read_buffer = Arc::clone(idx_dev_src.data().arc());
            gpu_gather(
                a, idx_src, in_ptr, out_ptr, indices_dev, axis_len, &out_shape,
                &in_strides, axis, dtype, &out_stream,
            )?;
        }
    }

    // ── Phase C：事件记录 + 构造输出 ──
    a.data().buffer().record_read(&out_stream);
    idx_read_buffer.record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            "gather",
            vec![a.shape().clone(), indices.shape().clone()],
            vec![a.device().clone(), indices.device().clone()],
            vec![dtype, indices.dtype()],
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
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

/// `scatter(a, indices, values, axis)` — 沿 axis 把 values 写入 indices 指定位置（copy 语义）。
///
/// 返回新数组 = a 的连续副本，其中 `out[..., indices[j], ...] = values[..., j, ...]`。
/// 不修改原数组。values.shape 必须等于 a.shape 中 axis 维替换为 indices.size()。
/// 重复 indices 时写入顺序未定义（与 PyTorch 一致）。
///
/// **越界检查（P1）**：与 `gather` 相同——GPU 路径越界索引由 device 侧错误
/// 标志报告，错误在下一次流同步时抛出（越界元素被跳过写入）；CPU 路径立即报错。
pub fn scatter(a: &Array, indices: &Array, values: &Array, axis: usize) -> Result<Array> {
    // ── Phase A：校验 ──
    let ndim = a.ndim();
    if axis >= ndim {
        return Err(ShapeError::Mismatch(format!(
            "scatter: axis {} out of bounds for ndim {}",
            axis, ndim
        ))
        .into());
    }
    if indices.ndim() != 1 {
        return Err(ShapeError::Mismatch(format!(
            "scatter: indices must be 1D, got ndim {}",
            indices.ndim()
        ))
        .into());
    }
    if indices.dtype() != Dtype::Int64 {
        return Err(DtypeError::Unsupported(format!(
            "scatter: indices dtype must be int64, got {}",
            indices.dtype()
        ))
        .into());
    }
    if values.dtype() != a.dtype() {
        return Err(DtypeError::Unsupported(format!(
            "scatter: values dtype {} != input dtype {}",
            values.dtype(),
            a.dtype()
        ))
        .into());
    }
    if values.device() != a.device() {
        return Err(DtypeError::Unsupported(format!(
            "scatter: values device {:?} != input device {:?}",
            values.device(),
            a.device()
        ))
        .into());
    }
    if values.ndim() != ndim {
        return Err(ShapeError::Mismatch(format!(
            "scatter: values ndim {} != input ndim {}",
            values.ndim(),
            ndim
        ))
        .into());
    }

    let n_indices = indices.size();
    let axis_len = a.shape()[axis];

    // values shape 校验：= a.shape 且 axis 维 = n_indices
    let expected_val_shape: Vec<usize> = a
        .shape()
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == axis { n_indices } else { s })
        .collect();
    if values.shape() != &expected_val_shape[..] {
        return Err(ShapeError::Mismatch(format!(
            "scatter: values shape {:?} != expected {:?}",
            values.shape(),
            expected_val_shape
        ))
        .into());
    }

    let device = a.device().clone();
    let dtype = a.dtype();
    let out_shape = a.shape().clone();
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let nbytes = n_out * dtype.element_size();

    // ── Phase B：分配输出 = a 的连续副本，再 scatter 覆盖 ──
    let out_stream: Arc<Stream> = resolution::get_current_stream()
        .unwrap_or_else(|| Arc::clone(a.stream()));

    let buffer = Buffer::alloc(nbytes.max(1), device.clone(), &out_stream)?;
    let out_data_ref = BufferRef::new(Arc::new(buffer));
    let out_ptr = out_data_ref.buffer().ptr();

    a.data().buffer().wait_last_write_on(&out_stream)?;
    indices.data().buffer().wait_last_write_on(&out_stream)?;
    values.data().buffer().wait_last_write_on(&out_stream)?;

    // Phase B.1：a → output（物化拷贝，处理视图 offset/strides）
    copy_into(a, out_ptr, &out_stream)?;

    // Phase B.2：scatter values → output
    let val_ptr = adjust_ptr_offset(
        values.data().buffer().ptr(),
        values.layout().offset,
        dtype.element_size(),
    );
    let val_strides = values.layout().strides.clone();
    // output 连续布局的各维 stride（元素单位）
    let mut out_strides: Vec<usize> = vec![1; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // kernel 读取的 indices buffer（物化/上传后的实际来源，Phase C 记录读事件用）
    let idx_read_buffer: Arc<Buffer>;

    match &device {
        Device::Cpu => {
            // CPU 路径：host 端同步校验（立即报错）
            let idx_host = read_indices_host(indices)?;
            check_indices_bounds(&idx_host, axis_len, "scatter", axis)?;
            cpu_scatter_bytes(
                out_ptr,
                val_ptr,
                &idx_host,
                &expected_val_shape,
                &val_strides,
                &out_strides,
                axis,
                dtype.element_size(),
            );
            idx_read_buffer = Arc::clone(indices.data().arc());
        }
        Device::Musa(_) => {
            // mock 构建保留同步 host 校验（mock 无 sync drain 机制）
            #[cfg(musapy_mock_musa)]
            {
                let idx_host = read_indices_host(indices)?;
                check_indices_bounds(&idx_host, axis_len, "scatter", axis)?;
            }
            // kernel 要求 indices 连续：非连续视图先物化（GPU copy kernel，异步）
            let idx_contig_holder;
            let idx_src: &Array = if indices.is_contiguous() {
                indices
            } else {
                idx_contig_holder = contiguous(indices)?;
                &idx_contig_holder
            };
            // indices 需与 a 同设备：CPU indices 直接上传原始字节（不做 host 校验）
            let idx_holder;
            let idx_dev_src: &Array = if idx_src.device() == &device {
                idx_src.data().buffer().wait_last_write_on(&out_stream)?;
                idx_src
            } else {
                idx_holder = upload_indices_bytes(idx_src, &device, &out_stream)?;
                &idx_holder
            };
            let indices_dev = adjust_ptr_offset(
                idx_dev_src.data().buffer().ptr(),
                idx_dev_src.layout().offset,
                8,
            );
            idx_read_buffer = Arc::clone(idx_dev_src.data().arc());
            gpu_scatter(
                out_ptr, values, val_ptr, indices_dev, idx_src, axis_len,
                &expected_val_shape, &val_strides, &out_strides, axis, dtype, n_out,
                &out_stream,
            )?;
        }
    }

    // ── Phase C：事件记录 + 构造输出 ──
    a.data().buffer().record_read(&out_stream);
    idx_read_buffer.record_read(&out_stream);
    values.data().buffer().record_read(&out_stream);
    out_data_ref.buffer().record_write(&out_stream);

    if musapy_core::debug::is_debug() {
        let mut ctx = OpContext::new(
            "scatter",
            vec![a.shape().clone(), indices.shape().clone(), values.shape().clone()],
            vec![a.device().clone(), indices.device().clone(), values.device().clone()],
            vec![dtype, indices.dtype(), values.dtype()],
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
        dtype,
        out_stream,
        DeviceResolution::new(device, ResolutionSource::InputArray),
        DtypeResolution::new(dtype, ResolutionSource::InputArray),
    ))
}

// ── 内部助手 ─────────────────────────────────────────────────

/// 逐元素拷贝 `a` → 连续目标指针（处理 offset + 任意 strides）。
fn copy_into(a: &Array, out_ptr: Option<NonNull<u8>>, out_stream: &Arc<Stream>) -> Result<()> {
    let dtype = a.dtype();
    let shape = a.shape();
    let n: usize = shape.iter().product::<usize>().max(1);
    let elem_size = dtype.element_size();
    let in_ptr = adjust_ptr_offset(a.data().buffer().ptr(), a.layout().offset, elem_size);
    let strides = a.layout().strides.clone();

    match a.device() {
        Device::Cpu => {
            cpu_copy_bytes(in_ptr, out_ptr, shape, &strides, n, elem_size);
            Ok(())
        }
        Device::Musa(_) => {
            let stream_raw = out_stream.raw();
            let ndim = shape.len() as i32;
            // P4：2D 转置模式检测（strides == [1, rows] ⇔ 连续数组的转置视图）
            // → tiled smem kernel（读/写两侧均合并）；其余走通用 copy kernel
            let transpose2d = shape.len() == 2
                && strides[0] == 1
                && strides[1] == shape[0] as isize;
            let launched = unsafe {
                match dtype {
                    Dtype::Float32 => {
                        if let (Some(ip), Some(op)) = (in_ptr, out_ptr) {
                            if transpose2d {
                                kernels::musapy_copy_transpose2d_f32(
                                    ip.as_ptr() as *const f32, op.as_ptr() as *mut f32,
                                    shape[0], shape[1], stream_raw,
                                );
                            } else {
                                kernels::musapy_copy_f32(
                                    ip.as_ptr() as *const f32, op.as_ptr() as *mut f32,
                                    ndim, shape.as_ptr(), strides.as_ptr(), stream_raw,
                                );
                            }
                        }
                        true
                    }
                    Dtype::Float64 => {
                        if let (Some(ip), Some(op)) = (in_ptr, out_ptr) {
                            if transpose2d {
                                kernels::musapy_copy_transpose2d_f64(
                                    ip.as_ptr() as *const f64, op.as_ptr() as *mut f64,
                                    shape[0], shape[1], stream_raw,
                                );
                            } else {
                                kernels::musapy_copy_f64(
                                    ip.as_ptr() as *const f64, op.as_ptr() as *mut f64,
                                    ndim, shape.as_ptr(), strides.as_ptr(), stream_raw,
                                );
                            }
                        }
                        true
                    }
                    Dtype::Int32 => {
                        if let (Some(ip), Some(op)) = (in_ptr, out_ptr) {
                            if transpose2d {
                                kernels::musapy_copy_transpose2d_i32(
                                    ip.as_ptr() as *const i32, op.as_ptr() as *mut i32,
                                    shape[0], shape[1], stream_raw,
                                );
                            } else {
                                kernels::musapy_copy_i32(
                                    ip.as_ptr() as *const i32, op.as_ptr() as *mut i32,
                                    ndim, shape.as_ptr(), strides.as_ptr(), stream_raw,
                                );
                            }
                        }
                        true
                    }
                    Dtype::Int64 => {
                        if let (Some(ip), Some(op)) = (in_ptr, out_ptr) {
                            if transpose2d {
                                kernels::musapy_copy_transpose2d_i64(
                                    ip.as_ptr() as *const i64, op.as_ptr() as *mut i64,
                                    shape[0], shape[1], stream_raw,
                                );
                            } else {
                                kernels::musapy_copy_i64(
                                    ip.as_ptr() as *const i64, op.as_ptr() as *mut i64,
                                    ndim, shape.as_ptr(), strides.as_ptr(), stream_raw,
                                );
                            }
                        }
                        true
                    }
                    _ => false,
                }
            };
            if launched {
                musa_ffi::check_last_kernel_launch("copy_transpose2d")?;
            } else {
                // 未实例化 dtype：D2H 整个 buffer → host gather → H2D
                gpu_copy_via_host(a, out_ptr, n, elem_size)?;
            }
            Ok(())
        }
    }
}

/// GPU fallback：D2H 整个源 buffer → host 端 gather → H2D 到目标。
fn gpu_copy_via_host(
    a: &Array,
    out_ptr: Option<NonNull<u8>>,
    n: usize,
    elem_size: usize,
) -> Result<()> {
    let Some(op) = out_ptr else { return Ok(()) };
    let buf_size = a.data().buffer().size();
    let mut host_buf = vec![0u8; buf_size];
    a.stream().synchronize()?;
    if let Some(base) = a.data().buffer().ptr() {
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaMemcpy(
                    host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                    base.as_ptr() as *const std::ffi::c_void,
                    buf_size,
                    musa_ffi::musaMemcpyKind::DeviceToHost,
                ),
                "musaMemcpy(D2H indexing fallback)",
            )?;
        }
    }
    // host 端 gather（host_buf 基址 = buffer 起始；offset/strides 相对它计算）
    let mut gathered = vec![0u8; n * elem_size];
    let shape = a.shape();
    let strides = &a.layout().strides;
    let base_off = a.layout().offset as isize;
    for idx in 0..n {
        let off = (base_off + cpu_offset_nd(idx, shape, strides)) as usize;
        let src = &host_buf[off * elem_size..(off + 1) * elem_size];
        gathered[idx * elem_size..(idx + 1) * elem_size].copy_from_slice(src);
    }
    unsafe {
        musa_ffi::check_musa(
            musa_ffi::musaMemcpy(
                op.as_ptr() as *mut std::ffi::c_void,
                gathered.as_ptr() as *const std::ffi::c_void,
                n * elem_size,
                musa_ffi::musaMemcpyKind::HostToDevice,
            ),
            "musaMemcpy(H2D indexing fallback)",
        )?;
    }
    Ok(())
}

/// GPU gather：dtype 已实例化则 launch kernel，否则 D2H→host gather→H2D。
///
/// `indices_dev` 是已就位于目标设备的 indices int64 指针（调用方保证设备匹配）。
#[allow(clippy::too_many_arguments)]
fn gpu_gather(
    a: &Array,
    indices_src: &Array,
    in_ptr: Option<NonNull<u8>>,
    out_ptr: Option<NonNull<u8>>,
    indices_dev: Option<NonNull<u8>>,
    axis_len: usize,
    out_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    dtype: Dtype,
    out_stream: &Arc<Stream>,
) -> Result<()> {
    let (Some(ip), Some(op)) = (in_ptr, out_ptr) else {
        return Ok(());
    };
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let stream_raw = out_stream.raw();
    let ndim = out_shape.len() as i32;
    let axis_i32 = axis as i32;

    let idx_dev = indices_dev;
    let n_indices = out_shape[axis];

    // dtype 已实例化 → launch v2 kernel（device 侧越界检查，P1 方案二）
    let instantiated = matches!(
        (dtype, idx_dev),
        (Dtype::Float32 | Dtype::Float64 | Dtype::Int32 | Dtype::Int64, Some(_))
    );

    if instantiated {
        let idp = idx_dev.unwrap();
        // 错误槽 16B：[flag i32][pos i32][val i64]，synchronize 时批量读回
        #[cfg(not(musapy_mock_musa))]
        let (err_flag, err_pos, err_val) = {
            let slot = out_stream.acquire_index_check(format!(
                "gather(axis={}, axis_len={}, n_indices={})",
                axis, axis_len, n_indices
            ))?;
            let p = slot.as_ptr();
            unsafe { (p as *mut i32, (p as *mut i32).add(1), (p as *mut i64).add(1)) }
        };
        #[cfg(musapy_mock_musa)]
        let (err_flag, err_pos, err_val): (*mut i32, *mut i32, *mut i64) =
            (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());

        unsafe {
            match dtype {
                Dtype::Float32 => kernels::musapy_gather_f32_v2(
                    ip.as_ptr() as *const f32, op.as_ptr() as *mut f32,
                    idp.as_ptr() as *const i64, ndim, axis_i32,
                    out_shape.as_ptr(), in_strides.as_ptr(), n_out, axis_len,
                    err_flag, err_pos, err_val, stream_raw,
                ),
                Dtype::Float64 => kernels::musapy_gather_f64_v2(
                    ip.as_ptr() as *const f64, op.as_ptr() as *mut f64,
                    idp.as_ptr() as *const i64, ndim, axis_i32,
                    out_shape.as_ptr(), in_strides.as_ptr(), n_out, axis_len,
                    err_flag, err_pos, err_val, stream_raw,
                ),
                Dtype::Int32 => kernels::musapy_gather_i32_v2(
                    ip.as_ptr() as *const i32, op.as_ptr() as *mut i32,
                    idp.as_ptr() as *const i64, ndim, axis_i32,
                    out_shape.as_ptr(), in_strides.as_ptr(), n_out, axis_len,
                    err_flag, err_pos, err_val, stream_raw,
                ),
                Dtype::Int64 => kernels::musapy_gather_i64_v2(
                    ip.as_ptr() as *const i64, op.as_ptr() as *mut i64,
                    idp.as_ptr() as *const i64, ndim, axis_i32,
                    out_shape.as_ptr(), in_strides.as_ptr(), n_out, axis_len,
                    err_flag, err_pos, err_val, stream_raw,
                ),
                _ => unreachable!("guarded by instantiated check"),
            }
        }
        musa_ffi::check_last_kernel_launch("gather_v2")?;
        Ok(())
    } else {
        // fallback：D2H 整个源 buffer → host gather → H2D
        // （未实例化 dtype；保留 host 端同步校验与立即报错语义）
        let idx_host = read_indices_host(indices_src)?;
        check_indices_bounds(&idx_host, axis_len, "gather", axis)?;
        gpu_gather_via_host(a, out_ptr, &idx_host, out_shape, in_strides, axis, dtype.element_size())
    }
}

/// GPU gather fallback：host 端完成 gather。
fn gpu_gather_via_host(
    a: &Array,
    out_ptr: Option<NonNull<u8>>,
    idx_host: &[i64],
    out_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    elem_size: usize,
) -> Result<()> {
    let Some(op) = out_ptr else { return Ok(()) };
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let buf_size = a.data().buffer().size();
    let mut host_buf = vec![0u8; buf_size];
    a.stream().synchronize()?;
    if let Some(base) = a.data().buffer().ptr() {
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaMemcpy(
                    host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                    base.as_ptr() as *const std::ffi::c_void,
                    buf_size,
                    musa_ffi::musaMemcpyKind::DeviceToHost,
                ),
                "musaMemcpy(D2H gather fallback)",
            )?;
        }
    }
    let base_off = a.layout().offset;
    let mut gathered = vec![0u8; n_out * elem_size];
    for idx in 0..n_out {
        let mut tmp = idx;
        let mut off = base_off as isize;
        for i in (0..out_shape.len()).rev() {
            let coord = tmp % out_shape[i];
            tmp /= out_shape[i];
            let k = if i == axis { idx_host[coord] as usize } else { coord };
            off += k as isize * in_strides[i];
        }
        let off = off as usize;
        gathered[idx * elem_size..(idx + 1) * elem_size]
            .copy_from_slice(&host_buf[off * elem_size..(off + 1) * elem_size]);
    }
    unsafe {
        musa_ffi::check_musa(
            musa_ffi::musaMemcpy(
                op.as_ptr() as *mut std::ffi::c_void,
                gathered.as_ptr() as *const std::ffi::c_void,
                n_out * elem_size,
                musa_ffi::musaMemcpyKind::HostToDevice,
            ),
            "musaMemcpy(H2D gather fallback)",
        )?;
    }
    Ok(())
}

/// GPU scatter：dtype 已实例化则 launch kernel，否则 D2H→host scatter→H2D。
///
/// `indices_dev` 是已就位于目标设备的 indices int64 指针；
/// `values` 用于 fallback 时读回 values 内容；`n_out_total` 为 output 总元素数。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn gpu_scatter(
    out_ptr: Option<NonNull<u8>>,
    values: &Array,
    val_ptr: Option<NonNull<u8>>,
    indices_dev: Option<NonNull<u8>>,
    indices_src: &Array,
    axis_len: usize,
    val_shape: &[usize],
    val_strides: &[isize],
    out_strides: &[usize],
    axis: usize,
    dtype: Dtype,
    n_out_total: usize,
    out_stream: &Arc<Stream>,
) -> Result<()> {
    let (Some(op), Some(vp)) = (out_ptr, val_ptr) else {
        return Ok(());
    };
    let n_values: usize = val_shape.iter().product::<usize>().max(1);
    let stream_raw = out_stream.raw();
    let ndim = val_shape.len() as i32;
    let axis_i32 = axis as i32;

    let n_indices = val_shape[axis];
    let instantiated = matches!(
        (dtype, indices_dev),
        (Dtype::Float32 | Dtype::Float64 | Dtype::Int32 | Dtype::Int64, Some(_))
    );

    if instantiated {
        let idp = indices_dev.unwrap();
        // 错误槽 16B：[flag i32][pos i32][val i64]，synchronize 时批量读回
        #[cfg(not(musapy_mock_musa))]
        let (err_flag, err_pos, err_val) = {
            let slot = out_stream.acquire_index_check(format!(
                "scatter(axis={}, axis_len={}, n_indices={})",
                axis, axis_len, n_indices
            ))?;
            let p = slot.as_ptr();
            unsafe { (p as *mut i32, (p as *mut i32).add(1), (p as *mut i64).add(1)) }
        };
        #[cfg(musapy_mock_musa)]
        let (err_flag, err_pos, err_val): (*mut i32, *mut i32, *mut i64) =
            (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());

        unsafe {
            match dtype {
                Dtype::Float32 => kernels::musapy_scatter_f32_v2(
                    op.as_ptr() as *mut f32, vp.as_ptr() as *const f32,
                    idp.as_ptr() as *const i64, ndim, axis_i32,
                    val_shape.as_ptr(), val_strides.as_ptr(), out_strides.as_ptr(),
                    n_values, axis_len, err_flag, err_pos, err_val, stream_raw,
                ),
                Dtype::Float64 => kernels::musapy_scatter_f64_v2(
                    op.as_ptr() as *mut f64, vp.as_ptr() as *const f64,
                    idp.as_ptr() as *const i64, ndim, axis_i32,
                    val_shape.as_ptr(), val_strides.as_ptr(), out_strides.as_ptr(),
                    n_values, axis_len, err_flag, err_pos, err_val, stream_raw,
                ),
                Dtype::Int32 => kernels::musapy_scatter_i32_v2(
                    op.as_ptr() as *mut i32, vp.as_ptr() as *const i32,
                    idp.as_ptr() as *const i64, ndim, axis_i32,
                    val_shape.as_ptr(), val_strides.as_ptr(), out_strides.as_ptr(),
                    n_values, axis_len, err_flag, err_pos, err_val, stream_raw,
                ),
                Dtype::Int64 => kernels::musapy_scatter_i64_v2(
                    op.as_ptr() as *mut i64, vp.as_ptr() as *const i64,
                    idp.as_ptr() as *const i64, ndim, axis_i32,
                    val_shape.as_ptr(), val_strides.as_ptr(), out_strides.as_ptr(),
                    n_values, axis_len, err_flag, err_pos, err_val, stream_raw,
                ),
                _ => unreachable!("guarded by instantiated check"),
            }
        }
        musa_ffi::check_last_kernel_launch("scatter_v2")?;
        Ok(())
    } else {
        // fallback 用同步 memcpy 读回 output（copy_into 刚在 out_stream 上写过）；
        // 未实例化 dtype 保留 host 端同步校验与立即报错语义
        out_stream.synchronize()?;
        let idx_host = read_indices_host(indices_src)?;
        check_indices_bounds(&idx_host, axis_len, "scatter", axis)?;
        gpu_scatter_via_host(
            out_ptr, values, &idx_host, val_shape, val_strides, out_strides, axis,
            dtype.element_size(), n_out_total,
        )
    }
}

/// GPU scatter fallback：D2H output（已含 a 副本）+ values → host scatter → H2D output。
#[allow(clippy::too_many_arguments)]
fn gpu_scatter_via_host(
    out_ptr: Option<NonNull<u8>>,
    values: &Array,
    idx_host: &[i64],
    val_shape: &[usize],
    val_strides: &[isize],
    out_strides: &[usize],
    axis: usize,
    elem_size: usize,
    n_out_total: usize,
) -> Result<()> {
    let Some(op) = out_ptr else { return Ok(()) };
    let n_values: usize = val_shape.iter().product::<usize>().max(1);

    // 1. D2H output（scatter 前的 a 连续副本）
    let mut out_host = vec![0u8; n_out_total * elem_size];
    unsafe {
        musa_ffi::check_musa(
            musa_ffi::musaMemcpy(
                out_host.as_mut_ptr() as *mut std::ffi::c_void,
                op.as_ptr() as *const std::ffi::c_void,
                n_out_total * elem_size,
                musa_ffi::musaMemcpyKind::DeviceToHost,
            ),
            "musaMemcpy(D2H scatter output)",
        )?;
    }

    // 2. D2H values 整个 buffer（host 端按 offset/strides gather）
    let val_buf_size = values.data().buffer().size();
    let mut val_host = vec![0u8; val_buf_size];
    values.stream().synchronize()?;
    if let Some(vbase) = values.data().buffer().ptr() {
        unsafe {
            musa_ffi::check_musa(
                musa_ffi::musaMemcpy(
                    val_host.as_mut_ptr() as *mut std::ffi::c_void,
                    vbase.as_ptr() as *const std::ffi::c_void,
                    val_buf_size,
                    musa_ffi::musaMemcpyKind::DeviceToHost,
                ),
                "musaMemcpy(D2H scatter values)",
            )?;
        }
    }

    // 3. host scatter
    let val_base_off = values.layout().offset;
    for idx in 0..n_values {
        let mut tmp = idx;
        let mut out_off = 0usize;
        let mut val_off = val_base_off as isize;
        for i in (0..val_shape.len()).rev() {
            let coord = tmp % val_shape[i];
            tmp /= val_shape[i];
            val_off += coord as isize * val_strides[i];
            let k = if i == axis { idx_host[coord] as usize } else { coord };
            out_off += k * out_strides[i];
        }
        let vo = val_off as usize * elem_size;
        out_host[out_off * elem_size..(out_off + 1) * elem_size]
            .copy_from_slice(&val_host[vo..vo + elem_size]);
    }

    // 4. H2D output
    unsafe {
        musa_ffi::check_musa(
            musa_ffi::musaMemcpy(
                op.as_ptr() as *mut std::ffi::c_void,
                out_host.as_ptr() as *const std::ffi::c_void,
                n_out_total * elem_size,
                musa_ffi::musaMemcpyKind::HostToDevice,
            ),
            "musaMemcpy(H2D scatter output)",
        )?;
    }
    Ok(())
}

/// 将 CPU 端连续 int64 indices 原样上传到指定设备（H2D 字节拷贝，
/// 不做 host 校验——P1 起越界由 device 侧错误标志报告）。
fn upload_indices_bytes(indices: &Array, device: &Device, stream: &Arc<Stream>) -> Result<Array> {
    let n = indices.size();
    let nbytes = (n * 8).max(1);
    let buffer = Buffer::alloc(nbytes, device.clone(), stream)?;
    let data_ref = BufferRef::new(Arc::new(buffer));
    if n > 0 {
        let src = adjust_ptr_offset(indices.data().buffer().ptr(), indices.layout().offset, 8);
        if let (Some(p), Some(sp)) = (data_ref.buffer().ptr(), src) {
            unsafe {
                musa_ffi::check_musa(
                    musa_ffi::musaMemcpy(
                        p.as_ptr() as *mut std::ffi::c_void,
                        sp.as_ptr() as *const std::ffi::c_void,
                        n * 8,
                        musa_ffi::musaMemcpyKind::HostToDevice,
                    ),
                    "musaMemcpy(H2D indices)",
                )?;
            }
        }
    }
    data_ref.buffer().record_write(stream);
    Ok(Array::new(
        data_ref,
        Layout::from_shape(vec![n]),
        Dtype::Int64,
        Arc::clone(stream),
        DeviceResolution::new(device.clone(), ResolutionSource::Arg),
        DtypeResolution::new(Dtype::Int64, ResolutionSource::Arg),
    ))
}

/// host 端索引越界校验（CPU 路径、via-host fallback、mock 构建使用）。
fn check_indices_bounds(idx_host: &[i64], axis_len: usize, op: &str, axis: usize) -> Result<()> {
    for (i, &k) in idx_host.iter().enumerate() {
        if k < 0 || k as usize >= axis_len {
            return Err(ShapeError::Mismatch(format!(
                "{}: index {} at position {} out of bounds for axis {} (size {})",
                op, k, i, axis, axis_len
            ))
            .into());
        }
    }
    Ok(())
}

/// 读取 indices 数组内容到 host（int64 1D；GPU 时同步 + D2H）。
fn read_indices_host(indices: &Array) -> Result<Vec<i64>> {
    let n = indices.size();
    if n == 0 {
        return Ok(Vec::new());
    }
    // 非连续时先物化（int64 在 GPU 有 copy kernel，CPU 直接 byte-gather）
    let holder;
    let src: &Array = if indices.is_contiguous() {
        indices
    } else {
        holder = contiguous(indices)?;
        &holder
    };

    let nbytes = n * 8;
    let mut bytes = vec![0u8; nbytes];
    match src.device() {
        Device::Cpu => {
            if let Some(p) = src.data().buffer().ptr() {
                unsafe {
                    std::ptr::copy_nonoverlapping(p.as_ptr(), bytes.as_mut_ptr(), nbytes);
                }
            }
        }
        Device::Musa(_) => {
            src.stream().synchronize()?;
            if let Some(p) = src.data().buffer().ptr() {
                unsafe {
                    musa_ffi::check_musa(
                        musa_ffi::musaMemcpy(
                            bytes.as_mut_ptr() as *mut std::ffi::c_void,
                            p.as_ptr() as *const std::ffi::c_void,
                            nbytes,
                            musa_ffi::musaMemcpyKind::DeviceToHost,
                        ),
                        "musaMemcpy(D2H indices)",
                    )?;
                }
            }
        }
    }
    // bytes → Vec<i64>（小端；MUSA 平台均为小端）
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b: [u8; 8] = bytes[i * 8..(i + 1) * 8].try_into().unwrap();
        out.push(i64::from_le_bytes(b));
    }
    Ok(out)
}

/// CPU byte-level copy：逐元素 gather 到连续目标（任意 dtype）。
fn cpu_copy_bytes(
    in_ptr: Option<NonNull<u8>>,
    out_ptr: Option<NonNull<u8>>,
    shape: &[usize],
    strides: &[isize],
    n: usize,
    elem_size: usize,
) {
    let (Some(ip), Some(op)) = (in_ptr, out_ptr) else {
        return;
    };
    unsafe {
        for idx in 0..n {
            let off = cpu_offset_nd(idx, shape, strides);
            std::ptr::copy_nonoverlapping(
                ip.as_ptr().offset(off * elem_size as isize),
                op.as_ptr().add(idx * elem_size),
                elem_size,
            );
        }
    }
}

/// CPU byte-level gather（任意 dtype）。
fn cpu_gather_bytes(
    in_ptr: Option<NonNull<u8>>,
    out_ptr: Option<NonNull<u8>>,
    indices: &[i64],
    out_shape: &[usize],
    in_strides: &[isize],
    axis: usize,
    elem_size: usize,
) {
    let (Some(ip), Some(op)) = (in_ptr, out_ptr) else {
        return;
    };
    let n_out: usize = out_shape.iter().product();
    unsafe {
        for idx in 0..n_out {
            let mut tmp = idx;
            let mut off = 0isize;
            for i in (0..out_shape.len()).rev() {
                let coord = tmp % out_shape[i];
                tmp /= out_shape[i];
                let k = if i == axis { indices[coord] as usize } else { coord };
                off += k as isize * in_strides[i];
            }
            std::ptr::copy_nonoverlapping(
                ip.as_ptr().offset(off * elem_size as isize),
                op.as_ptr().add(idx * elem_size),
                elem_size,
            );
        }
    }
}

/// CPU byte-level scatter（任意 dtype）。
#[allow(clippy::too_many_arguments)]
fn cpu_scatter_bytes(
    out_ptr: Option<NonNull<u8>>,
    val_ptr: Option<NonNull<u8>>,
    indices: &[i64],
    val_shape: &[usize],
    val_strides: &[isize],
    out_strides: &[usize],
    axis: usize,
    elem_size: usize,
) {
    let (Some(op), Some(vp)) = (out_ptr, val_ptr) else {
        return;
    };
    let n_values: usize = val_shape.iter().product();
    unsafe {
        for idx in 0..n_values {
            let mut tmp = idx;
            let mut out_off = 0usize;
            let mut val_off = 0isize;
            for i in (0..val_shape.len()).rev() {
                let coord = tmp % val_shape[i];
                tmp /= val_shape[i];
                val_off += coord as isize * val_strides[i];
                let k = if i == axis { indices[coord] as usize } else { coord };
                out_off += k * out_strides[i];
            }
            std::ptr::copy_nonoverlapping(
                vp.as_ptr().offset(val_off * elem_size as isize),
                op.as_ptr().add(out_off * elem_size),
                elem_size,
            );
        }
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::zeros;
    use musapy_core::Dtype;
    use std::sync::Arc;

    /// 创建测试用 Array（CPU，f32）。
    fn make_array(shape: Vec<usize>) -> Array {
        zeros(&shape, Some(Dtype::Float32), Some(musapy_core::Device::Cpu)).unwrap()
    }

    // --- 带值数组辅助（gather/scatter/contiguous 需要验证内容）---

    fn cpu_array_with_layout(bytes: &[u8], layout: Layout, dtype: Dtype) -> Array {
        let device = musapy_core::Device::Cpu;
        let stream = Arc::new(Stream::new(device.clone(), 0).unwrap());
        let buffer = Buffer::alloc(bytes.len().max(1), device.clone(), &stream).unwrap();
        let buf_arc = Arc::new(buffer);
        let data_ref = BufferRef::new(buf_arc);
        if let Some(ptr) = data_ref.buffer().ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len());
            }
        }
        Array::new(
            data_ref,
            layout,
            dtype,
            stream,
            DeviceResolution::new(device, ResolutionSource::Arg),
            DtypeResolution::new(dtype, ResolutionSource::Arg),
        )
    }

    fn f32_array(vals: &[f32], shape: Vec<usize>) -> Array {
        let bytes = unsafe {
            std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4)
        };
        cpu_array_with_layout(bytes, Layout::from_shape(shape), Dtype::Float32)
    }

    fn i64_array(vals: &[i64], shape: Vec<usize>) -> Array {
        let bytes = unsafe {
            std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 8)
        };
        cpu_array_with_layout(bytes, Layout::from_shape(shape), Dtype::Int64)
    }

    /// 读回连续 f32 数组内容（gather/scatter/contiguous 输出均为连续布局）。
    fn read_f32(a: &Array) -> Vec<f32> {
        let n = a.size();
        let mut out = vec![0f32; n];
        if let Some(ptr) = a.data().buffer().ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    out.as_mut_ptr() as *mut u8,
                    n * 4,
                );
            }
        }
        out
    }

    // --- transpose ---

    #[test]
    fn transpose_2d_default() {
        let a = make_array(vec![2, 3]);
        let t = transpose(&a, None).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.layout().strides, vec![1, 3]);
        // 零拷贝：共享 buffer
        assert!(Arc::ptr_eq(a.data().arc(), t.data().arc()));
    }

    #[test]
    fn transpose_with_axes() {
        let a = make_array(vec![2, 3, 4]);
        let t = transpose(&a, Some(&[1, 0, 2])).unwrap();
        assert_eq!(t.shape(), &[3, 2, 4]);
        assert_eq!(t.layout().strides, vec![4, 12, 1]);
    }

    #[test]
    fn transpose_invalid_axes() {
        let a = make_array(vec![2, 3]);
        assert!(transpose(&a, Some(&[0, 0])).is_err());
        assert!(transpose(&a, Some(&[0, 2])).is_err());
    }

    // --- permute ---

    #[test]
    fn permute_basic() {
        let a = make_array(vec![2, 3, 4]);
        let p = permute(&a, &[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.layout().strides, vec![1, 12, 4]);
    }

    // --- flip ---

    #[test]
    fn flip_axis0() {
        let a = make_array(vec![3, 4]);
        let f = flip(&a, 0).unwrap();
        assert_eq!(f.shape(), &[3, 4]);
        assert_eq!(f.layout().strides, vec![-4, 1]);
        assert_eq!(f.layout().offset, 8);
    }

    #[test]
    fn flip_axis1() {
        let a = make_array(vec![3, 4]);
        let f = flip(&a, 1).unwrap();
        assert_eq!(f.shape(), &[3, 4]);
        assert_eq!(f.layout().strides, vec![4, -1]);
        assert_eq!(f.layout().offset, 3);
    }

    #[test]
    fn flip_out_of_bounds() {
        let a = make_array(vec![3, 4]);
        assert!(flip(&a, 2).is_err());
    }

    // --- slice ---

    #[test]
    fn slice_1d() {
        let a = make_array(vec![10]);
        let s = slice(&a, &[SliceSpec { start: 2, stop: 7, step: 1 }]).unwrap();
        assert_eq!(s.shape(), &[5]);
        assert_eq!(s.layout().offset, 2);
    }

    #[test]
    fn slice_2d_with_step() {
        let a = make_array(vec![4, 6]);
        let s = slice(
            &a,
            &[
                SliceSpec { start: 1, stop: 3, step: 1 },
                SliceSpec { start: 0, stop: 6, step: 2 },
            ],
        )
        .unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        assert_eq!(s.layout().strides, vec![6, 2]);
        assert_eq!(s.layout().offset, 6);
    }

    // --- index_select ---

    #[test]
    fn index_select_axis0() {
        let a = make_array(vec![3, 4]);
        let s = index_select(&a, 0, 1).unwrap();
        assert_eq!(s.shape(), &[4]);
        assert_eq!(s.layout().offset, 4); // 1 * strides[0]=4
    }

    #[test]
    fn index_select_axis1() {
        let a = make_array(vec![3, 4]);
        let s = index_select(&a, 1, 2).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.layout().strides, vec![4]);
        assert_eq!(s.layout().offset, 2); // 2 * strides[1]=1
    }

    #[test]
    fn index_select_out_of_bounds() {
        let a = make_array(vec![3, 4]);
        assert!(index_select(&a, 0, 3).is_err());
        assert!(index_select(&a, 1, 4).is_err());
        assert!(index_select(&a, 2, 0).is_err());
    }

    // --- contiguous ---

    #[test]
    fn contiguous_already_contiguous_is_zero_copy() {
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = contiguous(&a).unwrap();
        // 已连续 → 共享 buffer 的视图
        assert!(Arc::ptr_eq(a.data().arc(), c.data().arc()));
        assert_eq!(read_f32(&c), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn contiguous_transposed_view_materializes() {
        // [[1,2,3],[4,5,6]] 转置后逻辑值 [[1,4],[2,5],[3,6]]
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t = transpose(&a, None).unwrap();
        let c = contiguous(&t).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert!(c.layout().is_contiguous());
        assert!(!Arc::ptr_eq(a.data().arc(), c.data().arc()));
        assert_eq!(read_f32(&c), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn contiguous_offset_slice_materializes() {
        // slice 产生 offset 视图：[10..20][2:5] = [12,13,14]
        let vals: Vec<f32> = (10..20).map(|v| v as f32).collect();
        let a = f32_array(&vals, vec![10]);
        let s = slice(&a, &[SliceSpec { start: 2, stop: 5, step: 1 }]).unwrap();
        assert_eq!(s.layout().offset, 2);
        let c = contiguous(&s).unwrap();
        assert!(c.layout().is_contiguous());
        assert_eq!(read_f32(&c), vec![12.0, 13.0, 14.0]);
    }

    #[test]
    fn contiguous_flipped_view_materializes() {
        // flip(axis=1) 产生负 stride + offset：[[1,2,3],[4,5,6]] → [[3,2,1],[6,5,4]]
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let f = flip(&a, 1).unwrap();
        let c = contiguous(&f).unwrap();
        assert!(c.layout().is_contiguous());
        assert_eq!(read_f32(&c), vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    // --- gather ---

    #[test]
    fn gather_2d_axis0() {
        // [[1,2,3],[4,5,6]]，indices=[1,0] → [[4,5,6],[1,2,3]]
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let idx = i64_array(&[1, 0], vec![2]);
        let g = gather(&a, &idx, 0).unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        assert_eq!(read_f32(&g), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn gather_2d_axis1() {
        // [[1,2,3],[4,5,6]]，indices=[0,2] → [[1,3],[4,6]]
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let idx = i64_array(&[0, 2], vec![2]);
        let g = gather(&a, &idx, 1).unwrap();
        assert_eq!(g.shape(), &[2, 2]);
        assert_eq!(read_f32(&g), vec![1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn gather_1d() {
        let a = f32_array(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        let idx = i64_array(&[3, 1, 3], vec![3]);
        let g = gather(&a, &idx, 0).unwrap();
        assert_eq!(g.shape(), &[3]);
        assert_eq!(read_f32(&g), vec![40.0, 20.0, 40.0]);
    }

    #[test]
    fn gather_3d_middle_axis() {
        // shape [2,3,2]，axis=1，indices=[2,0] → shape [2,2,2]
        let vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let a = f32_array(&vals, vec![2, 3, 2]);
        let idx = i64_array(&[2, 0], vec![2]);
        let g = gather(&a, &idx, 1).unwrap();
        assert_eq!(g.shape(), &[2, 2, 2]);
        // a[0,2,:] = [4,5]，a[0,0,:] = [0,1]；a[1,2,:] = [10,11]，a[1,0,:] = [6,7]
        assert_eq!(read_f32(&g), vec![4.0, 5.0, 0.0, 1.0, 10.0, 11.0, 6.0, 7.0]);
    }

    #[test]
    fn gather_errors() {
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let idx = i64_array(&[0], vec![1]);
        // axis 越界
        assert!(gather(&a, &idx, 2).is_err());
        // indices 非 1D
        let idx_2d = i64_array(&[0, 1], vec![1, 2]);
        assert!(gather(&a, &idx_2d, 0).is_err());
        // indices 非 int64
        let idx_f32 = f32_array(&[0.0], vec![1]);
        assert!(gather(&a, &idx_f32, 0).is_err());
        // 索引越界
        let idx_oob = i64_array(&[0, 2], vec![2]);
        assert!(gather(&a, &idx_oob, 0).is_err());
    }

    // --- scatter ---

    #[test]
    fn scatter_2d_axis0() {
        // [[1,2],[3,4]]，indices=[1]，values=[[10,11]] → [[1,2],[10,11]]
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let idx = i64_array(&[1], vec![1]);
        let vals = f32_array(&[10.0, 11.0], vec![1, 2]);
        let s = scatter(&a, &idx, &vals, 0).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(read_f32(&s), vec![1.0, 2.0, 10.0, 11.0]);
        // 原数组不被修改
        assert_eq!(read_f32(&a), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn scatter_2d_axis1() {
        // [[1,2,3],[4,5,6]]，indices=[0,2]，values=[[7,8],[9,10]]
        // → [[7,2,8],[9,5,10]]
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let idx = i64_array(&[0, 2], vec![2]);
        let vals = f32_array(&[7.0, 8.0, 9.0, 10.0], vec![2, 2]);
        let s = scatter(&a, &idx, &vals, 1).unwrap();
        assert_eq!(read_f32(&s), vec![7.0, 2.0, 8.0, 9.0, 5.0, 10.0]);
    }

    #[test]
    fn scatter_errors() {
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let idx = i64_array(&[0], vec![1]);
        let vals = f32_array(&[10.0, 11.0], vec![1, 2]);
        // axis 越界
        assert!(scatter(&a, &idx, &vals, 2).is_err());
        // values shape 不匹配（axis=0 时应为 [1, 2]）
        let vals_bad = f32_array(&[10.0, 11.0], vec![2, 1]);
        assert!(scatter(&a, &idx, &vals_bad, 0).is_err());
        // values dtype 不匹配
        let vals_i64 = i64_array(&[10, 11], vec![1, 2]);
        assert!(scatter(&a, &idx, &vals_i64, 0).is_err());
        // 索引越界
        let idx_oob = i64_array(&[2], vec![1]);
        assert!(scatter(&a, &idx_oob, &vals, 0).is_err());
    }

    // --- view + gather/scatter 组合（offset/负 stride 输入）---

    #[test]
    fn gather_on_flipped_view() {
        // flip([[1,2],[3,4]], axis=1) = [[2,1],[4,3]]，gather(axis=0, indices=[1]) → [[4,3]]
        let a = f32_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let f = flip(&a, 1).unwrap();
        let idx = i64_array(&[1], vec![1]);
        let g = gather(&f, &idx, 0).unwrap();
        assert_eq!(read_f32(&g), vec![4.0, 3.0]);
    }

    #[test]
    fn scatter_on_offset_view() {
        // slice([10,11,12,13,14][1:4]) = [11,12,13]，scatter(indices=[1], values=[99]) → [11,99,13]
        let a = f32_array(&[10.0, 11.0, 12.0, 13.0, 14.0], vec![5]);
        let s = slice(&a, &[SliceSpec { start: 1, stop: 4, step: 1 }]).unwrap();
        let idx = i64_array(&[1], vec![1]);
        let vals = f32_array(&[99.0], vec![1]);
        let r = scatter(&s, &idx, &vals, 0).unwrap();
        assert_eq!(read_f32(&r), vec![11.0, 99.0, 13.0]);
        // 原数组不变
        assert_eq!(read_f32(&a), vec![10.0, 11.0, 12.0, 13.0, 14.0]);
    }
}
