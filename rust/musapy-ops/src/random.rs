//! 随机数算子（v0.3-alpha Phase 4，ADR-003 003-D7）
//!
//! `rand` / `randn` / `uniform` / `normal` / `bernoulli`，**GPU-only**
//! （v0.3 数学库算子一律不建 CPU fallback，003-D4 修订）。
//! 实现走 muRAND（murandGenerateUniform/Double、murandGenerateNormal/Double），
//! generator 句柄经 `math_handle::with_murand_generator` 按 device 懒创建缓存复用。
//!
//! 关键语义：
//!   - **seed**：给定 seed → 调用前 `murandSetPseudoRandomGeneratorSeed` 重置，
//!     同 seed 紧邻两次调用逐元素可复现；无 seed → 不重置（自然推进）。
//!   - **shape=None → 0-dim 标量数组**（NumPy 对齐；用户确认，见 plan-phase4）。
//!   - **normal(loc, scale)**：`murandGenerateNormal` 原生支持 mean/stddev
//!     （2026-08-07 头文件核对），一步生成，无仿射变换。
//!   - **uniform(low, high)** = rand·(high−low)+low：复用既有 elementwise
//!     算子（mul/add）+ 0-dim full 标量（零新增 kernel，P4.4 惯例）。
//!   - **bernoulli(p)** = rand < p → **bool**：复用 comparison::lt。
//!   - 输出恒 C-contiguous；空 shape（0 元素）早退不调生成器（count=0
//!     行为未定义，规避）。

use musapy_core::math_handle;
use musapy_core::resolution;
use musapy_core::{Array, Buffer, BufferRef, Device, Dtype, Layout, Result, Stream, musa_x_ffi};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::linalg::{check_float_whitelist, require_musa};

// ── 生成动作 ─────────────────────────────────────────────────

/// muRAND 生成动作：uniform [0,1) 或 normal（原生 mean/stddev）。
enum GenerateKind {
    Uniform,
    Normal { mean: f64, stddev: f64 },
}

/// random 生成专用流：无显式 stream context 时按 device 缓存**单一**流。
///
/// 003-D9（2026-08-07 真机探针）：generator 为按 device 缓存的共享资源，
/// 若每个 op 新建流，生成 kernel 跨流并发会与 seed 重置交错，破坏
/// 「同 seed 逐元素可复现」；同流异步序列完全可复现（无需中间同步）。
/// 有用户 stream context 时仍遵循「每 op 显式传 stream」惯例（用户负责
/// 排序；多流并发调用 random 的复现性由调用方保证）。
fn generation_stream(device: &Device) -> Result<Arc<Stream>> {
    if let Some(s) = resolution::get_current_stream() {
        return Ok(s);
    }
    static RANDOM_STREAMS: OnceLock<Mutex<HashMap<Device, Arc<Stream>>>> = OnceLock::new();
    let map = RANDOM_STREAMS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap();
    if let Some(s) = map.get(device) {
        return Ok(Arc::clone(s));
    }
    let s = Arc::new(Stream::new(device.clone(), 0)?);
    map.insert(device.clone(), Arc::clone(&s));
    Ok(s)
}

// ── 骨架 ─────────────────────────────────────────────────────

/// 随机生成通用骨架：解析 shape/dtype/device → 分配 buffer → 生成 → 构造 Array。
///
/// `seed`：Some(s) → 生成前重置生成器；None → 不重置。
fn generate_skeleton(
    shape: &[usize],
    dtype_arg: Option<Dtype>,
    device_arg: Option<Device>,
    seed: Option<u64>,
    kind: GenerateKind,
) -> Result<Array> {
    // ── Phase A: 参数解析（capture-safe，仿 creation_skeleton）──

    // 1. Device resolution（无输入，Level 3 跳过）+ GPU-only 校验
    let dev_res = resolution::resolve_device(device_arg, &[])?;
    let device = dev_res.device.clone();
    require_musa("random", &device)?;

    // 2. Dtype resolution（无输入 → float32 兜底）+ f32/f64 白名单
    let dtype_res = resolution::resolve_dtype(dtype_arg, &[])?;
    let dtype = dtype_res.dtype;
    check_float_whitelist("random", dtype)?;

    // 3. Layout + nbytes
    let layout = Layout::from_shape(shape.to_vec());
    let n = layout.size();
    let nbytes = n * dtype.element_size();

    // 4. Stream 选择（random 专用缓存流：生成串行化，见 generation_stream）
    let stream: Arc<Stream> = generation_stream(&device)?;

    // 5. Buffer 分配（自动走 buffer pool）
    let buffer = Buffer::alloc(nbytes.max(1), device.clone(), &stream)?;
    let data_ref = BufferRef::new(Arc::new(buffer));

    // ── Phase B: 生成（0 元素早退，不调生成器）──

    if n > 0 {
        let out_ptr = data_ref.buffer().ptr().ok_or_else(|| {
            musapy_core::error::MusapyError::Device(
                musapy_core::error::DeviceError::MathLibCallFailed(
                    "random: null buffer pointer".into(),
                ),
            )
        })?;
        math_handle::with_murand_generator(&device, &stream, |generator| {
            if let Some(s) = seed {
                musa_x_ffi::check_murand(
                    unsafe { musa_x_ffi::murandSetPseudoRandomGeneratorSeed(generator, s) },
                    "murandSetPseudoRandomGeneratorSeed",
                )?;
            }
            let status = match (dtype, kind) {
                (Dtype::Float32, GenerateKind::Uniform) => unsafe {
                    musa_x_ffi::murandGenerateUniform(generator, out_ptr.as_ptr() as *mut f32, n)
                },
                (Dtype::Float64, GenerateKind::Uniform) => unsafe {
                    musa_x_ffi::murandGenerateUniformDouble(
                        generator,
                        out_ptr.as_ptr() as *mut f64,
                        n,
                    )
                },
                (Dtype::Float32, GenerateKind::Normal { mean, stddev }) => unsafe {
                    musa_x_ffi::murandGenerateNormal(
                        generator,
                        out_ptr.as_ptr() as *mut f32,
                        n,
                        mean as f32,
                        stddev as f32,
                    )
                },
                (Dtype::Float64, GenerateKind::Normal { mean, stddev }) => unsafe {
                    musa_x_ffi::murandGenerateNormalDouble(
                        generator,
                        out_ptr.as_ptr() as *mut f64,
                        n,
                        mean,
                        stddev,
                    )
                },
                _ => unreachable!("dtype already validated as float32/float64"),
            };
            musa_x_ffi::check_murand(status, "murand generate")?;
            Ok(())
        })?;
    }

    // ── Phase C: 后处理 ──

    data_ref.buffer().record_write(&stream);
    Ok(Array::new(
        data_ref, layout, dtype, stream, dev_res, dtype_res,
    ))
}

// ── 公开 API ─────────────────────────────────────────────────

/// `ms.random.rand(*shape, dtype=float32, device=None, seed=None)` — uniform [0,1)。
pub fn rand(
    shape: &[usize],
    dtype: Option<Dtype>,
    device: Option<Device>,
    seed: Option<u64>,
) -> Result<Array> {
    generate_skeleton(shape, dtype, device, seed, GenerateKind::Uniform)
}

/// `ms.random.randn(*shape, dtype=float32, seed=None)` — 标准正态 N(0,1)。
pub fn randn(
    shape: &[usize],
    dtype: Option<Dtype>,
    device: Option<Device>,
    seed: Option<u64>,
) -> Result<Array> {
    generate_skeleton(
        shape,
        dtype,
        device,
        seed,
        GenerateKind::Normal {
            mean: 0.0,
            stddev: 1.0,
        },
    )
}

/// `ms.random.uniform(low=0.0, high=1.0, shape=None, ...)` — [low, high) 均匀分布。
///
/// = rand·(high−low)+low（复用 mul/add + 0-dim 标量，零新增 kernel）。
pub fn uniform(
    shape: &[usize],
    low: f64,
    high: f64,
    dtype: Option<Dtype>,
    device: Option<Device>,
    seed: Option<u64>,
) -> Result<Array> {
    let r = rand(shape, dtype, device.clone(), seed)?;
    let r_dtype = r.dtype();
    let r_device = r.device().clone();
    // 0-dim 标量参与广播（broadcast_shape 空 shape = 标量，NumPy 规则）
    let scale = crate::creation::full(&[], high - low, Some(r_dtype), Some(r_device.clone()))?;
    let offset = crate::creation::full(&[], low, Some(r_dtype), Some(r_device))?;
    let scaled = crate::elementwise::mul(&r, &scale, None)?;
    crate::elementwise::add(&scaled, &offset, None)
}

/// `ms.random.normal(loc=0.0, scale=1.0, shape=None, ...)` — N(loc, scale²)。
///
/// 原生 mean/stddev 一步生成（murandGenerateNormal，头文件核对）。
pub fn normal(
    shape: &[usize],
    loc: f64,
    scale: f64,
    dtype: Option<Dtype>,
    device: Option<Device>,
    seed: Option<u64>,
) -> Result<Array> {
    generate_skeleton(
        shape,
        dtype,
        device,
        seed,
        GenerateKind::Normal {
            mean: loc,
            stddev: scale,
        },
    )
}

/// `ms.random.bernoulli(p=0.5, shape=None, seed=None)` — Bernoulli 分布 → bool。
///
/// = rand < p（f32 uniform 内部生成，comparison::lt 输出 bool）。
pub fn bernoulli(
    shape: &[usize],
    p: f64,
    device: Option<Device>,
    seed: Option<u64>,
) -> Result<Array> {
    let r = rand(shape, None, device.clone(), seed)?;
    let r_dtype = r.dtype();
    let r_device = r.device().clone();
    let p_arr = crate::creation::full(&[], p, Some(r_dtype), Some(r_device))?;
    crate::comparison::lt(&r, &p_arr, None)
}
