//! MUSA-X 句柄生命周期冒烟入口（v0.3 P1.7，ADR-003 003-D2）。
//!
//! 仅测试用：只注册进 `_core`，**不**进 `musapy/__init__.py` 公开 API。
//! 用途：
//!   1. 真机验证 4 库（muBLAS/muRAND/muFFT/muSPARSE）版本查询可走通；
//!   2. 验证「懒创建 → SetStream → evict（延迟销毁队列）→ reclaim」闭环
//!      不泄漏：mem_stats 持平、延迟销毁队列归零。
//!
//! muSOLVER 无独立句柄（复用 mublasHandle_t，003-D2），故冒烟覆盖 4 类句柄。

use crate::error;
use musapy_core::error::Result;
use musapy_core::{Device, Stream, math_handle, mem_stats, musa_ffi, musa_x_ffi};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// 每隔多少轮回收一次延迟销毁队列。
///
/// iters=1e6 时若不中途回收，队列会积压 4×iters 条目（host 内存上百 MB）；
/// 周期性 reclaim 既压测 Destroy 路径又限制队列规模。
const RECLAIM_EVERY: usize = 1024;

/// 查询设备空闲显存（驱动级视角）。
///
/// 句柄本身不在 mem_stats 记账内，驱动级 VRAM 对比才能捕捉
/// create/destroy 不平衡导致的真实泄漏。CPU 设备返回 None。
fn vram_free_bytes(dev: &Device) -> Result<Option<usize>> {
    let id = match dev {
        Device::Musa(id) => *id,
        Device::Cpu => return Ok(None),
    };
    musa_ffi::set_device(id as i32)?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    musa_ffi::check_musa(
        unsafe { musa_ffi::musaMemGetInfo(&mut free, &mut total) },
        "musaMemGetInfo",
    )?;
    Ok(Some(free))
}

/// 数学库句柄冒烟测试（仅测试用，非公开 API）。
///
/// 流程：4 库版本查询 → `iters` 轮「4 类句柄懒创建/SetStream → evict_device
/// 入延迟销毁队列」→ 周期性 reclaim_destroys → 结束 synchronize 清空残余
/// → 返回版本号、mem_stats 前后快照、残余队列计数。
///
/// Python 侧以 `_core._math_handle_smoke(device="musa:0", iters=1000)` 调用。
#[pyfunction(name = "_math_handle_smoke")]
#[pyo3(signature = (device = "musa:0".to_string(), iters = 1000))]
fn math_handle_smoke(py: Python<'_>, device: String, iters: usize) -> PyResult<PyObject> {
    let dev = Device::parse(&device).map_err(error::to_pyerr)?;
    let stream = Stream::new(dev.clone(), 0).map_err(error::to_pyerr)?;

    let before = mem_stats::snapshot();

    // 4 库版本查询（同时首次懒创建 mublas/musparse 句柄）
    let versions = math_handle::library_versions(&dev, &stream).map_err(error::to_pyerr)?;

    // VRAM 基线取在首次句柄创建之后：首次创建的一次性开销不算泄漏，
    // 之后的 N 轮 create/destroy 循环若再增长才是泄漏。
    let vram_before = vram_free_bytes(&dev).map_err(error::to_pyerr)?;

    // 冒烟用 FFT plan 规格（固定形状，池化复用键）
    let spec = math_handle::MufftPlanSpec::OneD {
        nx: 1024,
        ftype: musa_x_ffi::MUFFT_C2C,
        batch: 1,
    };

    let dev_ref = &dev;
    let stream_ref = &stream;
    let spec_ref = &spec;
    let run = || -> Result<()> {
        for i in 0..iters {
            // 懒创建（或复用）+ SetStream
            math_handle::with_mublas_handle(dev_ref, stream_ref, |_| Ok(()))?;
            math_handle::with_murand_generator(dev_ref, stream_ref, |_| Ok(()))?;
            math_handle::with_musparse_handle(dev_ref, stream_ref, |_| Ok(()))?;
            math_handle::with_mufft_plan(dev_ref, stream_ref, spec_ref, |_| Ok(()))?;
            // 句柄移入延迟销毁队列，下轮重建 → create/destroy 循环
            math_handle::evict_device(dev_ref);
            if i % RECLAIM_EVERY == RECLAIM_EVERY - 1 {
                math_handle::reclaim_destroys()?;
            }
        }
        Ok(())
    };

    // 长循环释放 GIL（避免卡 Python 信号）
    py.allow_threads(run).map_err(error::to_pyerr)?;

    // 终局回收：synchronize 成功分支会 reclaim 残余延迟销毁；再确认队列归零
    stream.synchronize().map_err(error::to_pyerr)?;
    let pending_after = math_handle::pending_destroy_count();

    let after = mem_stats::snapshot();
    let vram_after = vram_free_bytes(&dev).map_err(error::to_pyerr)?;

    let dict = PyDict::new(py);
    let versions_d = PyDict::new(py);
    versions_d.set_item("mublas", versions.mublas)?;
    versions_d.set_item("murand", versions.murand)?;
    versions_d.set_item("mufft", versions.mufft)?;
    versions_d.set_item("musparse", versions.musparse)?;
    dict.set_item("versions", versions_d)?;
    dict.set_item("iters", iters)?;
    dict.set_item("mem_allocated_bytes_before", before.allocated_bytes)?;
    dict.set_item("mem_allocated_bytes_after", after.allocated_bytes)?;
    dict.set_item("mem_allocated_buffers_before", before.allocated_buffers)?;
    dict.set_item("mem_allocated_buffers_after", after.allocated_buffers)?;
    dict.set_item("mem_cached_bytes_after", after.cached_bytes)?;
    dict.set_item("pending_destroys_after", pending_after)?;
    dict.set_item("vram_free_bytes_before", vram_before)?;
    dict.set_item("vram_free_bytes_after", vram_after)?;
    Ok(dict.into())
}

/// 注册进 `_core` 模块（由 lib.rs 调用）。
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(math_handle_smoke, m)?)
}
