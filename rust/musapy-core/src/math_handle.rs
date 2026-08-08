//! MUSA-X 句柄 / plan / workspace 生命周期管理(v0.3,ADR-003 003-D2)
//!
//! **每 device 4 类句柄**(懒创建、按 device 缓存、跨 op 复用):
//!   - `mublasHandle_t`(muBLAS **+ muSOLVER** 共享 —— SDK 3.1.0 无 musolver 独立句柄)
//!   - `murandGenerator_t`(muRAND)
//!   - `mufftHandle` plan 池(按 `MufftPlanSpec` 键缓存)
//!   - `musparseHandle_t`(muSPARSE)
//!
//! **生命周期规则**:
//!   - 创建:首次使用时懒创建;失败 → `DeviceError::MathLibCallFailed`(003-D2)。
//!   - stream 绑定:每次 op 前 `SetStream` 当前 op 的 stream
//!     (`MUstream ≡ musaStream_t`,无需转换)。
//!   - 释放:`evict_device()` 把句柄移入延迟销毁队列,`reclaim_destroys()`
//!     在 `Stream::synchronize` 成功后批量执行(L3-9/L3-10 惯例);
//!     workspace(设备内存)走 `deferred_free` 队列。
//!   - workspace:两段式使用的承载(bufferSize 查询 → get_workspace → 计算),
//!     按 next_power_of_two 分桶缓存复用(P1.5)。
//!
//! **架构边界(003-D1)**:本模块只做句柄生命周期,不发起任何计算调用;
//! 计算(gemm/getrf/fft/spmv)仅在 musapy-ops。

use crate::device::Device;
use crate::error::{DeviceError, MusapyError, Result};
use crate::musa_ffi;
use crate::musa_x_ffi::{
    self, mufftHandle, mufftType, mublasHandle_t, murandGenerator_t, musparseHandle_t,
    musparseSpMatDescr_t,
};
use crate::stream::Stream;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_int;
use std::ptr::NonNull;
use std::sync::OnceLock;

// ============================================================
// 1. 注册表(OnceLock + parking_lot::Mutex,仿 deferred_free/buffer_pool)
// ============================================================

/// mufft plan 池的键(plan 与方向无关,方向在 Exec 时传入)。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MufftPlanSpec {
    OneD { nx: i32, ftype: mufftType, batch: i32 },
    TwoD { nx: i32, ny: i32, ftype: mufftType },
    ThreeD { nx: i32, ny: i32, nz: i32, ftype: mufftType },
    Many {
        rank: i32,
        n: Vec<i32>,
        inembed: Vec<i32>,
        istride: i32,
        idist: i32,
        onembed: Vec<i32>,
        ostride: i32,
        odist: i32,
        ftype: mufftType,
        batch: i32,
    },
}

/// muSPARSE CSR 稀疏矩阵描述符缓存键（P-A3，2026-08-08）。
///
/// 描述符绑定底层 buffer 指针（musparseCreateCsr 持有 device 指针），
/// 故键为「三个 buffer 指针身份 + shape + nnz + dtype」。指针值在 Buffer
/// 生命周期内固定（alloc 时一次写入），且 CsrMatrix 的 BufferRef Arc 保活
/// buffer，故缓存不会读到失效数据。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MusparseSpMatSpec {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub data_ptr: usize,
    pub indices_ptr: usize,
    pub indptr_ptr: usize,
    pub dtype_code: i32,
}

/// 单个 device 的句柄缓存。
#[derive(Default)]
struct DeviceHandles {
    /// muBLAS 句柄,**兼供 muSOLVER**(003-D2)。
    mublas: Option<mublasHandle_t>,
    murand: Option<murandGenerator_t>,
    musparse: Option<musparseHandle_t>,
    mufft_plans: HashMap<MufftPlanSpec, mufftHandle>,
    /// musparse CSR 稀疏矩阵描述符池（P-A3：按规格键缓存，跨 op 复用）。
    musparse_descs: HashMap<MusparseSpMatSpec, musparseSpMatDescr_t>,
}

// 裸句柄指针跨线程:句柄绑定设备而非创建线程,所有使用点前都会 set_device
// (同 deferred_free::DeferredEntry / WorkspaceEntry 的理由)。
unsafe impl Send for DeviceHandles {}

/// workspace 缓存条目(设备内存,按桶尺寸分类)。
struct WorkspaceEntry {
    ptr: NonNull<u8>,
    size: usize,
}

// NonNull<u8> 跨线程:musaFree 绑定「当前设备」而非分配线程,
// reclaim 前会 set_device(同 deferred_free::DeferredEntry 的理由)。
unsafe impl Send for WorkspaceEntry {}

struct MathRegistry {
    handles: HashMap<Device, DeviceHandles>,
    /// workspace 分桶缓存:(device, bucket_size) → 空闲条目。
    workspaces: HashMap<(Device, usize), Vec<WorkspaceEntry>>,
}

static REGISTRY: OnceLock<Mutex<MathRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<MathRegistry> {
    REGISTRY.get_or_init(|| {
        Mutex::new(MathRegistry {
            handles: HashMap::new(),
            workspaces: HashMap::new(),
        })
    })
}

// ============================================================
// 2. 延迟销毁队列(句柄需 Destroy 调用,不能走 deferred_free)
// ============================================================

enum DeferredDestroy {
    Mublas(mublasHandle_t, Device),
    Murand(murandGenerator_t, Device),
    Mufft(mufftHandle, Device),
    Musparse(musparseHandle_t, Device),
    /// musparse CSR 稀疏矩阵描述符（P-A3：随 evict 入延迟销毁）。
    MusparseSpMat(musparseSpMatDescr_t, Device),
}

// 裸指针条目:Destroy 前会 set_device(同 DeferredEntry 先例)。
unsafe impl Send for DeferredDestroy {}

static DEFERRED_DESTROYS: OnceLock<Mutex<Vec<DeferredDestroy>>> = OnceLock::new();

fn deferred_destroys() -> &'static Mutex<Vec<DeferredDestroy>> {
    DEFERRED_DESTROYS.get_or_init(|| Mutex::new(Vec::new()))
}

/// 延迟销毁队列中待处理的句柄数(调试/测试用)。
pub fn pending_destroy_count() -> usize {
    deferred_destroys().lock().len()
}

// ============================================================
// 3. 内部辅助
// ============================================================

/// 取 musa 设备 id;CPU 设备不允许使用数学库。
fn musa_id(device: &Device) -> Result<u32> {
    device.musa_id().ok_or_else(|| {
        MusapyError::Device(DeviceError::Mismatch(
            "MUSA-X math libraries require a musa device, got cpu".into(),
        ))
    })
}

/// 批量执行延迟销毁(仿 deferred_free::reclaim_all:单个失败不中断,
/// 返回第一个错误)。每个句柄销毁前先 set_device(L3-10 惯例)。
pub fn reclaim_destroys() -> Result<()> {
    let entries: Vec<DeferredDestroy> = std::mem::take(&mut *deferred_destroys().lock());
    let mut first_err: Option<MusapyError> = None;
    for entry in entries {
        if let Err(e) = reclaim_one_destroy(entry) {
            // 保留第一个错误,继续尝试其余(避免一个坏句柄阻塞队列)
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// 销毁单个延迟句柄:先 set_device 再 Destroy(L3-10 惯例,
/// 同 deferred_free::reclaim_one —— 句柄/内存操作绑定当前设备)。
fn reclaim_one_destroy(entry: DeferredDestroy) -> Result<()> {
    let device = match &entry {
        DeferredDestroy::Mublas(_, d)
        | DeferredDestroy::Murand(_, d)
        | DeferredDestroy::Mufft(_, d)
        | DeferredDestroy::Musparse(_, d)
        | DeferredDestroy::MusparseSpMat(_, d) => d.clone(),
    };
    // 句柄绑定设备:必须先 set 再 Destroy(多设备场景下 current device 可能已切换)
    if let Some(id) = device.musa_id() {
        musa_ffi::set_device(id as i32)?;
    }
    match entry {
        DeferredDestroy::Mublas(h, _) => {
            musa_x_ffi::check_mublas(unsafe { musa_x_ffi::mublasDestroy(h) }, "mublasDestroy")
        }
        DeferredDestroy::Murand(g, _) => musa_x_ffi::check_murand(
            unsafe { musa_x_ffi::murandDestroyGenerator(g) },
            "murandDestroyGenerator",
        ),
        DeferredDestroy::Mufft(p, _) => {
            musa_x_ffi::check_mufft(unsafe { musa_x_ffi::mufftDestroy(p) }, "mufftDestroy")
        }
        DeferredDestroy::Musparse(h, _) => {
            musa_x_ffi::check_musparse(unsafe { musa_x_ffi::musparseDestroy(h) }, "musparseDestroy")
        }
        DeferredDestroy::MusparseSpMat(d, _) => musa_x_ffi::check_musparse(
            unsafe { musa_x_ffi::musparseDestroySpMat(d) },
            "musparseDestroySpMat",
        ),
    }
}

// ============================================================
// 4. 句柄访问 API(003-D2:with_* 闭包风格)
// ============================================================
//
// 流程:set_device → 懒创建 → SetStream(stream) → f(handle)。
// 注:Python 侧 op 分发经 GIL 串行化;Rust 侧若多线程并发使用同一
// device 的句柄需外部同步(与 cuBLAS 句柄语义一致)。

/// 取(懒创建)mublas 句柄并绑定 stream 后执行闭包。
/// **muSOLVER 例程同样使用此句柄**(003-D2)。
pub fn with_mublas_handle<T>(
    device: &Device,
    stream: &Stream,
    f: impl FnOnce(mublasHandle_t) -> Result<T>,
) -> Result<T> {
    let id = musa_id(device)?;
    musa_ffi::set_device(id as i32)?;
    let handle = {
        let mut reg = registry().lock();
        let dh = reg.handles.entry(device.clone()).or_default();
        match dh.mublas {
            Some(h) => h,
            None => {
                let mut h: mublasHandle_t = std::ptr::null_mut();
                musa_x_ffi::check_mublas(unsafe { musa_x_ffi::mublasCreate(&mut h) }, "mublasCreate")?;
                // HOST pointer mode:alpha/beta 以 host 标量传入(Phase 2 约定);
                // 创建时一次性设置,避免每次计算前重复调用。
                musa_x_ffi::check_mublas(
                    unsafe {
                        musa_x_ffi::mublasSetPointerMode(h, musa_x_ffi::MUBLAS_POINTER_MODE_HOST)
                    },
                    "mublasSetPointerMode",
                )?;
                dh.mublas = Some(h);
                h
            }
        }
    };
    musa_x_ffi::check_mublas(
        unsafe { musa_x_ffi::mublasSetStream(handle, stream.raw()) },
        "mublasSetStream",
    )?;
    f(handle)
}

/// 取(懒创建)murand 生成器并绑定 stream 后执行闭包。
pub fn with_murand_generator<T>(
    device: &Device,
    stream: &Stream,
    f: impl FnOnce(murandGenerator_t) -> Result<T>,
) -> Result<T> {
    let id = musa_id(device)?;
    musa_ffi::set_device(id as i32)?;
    let generator = {
        let mut reg = registry().lock();
        let dh = reg.handles.entry(device.clone()).or_default();
        match dh.murand {
            Some(g) => g,
            None => {
                let mut g: murandGenerator_t = std::ptr::null_mut();
                musa_x_ffi::check_murand(
                    unsafe {
                        musa_x_ffi::murandCreateGenerator(
                            &mut g,
                            musa_x_ffi::MURAND_RNG_PSEUDO_DEFAULT,
                        )
                    },
                    "murandCreateGenerator",
                )?;
                dh.murand = Some(g);
                g
            }
        }
    };
    musa_x_ffi::check_murand(
        unsafe { musa_x_ffi::murandSetStream(generator, stream.raw()) },
        "murandSetStream",
    )?;
    f(generator)
}

/// 取(懒创建)musparse 句柄并绑定 stream 后执行闭包。
pub fn with_musparse_handle<T>(
    device: &Device,
    stream: &Stream,
    f: impl FnOnce(musparseHandle_t) -> Result<T>,
) -> Result<T> {
    let id = musa_id(device)?;
    musa_ffi::set_device(id as i32)?;
    let handle = {
        let mut reg = registry().lock();
        let dh = reg.handles.entry(device.clone()).or_default();
        match dh.musparse {
            Some(h) => h,
            None => {
                let mut h: musparseHandle_t = std::ptr::null_mut();
                musa_x_ffi::check_musparse(
                    unsafe { musa_x_ffi::musparseCreate(&mut h) },
                    "musparseCreate",
                )?;
                dh.musparse = Some(h);
                h
            }
        }
    };
    musa_x_ffi::check_musparse(
        unsafe { musa_x_ffi::musparseSetStream(handle, stream.raw()) },
        "musparseSetStream",
    )?;
    f(handle)
}

/// 取(懒创建)mufft plan 并绑定 stream 后执行闭包。
/// plan 按 `MufftPlanSpec` 池化复用。
pub fn with_mufft_plan<T>(
    device: &Device,
    stream: &Stream,
    spec: &MufftPlanSpec,
    f: impl FnOnce(mufftHandle) -> Result<T>,
) -> Result<T> {
    let id = musa_id(device)?;
    musa_ffi::set_device(id as i32)?;
    let plan = {
        let mut reg = registry().lock();
        let dh = reg.handles.entry(device.clone()).or_default();
        match dh.mufft_plans.get(spec) {
            Some(p) => *p,
            None => {
                let mut p: mufftHandle = std::ptr::null_mut();
                match spec {
                    MufftPlanSpec::OneD { nx, ftype, batch } => {
                        musa_x_ffi::check_mufft(
                            unsafe { musa_x_ffi::mufftPlan1d(&mut p, *nx, *ftype, *batch) },
                            "mufftPlan1d",
                        )?;
                    }
                    MufftPlanSpec::TwoD { nx, ny, ftype } => {
                        musa_x_ffi::check_mufft(
                            unsafe { musa_x_ffi::mufftPlan2d(&mut p, *nx, *ny, *ftype) },
                            "mufftPlan2d",
                        )?;
                    }
                    MufftPlanSpec::ThreeD { nx, ny, nz, ftype } => {
                        musa_x_ffi::check_mufft(
                            unsafe { musa_x_ffi::mufftPlan3d(&mut p, *nx, *ny, *nz, *ftype) },
                            "mufftPlan3d",
                        )?;
                    }
                    MufftPlanSpec::Many {
                        rank,
                        n,
                        inembed,
                        istride,
                        idist,
                        onembed,
                        ostride,
                        odist,
                        ftype,
                        batch,
                    } => {
                        musa_x_ffi::check_mufft(
                            unsafe {
                                musa_x_ffi::mufftPlanMany(
                                    &mut p,
                                    *rank,
                                    n.as_ptr() as *mut c_int,
                                    if inembed.is_empty() { std::ptr::null_mut() } else { inembed.as_ptr() as *mut c_int },
                                    *istride,
                                    *idist,
                                    if onembed.is_empty() { std::ptr::null_mut() } else { onembed.as_ptr() as *mut c_int },
                                    *ostride,
                                    *odist,
                                    *ftype,
                                    *batch,
                                )
                            },
                            "mufftPlanMany",
                        )?;
                    }
                }
                dh.mufft_plans.insert(spec.clone(), p);
                p
            }
        }
    };
    musa_x_ffi::check_mufft(
        unsafe { musa_x_ffi::mufftSetStream(plan, stream.raw()) },
        "mufftSetStream",
    )?;
    f(plan)
}

/// 取(懒创建)musparse 句柄 + CSR 稀疏矩阵描述符，绑定 stream 后执行闭包（P-A3）。
///
/// 描述符按 `MusparseSpMatSpec`（三 buffer 指针 + shape + nnz + dtype）池化
/// 缓存，跨 spmv/spmm 调用复用——避免每次 create/destroy 的固定开销
/// （小矩阵 spmv 实测 ~0.7ms 中描述符生命周期主导）。
/// 闭包收 `(handle, desc)`（SpMV/SpMM 需要 handle 传 stream 语义；
/// stream 由 `musparseSetStream(handle, ...)` 绑定，描述符不独立绑 stream）。
/// `data_ptr/indices_ptr/indptr_ptr` 为 device buffer 裸指针；调用方须保证
/// buffer 在描述符使用期间保活（CsrMatrix 的 BufferRef Arc 已满足）。
#[allow(clippy::too_many_arguments)]
pub fn with_musparse_csr<T>(
    device: &Device,
    stream: &Stream,
    spec: &MusparseSpMatSpec,
    data_ptr: *mut std::ffi::c_void,
    indices_ptr: *mut std::ffi::c_void,
    indptr_ptr: *mut std::ffi::c_void,
    f: impl FnOnce(musparseHandle_t, musparseSpMatDescr_t) -> Result<T>,
) -> Result<T> {
    let id = musa_id(device)?;
    musa_ffi::set_device(id as i32)?;
    let (handle, desc) = {
        let mut reg = registry().lock();
        let dh = reg.handles.entry(device.clone()).or_default();
        // 懒创建 musparse handle
        let handle = match dh.musparse {
            Some(h) => h,
            None => {
                let mut h: musparseHandle_t = std::ptr::null_mut();
                musa_x_ffi::check_musparse(
                    unsafe { musa_x_ffi::musparseCreate(&mut h) },
                    "musparseCreate",
                )?;
                dh.musparse = Some(h);
                h
            }
        };
        // 缓存或创建 CSR 描述符
        let desc = match dh.musparse_descs.get(spec) {
            Some(d) => *d,
            None => {
                let mut d: musparseSpMatDescr_t = std::ptr::null_mut();
                musa_x_ffi::check_musparse(
                    unsafe {
                        musa_x_ffi::musparseCreateCsr(
                            &mut d,
                            spec.rows as i64,
                            spec.cols as i64,
                            spec.nnz as i64,
                            indptr_ptr,
                            indices_ptr,
                            data_ptr,
                            musa_x_ffi::MUSPARSE_INDEX_32I,
                            musa_x_ffi::MUSPARSE_INDEX_32I,
                            musa_x_ffi::MUSPARSE_INDEX_BASE_ZERO,
                            spec.dtype_code,
                        )
                    },
                    "musparseCreateCsr",
                )?;
                dh.musparse_descs.insert(spec.clone(), d);
                d
            }
        };
        (handle, desc)
    };
    musa_x_ffi::check_musparse(
        unsafe { musa_x_ffi::musparseSetStream(handle, stream.raw()) },
        "musparseSetStream",
    )?;
    f(handle, desc)
}

// ============================================================
// 5. 版本查询(P1.7 冒烟测试)
// ============================================================

/// 四个数学库的版本号(懒创建句柄后查询)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MathLibVersions {
    pub mublas: i32,
    pub murand: i32,
    pub mufft: i32,
    pub musparse: i32,
}

/// 查询 4 个 MUSA-X 库的版本(mublas/musparse 需句柄,murand/mufft 不需要)。
pub fn library_versions(device: &Device, stream: &Stream) -> Result<MathLibVersions> {
    let id = musa_id(device)?;
    musa_ffi::set_device(id as i32)?;

    let mublas = with_mublas_handle(device, stream, |h| {
        let mut v: c_int = 0;
        musa_x_ffi::check_mublas(
            unsafe { musa_x_ffi::mublasGetVersion(h, &mut v) },
            "mublasGetVersion",
        )?;
        Ok(v)
    })?;

    let mut murand_v: c_int = 0;
    musa_x_ffi::check_murand(
        unsafe { musa_x_ffi::murandGetVersion(&mut murand_v) },
        "murandGetVersion",
    )?;

    let mut mufft_v: c_int = 0;
    musa_x_ffi::check_mufft(
        unsafe { musa_x_ffi::mufftGetVersion(&mut mufft_v) },
        "mufftGetVersion",
    )?;

    let musparse = with_musparse_handle(device, stream, |h| {
        let mut v: c_int = 0;
        musa_x_ffi::check_musparse(
            unsafe { musa_x_ffi::musparseGetVersion(h, &mut v) },
            "musparseGetVersion",
        )?;
        Ok(v)
    })?;

    Ok(MathLibVersions {
        mublas,
        murand: murand_v,
        mufft: mufft_v,
        musparse,
    })
}

// ============================================================
// 6. workspace 管理(P1.5:两段式使用的承载,分桶缓存)
// ============================================================

/// 最小桶尺寸(4KB;小于此值的查询统一按 4KB 分配)。
const MIN_WORKSPACE_BUCKET: usize = 4096;

fn bucket_for(required: usize) -> usize {
    required.max(MIN_WORKSPACE_BUCKET).next_power_of_two()
}

/// workspace 租约:Drop 时归还桶缓存。
pub struct WorkspaceLease {
    ptr: NonNull<u8>,
    size: usize,
    device: Device,
}

impl WorkspaceLease {
    /// 设备指针(传给 *_bufferSize 之后的计算例程)。
    pub fn ptr(&self) -> *mut std::ffi::c_void {
        self.ptr.as_ptr() as *mut std::ffi::c_void
    }

    /// 实际分配的桶尺寸(≥ 请求值)。
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for WorkspaceLease {
    fn drop(&mut self) {
        let mut reg = registry().lock();
        reg.workspaces
            .entry((self.device.clone(), self.size))
            .or_default()
            .push(WorkspaceEntry {
                ptr: self.ptr,
                size: self.size,
            });
    }
}

/// 获取 ≥ `required` 字节的 workspace(分桶缓存复用;桶 = next_power_of_two)。
///
/// 与 muSOLVER/muSPARSE 两段式协议配合:ops 侧先 `*_bufferSize` 查询所需
/// 大小,再 `get_workspace` 取租约,把 `lease.ptr()` 传给计算例程。
pub fn get_workspace(device: &Device, required: usize) -> Result<WorkspaceLease> {
    let id = musa_id(device)?;
    let bucket = bucket_for(required);

    // 缓存命中?
    {
        let mut reg = registry().lock();
        if let Some(entry) = reg
            .workspaces
            .get_mut(&(device.clone(), bucket))
            .and_then(|v| v.pop())
        {
            return Ok(WorkspaceLease {
                ptr: entry.ptr,
                size: bucket,
                device: device.clone(),
            });
        }
    }

    // 未命中:musaMalloc(绑定当前设备,先 set_device —— 同 Buffer::alloc 纪律)
    musa_ffi::set_device(id as i32)?;
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    musa_ffi::check_musa(unsafe { musa_ffi::musaMalloc(&mut dev_ptr, bucket) }, "musaMalloc")?;
    let ptr = NonNull::new(dev_ptr as *mut u8).ok_or_else(|| {
        MusapyError::Device(DeviceError::MathLibCallFailed(
            "musaMalloc returned null workspace pointer".into(),
        ))
    })?;
    crate::mem_stats::record_alloc(bucket);
    Ok(WorkspaceLease {
        ptr,
        size: bucket,
        device: device.clone(),
    })
}

// ============================================================
// 7. device 注销(P1.7 泄漏测试 / 未来 ms.reset_device 的入口)
// ============================================================

/// 把 device 的全部句柄移入延迟销毁队列、workspace 移入 deferred_free 队列。
///
/// 实际的 Destroy/musaFree 在下一次 `Stream::synchronize` 成功时执行
/// (`reclaim_destroys()` + `deferred_free::reclaim_all()`)。
pub fn evict_device(device: &Device) {
    let (handles, workspaces) = {
        let mut reg = registry().lock();
        let dh = reg.handles.remove(device);
        let mut ws: Vec<Vec<WorkspaceEntry>> = Vec::new();
        reg.workspaces.retain(|(d, _bucket), entries| {
            if d == device {
                ws.push(std::mem::take(entries));
                false
            } else {
                true
            }
        });
        (dh, ws)
    };

    if let Some(dh) = handles {
        let mut q = deferred_destroys().lock();
        if let Some(h) = dh.mublas {
            q.push(DeferredDestroy::Mublas(h, device.clone()));
        }
        if let Some(g) = dh.murand {
            q.push(DeferredDestroy::Murand(g, device.clone()));
        }
        if let Some(h) = dh.musparse {
            q.push(DeferredDestroy::Musparse(h, device.clone()));
        }
        for (_, plan) in dh.mufft_plans {
            q.push(DeferredDestroy::Mufft(plan, device.clone()));
        }
        for (_, desc) in dh.musparse_descs {
            q.push(DeferredDestroy::MusparseSpMat(desc, device.clone()));
        }
    }

    // workspace 是真实设备内存:记账转出 + deferred_free(L3-9)
    for entries in workspaces {
        for entry in entries {
            crate::mem_stats::record_dealloc(entry.size);
            release_workspace_memory(entry.ptr, device.clone(), entry.size);
        }
    }
}

/// workspace 内存的延迟释放路由:默认路径走 deferred_free 队列;
/// stream-ordered feature 下直接 musaFree(该 feature 不编译 deferred_free)。
#[cfg(not(feature = "stream-ordered"))]
fn release_workspace_memory(ptr: NonNull<u8>, device: Device, size: usize) {
    crate::deferred_free::enqueue(ptr, device, size);
}

#[cfg(feature = "stream-ordered")]
fn release_workspace_memory(ptr: NonNull<u8>, device: Device, size: usize) {
    if let Some(id) = device.musa_id() {
        let _ = musa_ffi::set_device(id as i32);
        let _ = musa_ffi::check_musa(
            unsafe { musa_ffi::musaFree(ptr.as_ptr() as *mut std::ffi::c_void) },
            "musaFree(workspace evict)",
        );
    }
    crate::mem_stats::record_cached(size);
    crate::mem_stats::record_reclaimed(size);
}

// ============================================================
// 8. 测试(仅 mock 模式:真实模式下 musapy-core 不链接 MUSA-X,003-D1)
// ============================================================

#[cfg(all(test, musapy_mock_musa))]
mod tests {
    use super::*;

    /// 注册表 / 延迟销毁队列 / mem_stats 都是进程级全局状态,
    /// cargo test 默认并行 → 本模块测试全部串行执行。
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn guard() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn test_stream(dev: &Device) -> Stream {
        // mock 模式下 Stream::new 走 mock stub
        Stream::new(dev.clone(), 0).expect("mock stream")
    }

    #[test]
    fn test_lazy_create_and_reuse() {
        let _g = guard();
        let dev = Device::Musa(101);
        let stream = test_stream(&dev);

        let h1 = with_mublas_handle(&dev, &stream, |h| Ok(h as usize)).unwrap();
        let h2 = with_mublas_handle(&dev, &stream, |h| Ok(h as usize)).unwrap();
        assert_eq!(h1, h2, "句柄应跨 op 复用(懒创建幂等)");

        let g1 = with_murand_generator(&dev, &stream, |g| Ok(g as usize)).unwrap();
        let g2 = with_murand_generator(&dev, &stream, |g| Ok(g as usize)).unwrap();
        assert_eq!(g1, g2);

        let s1 = with_musparse_handle(&dev, &stream, |h| Ok(h as usize)).unwrap();
        let s2 = with_musparse_handle(&dev, &stream, |h| Ok(h as usize)).unwrap();
        assert_eq!(s1, s2);

        let spec = MufftPlanSpec::OneD { nx: 64, ftype: musa_x_ffi::MUFFT_C2C, batch: 1 };
        let p1 = with_mufft_plan(&dev, &stream, &spec, |p| Ok(p as usize)).unwrap();
        let p2 = with_mufft_plan(&dev, &stream, &spec, |p| Ok(p as usize)).unwrap();
        assert_eq!(p1, p2, "plan 池应复用同规格 plan");

        let spec2 = MufftPlanSpec::OneD { nx: 128, ftype: musa_x_ffi::MUFFT_C2C, batch: 1 };
        let p3 = with_mufft_plan(&dev, &stream, &spec2, |p| Ok(p as usize)).unwrap();
        assert_ne!(p1, p3, "不同规格应创建新 plan");

        evict_device(&dev);
        reclaim_destroys().unwrap();
    }

    #[test]
    fn test_cpu_device_rejected() {
        let _g = guard();
        let stream = Stream::new(Device::Cpu, 0).unwrap();
        let r = with_mublas_handle(&Device::Cpu, &stream, |_| Ok(()));
        assert!(matches!(
            r,
            Err(MusapyError::Device(DeviceError::Mismatch(_)))
        ));
    }

    #[test]
    fn test_evict_and_reclaim() {
        let _g = guard();
        let dev = Device::Musa(102);
        let stream = test_stream(&dev);

        with_mublas_handle(&dev, &stream, |_| Ok(())).unwrap();
        with_murand_generator(&dev, &stream, |_| Ok(())).unwrap();
        with_musparse_handle(&dev, &stream, |_| Ok(())).unwrap();
        let spec = MufftPlanSpec::TwoD { nx: 4, ny: 4, ftype: musa_x_ffi::MUFFT_Z2Z };
        with_mufft_plan(&dev, &stream, &spec, |_| Ok(())).unwrap();
        // P-A3：musparse CSR 描述符缓存（mock 下 buffer 为 host 内存，任意非空指针）
        let p = NonNull::new(0x1000usize as *mut u8).unwrap();
        let spmat_spec = MusparseSpMatSpec {
            rows: 2,
            cols: 2,
            nnz: 1,
            data_ptr: 0x1000,
            indices_ptr: 0x1004,
            indptr_ptr: 0x1008,
            dtype_code: 0, // MUSA_R_32F
        };
        with_musparse_csr(
            &dev,
            &stream,
            &spmat_spec,
            p.as_ptr() as *mut std::ffi::c_void,
            p.as_ptr() as *mut std::ffi::c_void,
            p.as_ptr() as *mut std::ffi::c_void,
            |_, _| Ok(()),
        )
        .unwrap();

        let base = pending_destroy_count();
        evict_device(&dev);
        assert_eq!(
            pending_destroy_count(),
            base + 5,
            "4 类句柄 + musparse 描述符应全部入延迟销毁队列"
        );
        reclaim_destroys().unwrap();
        assert_eq!(pending_destroy_count(), 0);

        // evict 后再次使用 → 懒创建新句柄
        let h = with_mublas_handle(&dev, &stream, |h| Ok(h as usize)).unwrap();
        assert!(h != 0);
        evict_device(&dev);
        reclaim_destroys().unwrap();
    }

    #[test]
    fn test_workspace_bucket_reuse() {
        let _g = guard();
        let dev = Device::Musa(103);
        let before = crate::mem_stats::snapshot();

        let ptr1 = {
            let lease = get_workspace(&dev, 1000).unwrap();
            assert_eq!(lease.size(), MIN_WORKSPACE_BUCKET, "小请求应圆整到最小桶");
            lease.ptr() as usize
        }; // lease drop → 归还桶缓存

        let ptr2 = {
            let lease = get_workspace(&dev, 2000).unwrap();
            assert_eq!(lease.size(), MIN_WORKSPACE_BUCKET);
            lease.ptr() as usize
        };
        assert_eq!(ptr1, ptr2, "同桶 workspace 应复用同一块内存");

        let ptr3 = {
            let lease = get_workspace(&dev, 1 << 20).unwrap();
            assert_eq!(lease.size(), 1 << 20, "大请求应按 next_power_of_two 分桶");
            lease.ptr() as usize
        };
        assert_ne!(ptr1, ptr3);

        // evict → workspace 转 deferred_free(或 stream-ordered 直释)
        evict_device(&dev);
        let mid = crate::mem_stats::snapshot();
        assert_eq!(mid.allocated_bytes, before.allocated_bytes, "evict 后 allocated 应归零");

        #[cfg(not(feature = "stream-ordered"))]
        {
            crate::deferred_free::reclaim_all().unwrap();
        }
        let after = crate::mem_stats::snapshot();
        assert_eq!(after.cached_bytes, before.cached_bytes);
    }

    #[test]
    fn test_library_versions() {
        let _g = guard();
        let dev = Device::Musa(104);
        let stream = test_stream(&dev);
        let v = library_versions(&dev, &stream).unwrap();
        assert_eq!(v.mublas, 30100);
        assert_eq!(v.murand, 30100);
        assert_eq!(v.mufft, 30100);
        assert_eq!(v.musparse, 30100);
        evict_device(&dev);
        reclaim_destroys().unwrap();
    }
}
