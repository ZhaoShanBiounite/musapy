//! MUSA Runtime FFI 绑定（ADR L3-9, L2-3）
//!
//! 手写 FFI 声明，对标 musa_runtime_api.h。
//! MUSA 对标 CUDA 12.8（761 兼容 API），签名与 CUDA 等价。
//!
//! ⚠️ 签名验证：本文件的 FFI 声明基于 CUDA 等价签名推断。
//! 首次使用前，请对照 /usr/local/musa/include/musa_runtime_api.h 确认：
//!   1. musaStream_t / musaEvent_t 类型定义（应为 opaque 指针）
//!   2. 各函数的参数类型和顺序
//!   3. musaDevAttrMemoryPoolsSupported 的枚举值（CUDA 为 115）
//!
//! mock 模式（musapy_mock_musa）：提供 Rust stub，不调用真实 FFI。

#![allow(non_camel_case_types)]
#![allow(non_snake_case)] // C API 命名风格（与 musa_x_ffi.rs 一致）
#![allow(clippy::missing_safety_doc)] // FFI 绑定文档另见 musa_runtime_api.h（与 musa_x_ffi.rs 一致）
use crate::error::{KernelError, MusapyError, Result, StreamError};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

// ============================================================
// 类型定义（对标 CUDA/MUSA）
// ============================================================

/// MUSA 错误码（0 = 成功，非 0 = 错误）。
pub type musaError_t = c_int;

/// 成功。
pub const MUSA_SUCCESS: musaError_t = 0;

/// MUSA 流句柄（opaque 指针，对标 cudaStream_t）。
pub type musaStream_t = *mut c_void;

/// MUSA 事件句柄（opaque 指针，对标 cudaEvent_t）。
pub type musaEvent_t = *mut c_void;

/// 设备属性枚举（对标 cudaDeviceAttr / musaDeviceAttr）。
pub type musaDeviceAttr = c_int;

/// 多处理器数量（CU 数）。driver_types.h: musaDevAttrMultiProcessorCount = 16。
pub const MUSA_DEV_ATTR_MULTIPROCESSOR_COUNT: musaDeviceAttr = 16;

/// 计算能力主版本号。driver_types.h: musaDevAttrComputeCapabilityMajor = 75。
pub const MUSA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: musaDeviceAttr = 75;

/// 计算能力次版本号。driver_types.h: musaDevAttrComputeCapabilityMinor = 76。
pub const MUSA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR: musaDeviceAttr = 76;

/// 内存拷贝方向（对标 cudaMemcpyKind / musaMemcpyKind）。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum musaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

/// MUSA 设备属性结构体（对标 musaDeviceProp / cudaDeviceProp）。
///
/// 仅声明 `name` 字段（偏移 0），其余用 `_opaque` 填充。
/// 总大小 1024 字节 > SDK 实际结构体（~600 字节），安全覆盖。
/// 通过 `musaGetDeviceProperties` 填充后读取 `name`。
#[repr(C)]
pub struct MusaDeviceProp {
    /// 设备名称（ASCII，NUL 终止）。
    pub name: [c_char; 256],
    /// 覆盖 SDK 结构体剩余字段（uuid, totalGlobalMem, major, minor, ...）。
    _opaque: [u8; 768],
}

// ============================================================
// FFI 声明（真实模式）
// ============================================================

#[cfg(not(musapy_mock_musa))]
mod real {
    use super::*;

    unsafe extern "C" {
        // --- Stream（ADR L1-7, L1-9）---
        pub fn musaStreamCreate(pStream: *mut musaStream_t) -> musaError_t;
        pub fn musaStreamCreateWithPriority(
            pStream: *mut musaStream_t,
            flags: c_uint,
            priority: c_int,
        ) -> musaError_t;
        pub fn musaStreamDestroy(stream: musaStream_t) -> musaError_t;
        pub fn musaStreamSynchronize(stream: musaStream_t) -> musaError_t;

        // --- Event（ADR L1-8, L3-10）---
        pub fn musaEventCreate(pEvent: *mut musaEvent_t) -> musaError_t;
        pub fn musaEventDestroy(event: musaEvent_t) -> musaError_t;
        pub fn musaEventRecord(event: musaEvent_t, stream: musaStream_t) -> musaError_t;
        pub fn musaStreamWaitEvent(
            stream: musaStream_t,
            event: musaEvent_t,
            flags: c_uint,
        ) -> musaError_t;

        // --- Device（ADR L1-2, L1-3, L3-11）---
        pub fn musaGetDeviceCount(count: *mut c_int) -> musaError_t;
        pub fn musaDeviceGetAttribute(
            value: *mut c_int,
            attr: musaDeviceAttr,
            device: c_int,
        ) -> musaError_t;
        // musaMalloc/musaFree 绑定到调用线程的"当前设备"，所以必须先 set。
        pub fn musaSetDevice(device: c_int) -> musaError_t;

        // --- Device: 属性查询（ADR L1-3, P5.9 device_summary）---
        pub fn musaGetDeviceProperties(prop: *mut MusaDeviceProp, device: c_int) -> musaError_t;
        pub fn musaMemGetInfo(free: *mut usize, total: *mut usize) -> musaError_t;

        // --- Memory: 默认路径（ADR L3-11，所有 SDK 版本可用）---
        // musaMalloc/musaFree 是同步分配/释放，无 stream 参数。
        // 在 MUSA 3.x/4.x/5.x 的 libmusart.so 中均导出（nm -D 确认）。
        // 调用 musaMalloc 前必须 musaSetDevice(id)。
        pub fn musaMalloc(devPtr: *mut *mut c_void, size: usize) -> musaError_t;
        pub fn musaFree(devPtr: *mut c_void) -> musaError_t;

        // --- Memory: stream-ordered 路径（ADR L3-9，仅 5.x+ SDK）---
        // musaMallocAsync/musaFreeAsync 在 3.x/4.x 是 static inline 头文件函数，
        // libmusart.so 不导出符号；仅 5.x+ 才有真实实现。
        // 用 Cargo feature gate 控制，默认不编译，避免 3.x/4.x 链接失败。
        #[cfg(feature = "stream-ordered")]
        pub fn musaMallocAsync(
            devPtr: *mut *mut c_void,
            size: usize,
            stream: musaStream_t,
        ) -> musaError_t;
        #[cfg(feature = "stream-ordered")]
        pub fn musaFreeAsync(devPtr: *mut c_void, stream: musaStream_t) -> musaError_t;

        // --- Memory: 数据拷贝（H2D/D2H/D2D，ADR L1-11）---
        // musaMemcpy 是同步拷贝，对标 cudaMemcpy。
        // 用于 ms.array() 时将 Python host 数据拷到 GPU buffer。
        pub fn musaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: musaMemcpyKind,
        ) -> musaError_t;

        // --- Error（ADR L3-1）---
        pub fn musaGetErrorString(error: musaError_t) -> *const c_char;
        pub fn musaGetLastError() -> musaError_t;
    }
}

// ============================================================
// Mock 模式 stub（CI/无 GPU 开发机）
// ============================================================

#[cfg(musapy_mock_musa)]
mod mock {
    // mock stub 在 unsafe fn 体内直接操作裸指针（host 内存 + dummy 句柄，语义安全）。
    // edition 2024 的 unsafe_op_in_unsafe_fn 在此为纯噪音，allow 掉。
    #![allow(unsafe_op_in_unsafe_fn)]
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static MOCK_HANDLE_COUNTER: AtomicUsize = AtomicUsize::new(1);

    /// mock 分配的内存记录（用裸指针地址跟踪）。
    static MOCK_ALLOCATIONS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

    /// 生成非空 dummy 句柄。
    fn next_handle() -> *mut c_void {
        let id = MOCK_HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed);
        id as *mut c_void
    }

    pub unsafe fn musaStreamCreate(pStream: *mut musaStream_t) -> musaError_t {
        if pStream.is_null() {
            return 1;
        }
        *pStream = next_handle();
        MUSA_SUCCESS
    }

    pub unsafe fn musaStreamCreateWithPriority(
        pStream: *mut musaStream_t,
        _flags: c_uint,
        _priority: c_int,
    ) -> musaError_t {
        musaStreamCreate(pStream)
    }

    pub unsafe fn musaStreamDestroy(_stream: musaStream_t) -> musaError_t {
        MUSA_SUCCESS
    }

    pub unsafe fn musaStreamSynchronize(_stream: musaStream_t) -> musaError_t {
        MUSA_SUCCESS
    }

    pub unsafe fn musaEventCreate(pEvent: *mut musaEvent_t) -> musaError_t {
        if pEvent.is_null() {
            return 1;
        }
        *pEvent = next_handle();
        MUSA_SUCCESS
    }

    pub unsafe fn musaEventDestroy(_event: musaEvent_t) -> musaError_t {
        MUSA_SUCCESS
    }

    pub unsafe fn musaEventRecord(_event: musaEvent_t, _stream: musaStream_t) -> musaError_t {
        MUSA_SUCCESS
    }

    pub unsafe fn musaStreamWaitEvent(
        _stream: musaStream_t,
        _event: musaEvent_t,
        _flags: c_uint,
    ) -> musaError_t {
        MUSA_SUCCESS
    }

    pub unsafe fn musaGetDeviceCount(count: *mut c_int) -> musaError_t {
        if count.is_null() {
            return 1;
        }
        *count = 1; // mock: 1 个设备
        MUSA_SUCCESS
    }

    pub unsafe fn musaDeviceGetAttribute(
        value: *mut c_int,
        attr: musaDeviceAttr,
        _device: c_int,
    ) -> musaError_t {
        if value.is_null() {
            return 1;
        }
        *value = match attr {
            MUSA_DEV_ATTR_MULTIPROCESSOR_COUNT => 80, // mock: 80 CUs
            MUSA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR => 2, // mock: arch 2.2
            MUSA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR => 2,
            _ => 0,
        };
        MUSA_SUCCESS
    }

    pub unsafe fn musaSetDevice(_device: c_int) -> musaError_t {
        MUSA_SUCCESS
    }

    pub unsafe fn musaGetDeviceProperties(
        prop: *mut MusaDeviceProp,
        _device: c_int,
    ) -> musaError_t {
        if prop.is_null() {
            return 1;
        }
        // mock: 填充设备名 "Mock MUSA GPU"
        let name = b"Mock MUSA GPU\0";
        let dst = (*prop).name.as_mut_ptr();
        std::ptr::copy_nonoverlapping(name.as_ptr() as *const c_char, dst, name.len());
        // 其余字节已零初始化（调用方用 MaybeUninit::zeroed）
        MUSA_SUCCESS
    }

    pub unsafe fn musaMemGetInfo(free: *mut usize, total: *mut usize) -> musaError_t {
        if free.is_null() || total.is_null() {
            return 1;
        }
        // mock: 24 GB 总内存，16 GB 空闲
        *total = 24 * 1024 * 1024 * 1024;
        *free = 16 * 1024 * 1024 * 1024;
        MUSA_SUCCESS
    }

    // --- Memory: 默认路径 mock stub（ADR L3-11，所有构建模式可用）---

    pub unsafe fn musaMalloc(devPtr: *mut *mut c_void, size: usize) -> musaError_t {
        mock_alloc(devPtr, size)
    }

    pub unsafe fn musaFree(devPtr: *mut c_void) -> musaError_t {
        mock_free(devPtr)
    }

    // --- Memory: stream-ordered 路径 mock stub（ADR L3-9，仅 feature gate）---

    #[cfg(feature = "stream-ordered")]
    pub unsafe fn musaMallocAsync(
        devPtr: *mut *mut c_void,
        size: usize,
        _stream: musaStream_t,
    ) -> musaError_t {
        mock_alloc(devPtr, size)
    }

    #[cfg(feature = "stream-ordered")]
    pub unsafe fn musaFreeAsync(devPtr: *mut c_void, _stream: musaStream_t) -> musaError_t {
        mock_free(devPtr)
    }

    // --- Memory: 数据拷贝 mock stub ---

    pub unsafe fn musaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        _kind: musaMemcpyKind,
    ) -> musaError_t {
        if dst.is_null() || src.is_null() {
            return 1;
        }
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
        MUSA_SUCCESS
    }

    /// mock 共享分配器：用 std::alloc 分配真实内存，让指针非空且可读写。
    unsafe fn mock_alloc(devPtr: *mut *mut c_void, size: usize) -> musaError_t {
        if devPtr.is_null() {
            return 1;
        }
        let layout = std::alloc::Layout::from_size_align(size.max(1), 8).unwrap();
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            return 2; // OOM
        }
        *devPtr = ptr as *mut c_void;
        MOCK_ALLOCATIONS.lock().unwrap().push(ptr as usize);
        MUSA_SUCCESS
    }

    /// mock 共享释放器：从跟踪表移除（不真正 dealloc，靠进程退出回收）。
    unsafe fn mock_free(devPtr: *mut c_void) -> musaError_t {
        if devPtr.is_null() {
            return MUSA_SUCCESS;
        }
        let mut allocs = MOCK_ALLOCATIONS.lock().unwrap();
        if let Some(idx) = allocs.iter().position(|&p| p == devPtr as usize) {
            allocs.swap_remove(idx);
        }
        MUSA_SUCCESS
    }

    pub fn musaGetErrorString(_error: musaError_t) -> *const c_char {
        b"mock MUSA error\0".as_ptr() as *const c_char
    }

    pub unsafe fn musaGetLastError() -> musaError_t {
        MUSA_SUCCESS
    }
}

// ============================================================
// 统一导出
// ============================================================

#[cfg(musapy_mock_musa)]
pub use mock::*;
#[cfg(not(musapy_mock_musa))]
pub use real::*;

// ============================================================
// 高层辅助函数
// ============================================================

/// 检查 MUSA 调用结果，非成功则返回 MusapyError。
///
/// `context` 是调用场景描述，用于错误消息（如 "musaStreamCreate"）。
pub fn check_musa(err: musaError_t, context: &str) -> Result<()> {
    if err == MUSA_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe { musa_error_to_string(err) };
        Err(MusapyError::Stream(StreamError::MusaCallFailed(format!(
            "{}: {} (error code {})",
            context, msg, err
        ))))
    }
}

/// 将 MUSA 错误码转为字符串。
unsafe fn musa_error_to_string(err: musaError_t) -> String {
    unsafe {
        let s = musaGetErrorString(err);
        if s.is_null() {
            format!("unknown MUSA error {}", err)
        } else {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        }
    }
}

/// 检查最近一次 kernel launch 是否有错误（ADR L3-1 即时检测）。
///
/// 与 `check_musa` 不同，此处返回 `KernelError::LaunchFailed`，
/// 因为 kernel launch 错误属于 launch 层而非 stream 层。
pub fn check_last_kernel_launch(context: &str) -> Result<()> {
    let err = unsafe { musaGetLastError() };
    if err == MUSA_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe { musa_error_to_string(err) };
        Err(MusapyError::Kernel(KernelError::LaunchFailed(format!(
            "{}: {} (error code {})",
            context, msg, err
        ))))
    }
}

/// 查询设备数量（ADR L1-3）。
pub fn get_device_count() -> Result<i32> {
    let mut count: c_int = 0;
    unsafe {
        check_musa(musaGetDeviceCount(&mut count), "musaGetDeviceCount")?;
    }
    Ok(count)
}

/// 设置调用线程的当前设备（ADR L1-1）。
///
/// `musaMalloc`/`musaFree`（默认路径）和 stream 创建都绑定到当前设备，
/// 所以在 alloc / Stream::new 前必须调用此函数切换到目标设备。
pub fn set_device(device_id: i32) -> Result<()> {
    unsafe { check_musa(musaSetDevice(device_id), "musaSetDevice") }
}

/// 设备属性快照（ADR L1-3，P5.9 device_summary）。
#[derive(Clone, Debug)]
pub struct DeviceProperties {
    /// 设备名称（如 "MTT S4000"）。
    pub name: String,
    /// 计算能力主版本号。
    pub arch_major: i32,
    /// 计算能力次版本号。
    pub arch_minor: i32,
    /// 总显存（字节）。
    pub total_memory: usize,
    /// 空闲显存（字节）。
    pub free_memory: usize,
    /// 多处理器数量（CU 数）。
    pub multiprocessor_count: i32,
}

/// 查询指定设备的属性（ADR L1-3）。
///
/// 组合 `musaGetDeviceProperties`（名称）、`musaDeviceGetAttribute`（arch/CU 数）
/// 和 `musaMemGetInfo`（显存）的结果。
///
/// 调用前会自动 `musaSetDevice(device_id)`（musaMemGetInfo 绑定当前设备）。
pub fn get_device_properties(device_id: i32) -> Result<DeviceProperties> {
    // musaMemGetInfo 绑定当前设备，必须先 set
    set_device(device_id)?;

    // 1. 设备名称（musaGetDeviceProperties）
    let mut prop = std::mem::MaybeUninit::<MusaDeviceProp>::zeroed();
    unsafe {
        check_musa(
            musaGetDeviceProperties(prop.as_mut_ptr(), device_id),
            "musaGetDeviceProperties",
        )?;
    }
    let prop = unsafe { prop.assume_init() };
    let name = unsafe { CStr::from_ptr(prop.name.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    // 2. 计算能力 + CU 数（musaDeviceGetAttribute）
    let mut arch_major: c_int = 0;
    let mut arch_minor: c_int = 0;
    let mut mp_count: c_int = 0;
    unsafe {
        check_musa(
            musaDeviceGetAttribute(
                &mut arch_major,
                MUSA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR,
                device_id,
            ),
            "musaDeviceGetAttribute(major)",
        )?;
        check_musa(
            musaDeviceGetAttribute(
                &mut arch_minor,
                MUSA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR,
                device_id,
            ),
            "musaDeviceGetAttribute(minor)",
        )?;
        check_musa(
            musaDeviceGetAttribute(&mut mp_count, MUSA_DEV_ATTR_MULTIPROCESSOR_COUNT, device_id),
            "musaDeviceGetAttribute(mp_count)",
        )?;
    }

    // 3. 显存信息（musaMemGetInfo）
    let mut free_mem: usize = 0;
    let mut total_mem: usize = 0;
    unsafe {
        check_musa(
            musaMemGetInfo(&mut free_mem, &mut total_mem),
            "musaMemGetInfo",
        )?;
    }

    Ok(DeviceProperties {
        name,
        arch_major,
        arch_minor,
        total_memory: total_mem,
        free_memory: free_mem,
        multiprocessor_count: mp_count,
    })
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_device_count_works() {
        // 真实环境应返回 >= 1；mock 模式返回 1
        let count = get_device_count().unwrap();
        assert!(count >= 1, "expected at least 1 MUSA device, got {}", count);
    }

    #[test]
    fn set_device_works() {
        // 真实环境切换到设备 0；mock 模式 no-op 成功
        assert!(set_device(0).is_ok());
    }

    #[test]
    fn musa_malloc_free_sync_path() {
        // 默认路径（musaMalloc/musaFree）端到端：set_device → malloc → free。
        // 真实环境在 GPU 上分配/释放；mock 用 std::alloc。
        set_device(0).unwrap();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            check_musa(musaMalloc(&mut ptr, 256), "musaMalloc").unwrap();
        }
        assert!(!ptr.is_null(), "musaMalloc returned null pointer");
        unsafe {
            check_musa(musaFree(ptr), "musaFree").unwrap();
        }
    }

    #[test]
    fn check_musa_success() {
        assert!(check_musa(MUSA_SUCCESS, "test").is_ok());
    }

    #[test]
    fn check_musa_failure() {
        let result = check_musa(1, "test_call");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            MusapyError::Stream(StreamError::MusaCallFailed(msg)) => {
                assert!(msg.contains("test_call"));
            }
            _ => panic!("expected MusaCallFailed error"),
        }
    }

    #[test]
    fn get_device_properties_works() {
        // 真实环境返回真实设备名；mock 返回 "Mock MUSA GPU"
        let props = get_device_properties(0).unwrap();
        assert!(!props.name.is_empty());
        assert!(props.total_memory > 0);
        assert!(props.multiprocessor_count > 0);
    }
}
