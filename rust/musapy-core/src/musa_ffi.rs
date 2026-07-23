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
use crate::error::{MusapyError, Result, StreamError};
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

/// 内存池是否支持（stream-ordered alloc/free）。
/// ⚠️ CUDA 值为 115，MUSA 对标应一致 —— 请对照 driver_types.h 确认。
pub const MUSA_DEV_ATTR_MEMORY_POOLS_SUPPORTED: musaDeviceAttr = 115;

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
        pub fn musaGetDevice(device: *mut c_int) -> musaError_t;

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

        // --- Error（ADR L3-1）---
        pub fn musaGetErrorString(error: musaError_t) -> *const c_char;
    }
}

// ============================================================
// Mock 模式 stub（CI/无 GPU 开发机）
// ============================================================

#[cfg(musapy_mock_musa)]
mod mock {
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
        // mock: 内存池支持
        *value = if attr == MUSA_DEV_ATTR_MEMORY_POOLS_SUPPORTED {
            1
        } else {
            0
        };
        MUSA_SUCCESS
    }

    pub unsafe fn musaSetDevice(_device: c_int) -> musaError_t {
        MUSA_SUCCESS
    }

    pub unsafe fn musaGetDevice(device: *mut c_int) -> musaError_t {
        if device.is_null() {
            return 1;
        }
        *device = 0;
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
}

// ============================================================
// 统一导出
// ============================================================

#[cfg(not(musapy_mock_musa))]
pub use real::*;
#[cfg(musapy_mock_musa)]
pub use mock::*;

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

/// 探测设备是否支持内存池（stream-ordered alloc/free，ADR L3-9, L3-11）。
///
/// 返回 `true` 表示该设备的 MUSA Runtime 有 `musaMallocAsync`/`musaFreeAsync`
/// 真实实现（5.x+），可启用 `stream-ordered` feature 走全量 stream-ordered 方案。
/// 返回 `false` 表示需走 deferred-free 默认路径（3.x/4.x）。
///
/// 注意：3.x/4.x 头文件里 `musaMallocAsync` 是 static inline 包装，
/// libmusart.so 不导出符号，所以即使此探测返回 true 也不代表能链接——
/// feature gate 是编译期硬保证，此 probe 是运行期双重保险。
///
/// 探测失败（属性查询返回错误）时保守返回 `false`。
pub fn probe_memory_pools_supported(device_id: i32) -> bool {
    let mut value: c_int = 0;
    let err = unsafe {
        musaDeviceGetAttribute(&mut value, MUSA_DEV_ATTR_MEMORY_POOLS_SUPPORTED, device_id)
    };
    err == MUSA_SUCCESS && value != 0
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
    fn probe_memory_pools_supported_works() {
        // 只要不 panic 就行；真实环境可能 true 或 false，mock 返回 true
        let _supported = probe_memory_pools_supported(0);
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
}