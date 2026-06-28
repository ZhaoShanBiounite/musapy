use thiserror::Error;

// ── Device errors (ADR L3-5, L3-6: DeviceError category) ──

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    #[error("device not configured: call set_default_device() first")]
    NotConfigured,
    #[error("device mismatch: {0}")]
    Mismatch(String),
    #[error("device unavailable: {0}")]
    Unavailable(String),
}

// ── Dtype errors ──

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum DtypeError {
    #[error("unsupported dtype: {0}")]
    Unsupported(String),
}

// ── Shape errors ──

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ShapeError {
    #[error("shape mismatch: {0}")]
    Mismatch(String),
}

// ── Memory errors (ADR L3-7: OutOfMemory NOT inheriting builtin MemoryError) ──

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum MemoryError {
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    #[error("alias detected: buffer cannot be both input and output")]
    AliasDetected,
}

// ── Stream errors ──

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum StreamError {
    #[error("poisoned stream: {0}")]
    Poisoned(String),
    #[error("sync cycle detected: {0}")]
    SyncCycle(String),
}

// ── Kernel errors ──

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum KernelError {
    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),
    #[error("kernel execution failed: {0}")]
    ExecutionFailed(String),
}

// ── Interop errors ──

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum InteropError {
    #[error("DLPack export failed: {0}")]
    DlpackExport(String),
    #[error("unsupported protocol: {0}")]
    UnsupportedProtocol(String),
}

// ── Top-level error (ADR L3-5: two-level shallow hierarchy) ──

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum MusapyError {
    #[error("{0}")]
    Device(DeviceError),
    #[error("{0}")]
    Dtype(DtypeError),
    #[error("{0}")]
    Shape(ShapeError),
    #[error("{0}")]
    Memory(MemoryError),
    #[error("{0}")]
    Stream(StreamError),
    #[error("{0}")]
    Kernel(KernelError),
    #[error("{0}")]
    Interop(InteropError),
}

// ── From impls for ergonomic conversions ──

impl From<DeviceError> for MusapyError {
    fn from(e: DeviceError) -> Self {
        MusapyError::Device(e)
    }
}

impl From<DtypeError> for MusapyError {
    fn from(e: DtypeError) -> Self {
        MusapyError::Dtype(e)
    }
}

impl From<ShapeError> for MusapyError {
    fn from(e: ShapeError) -> Self {
        MusapyError::Shape(e)
    }
}

impl From<MemoryError> for MusapyError {
    fn from(e: MemoryError) -> Self {
        MusapyError::Memory(e)
    }
}

impl From<StreamError> for MusapyError {
    fn from(e: StreamError) -> Self {
        MusapyError::Stream(e)
    }
}

impl From<KernelError> for MusapyError {
    fn from(e: KernelError) -> Self {
        MusapyError::Kernel(e)
    }
}

impl From<InteropError> for MusapyError {
    fn from(e: InteropError) -> Self {
        MusapyError::Interop(e)
    }
}

/// musapy 核心 Result 类型别名
pub type Result<T> = std::result::Result<T, MusapyError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = MusapyError::Device(DeviceError::NotConfigured);
        assert_eq!(
            e.to_string(),
            "device not configured: call set_default_device() first"
        );
    }

    #[test]
    fn test_from_conversion() {
        let e: MusapyError = DeviceError::Mismatch("expected musa:0, got cpu".into()).into();
        match e {
            MusapyError::Device(DeviceError::Mismatch(msg)) => {
                assert_eq!(msg, "expected musa:0, got cpu");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_result_alias() {
        let ok: Result<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: Result<i32> = Err(MusapyError::Memory(MemoryError::OutOfMemory(
            "24GB requested".into(),
        )));
        assert!(err.is_err());
    }

    #[test]
    fn test_all_variants_constructible() {
        // 确保所有 variant 都能构造（防止未来重构漏掉）
        let _ = MusapyError::Device(DeviceError::NotConfigured);
        let _ = MusapyError::Device(DeviceError::Mismatch("x".into()));
        let _ = MusapyError::Device(DeviceError::Unavailable("x".into()));
        let _ = MusapyError::Dtype(DtypeError::Unsupported("x".into()));
        let _ = MusapyError::Shape(ShapeError::Mismatch("x".into()));
        let _ = MusapyError::Memory(MemoryError::OutOfMemory("x".into()));
        let _ = MusapyError::Memory(MemoryError::AliasDetected);
        let _ = MusapyError::Stream(StreamError::Poisoned("x".into()));
        let _ = MusapyError::Stream(StreamError::SyncCycle("x".into()));
        let _ = MusapyError::Kernel(KernelError::LaunchFailed("x".into()));
        let _ = MusapyError::Kernel(KernelError::ExecutionFailed("x".into()));
        let _ = MusapyError::Interop(InteropError::DlpackExport("x".into()));
        let _ = MusapyError::Interop(InteropError::UnsupportedProtocol("x".into()));
    }
}
