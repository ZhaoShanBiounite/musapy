use thiserror::Error;

// ── Device errors (L3-5: DeviceError category) ──

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

// ── Memory errors ──

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

// ── Top-level error (ADR L3-5 two-level hierarchy) ──

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

pub type Result<T> = std::result::Result<T, MusapyError>;
