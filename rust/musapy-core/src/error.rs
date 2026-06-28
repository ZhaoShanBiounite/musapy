use thiserror::Error;

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum MusapyError {
    // Device errors
    #[error("device not configured: call set_default_device() first")]
    DeviceNotConfigured,

    #[error("device mismatch: {0}")]
    DeviceMismatch(String),

    #[error("device unavailable: {0}")]
    DeviceUnavailable(String),

    // Dtype errors
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),

    // Shape errors
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    // Memory errors
    #[error("out of memory: {0}")]
    OutOfMemory(String),

    #[error("alias detected: buffer cannot be both input and output")]
    AliasDetected,

    // Stream errors
    #[error("poisoned stream: {0}")]
    PoisonedStream(String),

    #[error("sync cycle detected: {0}")]
    SyncCycle(String),

    // Kernel errors
    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),

    #[error("kernel execution failed: {0}")]
    KernelFailed(String),

    // Interop errors
    #[error("DLPack export failed: {0}")]
    DlpackExport(String),

    #[error("unsupported protocol: {0}")]
    UnsupportedProtocol(String),
}

pub type Result<T> = std::result::Result<T, MusapyError>;
