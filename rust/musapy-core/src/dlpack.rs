//! DLPack 互操作基础结构（v0.4 Phase 1，ADR-004 004-D1/D2/D3，主 ADR L3-16~L3-20）
//!
//! 设计决策（004-D1）：
//!   - 系统无独立 `dlpack.h`（仅 torch 自带），**不引入外部依赖**——vendor 最小
//!     `repr(C)` 结构（DLPack v0.8 classic 版，capsule 兼容性最广），放
//!     musapy-core（FFI 基础设施层）；capsule 构造/解析在
//!     `musapy-python/src/interop.rs`（Phase 2）。
//!   - 设备码 `kDLMUSA = 100`（DLPack reserved 范围，主 ADR L3-16；
//!     稳定后向上游申请官方枚举值）。
//!
//! dtype 映射（004-D2）：
//!   - 导出：`Dtype::to_dlpack`——bool→kDLUInt/8（DLPack 无 bool 类别，
//!     对齐 PyTorch 惯例）、complex 位宽含分量双倍（c64→kDLComplex/64）、
//!     lanes 恒 1。
//!   - 导入：`dtype_from_dlpack`——kDLUInt/8 按 PyTorch 惯例映射 **Uint8**
//!     （bool round-trip 有损，调用方需 `astype('b1')` 转回）；`lanes≠1`
//!     一律拒绝（`InteropError::UnsupportedProtocol`）。
//!
//! 生命周期（004-D3）：导出侧 `DLManagedTensor.manager_ctx` 持 `Arc<Buffer>`
//! （经 `BufferRef::arc()`），deleter 负责回收——具体构造在 interop.rs。

use crate::device::Device;
use crate::dtype::Dtype;
use crate::error::{InteropError, Result};

// ============================================================
// 1. DLDataTypeCode（DLPack v0.8）
// ============================================================

/// 有符号整数。
pub const K_DL_INT: u8 = 0;
/// 无符号整数。
pub const K_DL_UINT: u8 = 1;
/// IEEE 浮点。
pub const K_DL_FLOAT: u8 = 2;
/// bfloat16。
pub const K_DL_BFLOAT: u8 = 4;
/// 复数（位宽 = 分量位宽 × 2）。
pub const K_DL_COMPLEX: u8 = 5;

// ============================================================
// 2. DLDeviceType
// ============================================================

/// CPU（主机内存）。
pub const K_DL_CPU: i32 = 1;
/// NVIDIA CUDA（v0.4 不支持导入，仅用于错误消息识别）。
pub const K_DL_CUDA: i32 = 2;
/// musapy 自定义设备码（主 ADR L3-16：reserved 范围取值 100）。
pub const K_DL_MUSA: i32 = 100;

// ============================================================
// 3. C 结构（repr(C)，与 dlpack.h v0.8 布局一致）
// ============================================================

/// DLPack 数据类型：(code, bits, lanes)。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType {
    /// 类型类别（K_DL_* 常量）。
    pub code: u8,
    /// 位宽（complex 为总位宽：分量 × 2）。
    pub bits: u8,
    /// SIMD lane 数（musapy 恒 1，≠1 拒绝）。
    pub lanes: u16,
}

/// DLPack 设备：(device_type, device_id)。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice {
    /// 设备类型码（K_DL_* 常量）。
    pub device_type: i32,
    /// 设备 ID（CPU 恒 0）。
    pub device_id: i32,
}

/// DLPack 张量描述符。
#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    /// 数据指针（经 `byte_offset` 偏移后的实际起始）。
    pub data: *mut std::ffi::c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    /// 形状（长度 = ndim）。
    pub shape: *mut i64,
    /// 元素单位步长；NULL 表示连续（等价 C-order 连续步长）。
    pub strides: *mut i64,
    /// 字节偏移（musapy 导出恒 0——offset 折进 data 指针）。
    pub byte_offset: u64,
}

/// 带生命周期管理的张量（capsule 载体）。
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    /// 所有者上下文（musapy 导出侧为 `Arc<Buffer>` 的 `into_raw` 指针）。
    pub manager_ctx: *mut std::ffi::c_void,
    /// 释放回调；消费方接管后必须调用恰好一次。
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

// ============================================================
// 4. Dtype 双向映射（004-D2）
// ============================================================

impl Dtype {
    /// 导出映射：musapy Dtype → DLDataType（004-D2）。
    ///
    /// bool→kDLUInt/8（无 bool 类别，PyTorch 惯例）；
    /// complex 位宽含分量双倍；lanes 恒 1。
    pub fn to_dlpack(self) -> DLDataType {
        let (code, bits) = match self {
            // bool 按 004-D2 映射到 kDLUInt/8
            Dtype::Bool => (K_DL_UINT, 8),
            Dtype::Int8 => (K_DL_INT, 8),
            Dtype::Int16 => (K_DL_INT, 16),
            Dtype::Int32 => (K_DL_INT, 32),
            Dtype::Int64 => (K_DL_INT, 64),
            Dtype::Uint8 => (K_DL_UINT, 8),
            Dtype::Uint16 => (K_DL_UINT, 16),
            Dtype::Uint32 => (K_DL_UINT, 32),
            Dtype::Uint64 => (K_DL_UINT, 64),
            Dtype::Float16 => (K_DL_FLOAT, 16),
            Dtype::Float32 => (K_DL_FLOAT, 32),
            Dtype::Float64 => (K_DL_FLOAT, 64),
            Dtype::Bfloat16 => (K_DL_BFLOAT, 16),
            Dtype::Complex64 => (K_DL_COMPLEX, 64),
            Dtype::Complex128 => (K_DL_COMPLEX, 128),
        };
        DLDataType {
            code,
            bits,
            lanes: 1,
        }
    }
}

/// 导入映射：DLDataType → musapy Dtype（004-D2）。
///
/// - `lanes≠1` 一律拒绝（向量化类型不在 v0.4 范围）。
/// - `kDLUInt/8` 映射 **Uint8**（PyTorch 惯例；bool 导出后经 round-trip
///   变 Uint8，属已知有损行为，调用方 `astype('b1')` 转回）。
/// - 未知 code/bits 组合返回 `InteropError::DlpackImport`（不 panic、
///   不静默降级，004-D5）。
pub fn dtype_from_dlpack(dt: DLDataType) -> Result<Dtype> {
    if dt.lanes != 1 {
        return Err(InteropError::UnsupportedProtocol(format!(
            "DLPack lanes={} not supported (musapy only supports lanes=1)",
            dt.lanes
        ))
        .into());
    }
    let dtype = match (dt.code, dt.bits) {
        (K_DL_INT, 8) => Dtype::Int8,
        (K_DL_INT, 16) => Dtype::Int16,
        (K_DL_INT, 32) => Dtype::Int32,
        (K_DL_INT, 64) => Dtype::Int64,
        // 004-D2：bool 经 kDLUInt/8 round-trip 后为 Uint8（有损，文档标注）
        (K_DL_UINT, 8) => Dtype::Uint8,
        (K_DL_UINT, 16) => Dtype::Uint16,
        (K_DL_UINT, 32) => Dtype::Uint32,
        (K_DL_UINT, 64) => Dtype::Uint64,
        (K_DL_FLOAT, 16) => Dtype::Float16,
        (K_DL_FLOAT, 32) => Dtype::Float32,
        (K_DL_FLOAT, 64) => Dtype::Float64,
        (K_DL_BFLOAT, 16) => Dtype::Bfloat16,
        (K_DL_COMPLEX, 64) => Dtype::Complex64,
        (K_DL_COMPLEX, 128) => Dtype::Complex128,
        (code, bits) => {
            return Err(InteropError::DlpackImport(format!(
                "unknown DLPack dtype: code={}, bits={} (supported: int/uint 8-64, \
                 float 16/32/64, bfloat16, complex 64/128)",
                code, bits
            ))
            .into());
        }
    };
    Ok(dtype)
}

// ============================================================
// 5. Device 双向映射
// ============================================================

impl Device {
    /// 导出映射：musapy Device → DLDevice。
    ///
    /// Cpu→(kDLCPU, 0)；Musa(id)→(kDLMUSA, id)。
    pub fn to_dlpack(&self) -> DLDevice {
        match self {
            Device::Cpu => DLDevice {
                device_type: K_DL_CPU,
                device_id: 0,
            },
            Device::Musa(id) => DLDevice {
                device_type: K_DL_MUSA,
                device_id: *id as i32,
            },
        }
    }
}

/// 导入映射：DLDevice → musapy Device（004-D5）。
///
/// v0.4 仅接受 kDLCPU 与 kDLMUSA（L3-20：round-trip 限 musapy 内部）；
/// 其他设备类型（kDLCUDA 等）返回 `InteropError::UnsupportedProtocol`，
/// 错误消息标注设备码便于诊断（含 torch_musa 探针场景）。
pub fn device_from_dlpack(dev: DLDevice) -> Result<Device> {
    match dev.device_type {
        K_DL_CPU => Ok(Device::Cpu),
        K_DL_MUSA => {
            if dev.device_id < 0 {
                return Err(InteropError::DlpackImport(format!(
                    "invalid kDLMUSA device id: {}",
                    dev.device_id
                ))
                .into());
            }
            Ok(Device::Musa(dev.device_id as u32))
        }
        other => Err(InteropError::UnsupportedProtocol(format!(
            "unsupported DLPack device type: {} (v0.4 accepts kDLCPU={} and kDLMUSA={} only)",
            other, K_DL_CPU, K_DL_MUSA
        ))
        .into()),
    }
}

// ============================================================
// 6. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    // ── 结构布局（与 dlpack.h v0.8 的 ABI 一致性前提）──

    #[test]
    fn test_struct_sizes() {
        // dlpack.h v0.8：DLDataType=4B，DLDevice=8B，
        // DLTensor=48B（x86_64），DLManagedTensor=64B（x86_64，niche 优化 Option<fn_ptr>=8B）。
        assert_eq!(std::mem::size_of::<DLDataType>(), 4);
        assert_eq!(std::mem::size_of::<DLDevice>(), 8);
        assert_eq!(std::mem::size_of::<DLTensor>(), 48);
        assert_eq!(std::mem::size_of::<DLManagedTensor>(), 64);
    }

    #[test]
    fn test_struct_field_offsets() {
        // capsule 消费方按标准偏移取值，布局错一位即 segfault，故显式断言。
        assert_eq!(offset_of!(DLManagedTensor, dl_tensor), 0);
        assert_eq!(offset_of!(DLManagedTensor, manager_ctx), 48);
        assert_eq!(offset_of!(DLManagedTensor, deleter), 56);
        assert_eq!(offset_of!(DLTensor, data), 0);
        assert_eq!(offset_of!(DLTensor, device), 8);
        assert_eq!(offset_of!(DLTensor, ndim), 16);
        assert_eq!(offset_of!(DLTensor, dtype), 20);
        assert_eq!(offset_of!(DLTensor, shape), 24);
        assert_eq!(offset_of!(DLTensor, strides), 32);
        assert_eq!(offset_of!(DLTensor, byte_offset), 40);
    }

    // ── dtype 导出映射矩阵（004-D2）──

    #[test]
    fn test_dtype_export_matrix() {
        let cases = [
            (Dtype::Bool, K_DL_UINT, 8), // bool→kDLUInt/8（有损，导入回 Uint8）
            (Dtype::Int8, K_DL_INT, 8),
            (Dtype::Int16, K_DL_INT, 16),
            (Dtype::Int32, K_DL_INT, 32),
            (Dtype::Int64, K_DL_INT, 64),
            (Dtype::Uint8, K_DL_UINT, 8),
            (Dtype::Uint16, K_DL_UINT, 16),
            (Dtype::Uint32, K_DL_UINT, 32),
            (Dtype::Uint64, K_DL_UINT, 64),
            (Dtype::Float16, K_DL_FLOAT, 16),
            (Dtype::Float32, K_DL_FLOAT, 32),
            (Dtype::Float64, K_DL_FLOAT, 64),
            (Dtype::Bfloat16, K_DL_BFLOAT, 16),
            (Dtype::Complex64, K_DL_COMPLEX, 64), // 分量位宽 × 2
            (Dtype::Complex128, K_DL_COMPLEX, 128),
        ];
        for (dtype, code, bits) in cases {
            let dl = dtype.to_dlpack();
            assert_eq!(dl.code, code, "code mismatch for {:?}", dtype);
            assert_eq!(dl.bits, bits, "bits mismatch for {:?}", dtype);
            assert_eq!(dl.lanes, 1, "lanes must be 1 for {:?}", dtype);
        }
    }

    #[test]
    fn test_dtype_import_roundtrip() {
        // 除 Bool 外（有损映射到 Uint8）全部 round-trip 一致。
        for dtype in [
            Dtype::Int8,
            Dtype::Int16,
            Dtype::Int32,
            Dtype::Int64,
            Dtype::Uint8,
            Dtype::Uint16,
            Dtype::Uint32,
            Dtype::Uint64,
            Dtype::Float16,
            Dtype::Float32,
            Dtype::Float64,
            Dtype::Bfloat16,
            Dtype::Complex64,
            Dtype::Complex128,
        ] {
            assert_eq!(
                dtype_from_dlpack(dtype.to_dlpack()).unwrap(),
                dtype,
                "round-trip failed for {:?}",
                dtype
            );
        }
        // bool round-trip 有损：导出→导入变 Uint8（004-D2 显式约定）
        assert_eq!(
            dtype_from_dlpack(Dtype::Bool.to_dlpack()).unwrap(),
            Dtype::Uint8
        );
    }

    #[test]
    fn test_dtype_import_rejects_lanes() {
        let dt = DLDataType {
            code: K_DL_FLOAT,
            bits: 32,
            lanes: 4,
        };
        let err = dtype_from_dlpack(dt).unwrap_err();
        match err {
            crate::error::MusapyError::Interop(InteropError::UnsupportedProtocol(msg)) => {
                assert!(
                    msg.contains("lanes"),
                    "message should mention lanes: {}",
                    msg
                );
            }
            other => panic!("expected UnsupportedProtocol, got {:?}", other),
        }
    }

    #[test]
    fn test_dtype_import_rejects_unknown() {
        // kDLFloat/8 不存在；code=99 为非法类别；kDLBfloat/32 不存在。
        for (code, bits) in [(K_DL_FLOAT, 8), (99u8, 32), (K_DL_BFLOAT, 32)] {
            let dt = DLDataType {
                code,
                bits,
                lanes: 1,
            };
            let err = dtype_from_dlpack(dt).unwrap_err();
            assert!(
                matches!(
                    err,
                    crate::error::MusapyError::Interop(InteropError::DlpackImport(_))
                ),
                "expected DlpackImport for code={} bits={}, got {:?}",
                code,
                bits,
                err
            );
        }
    }

    // ── device 双向映射 ──

    #[test]
    fn test_device_export() {
        assert_eq!(
            Device::Cpu.to_dlpack(),
            DLDevice {
                device_type: K_DL_CPU,
                device_id: 0
            }
        );
        assert_eq!(
            Device::Musa(0).to_dlpack(),
            DLDevice {
                device_type: K_DL_MUSA,
                device_id: 0
            }
        );
        assert_eq!(
            Device::Musa(3).to_dlpack(),
            DLDevice {
                device_type: K_DL_MUSA,
                device_id: 3
            }
        );
    }

    #[test]
    fn test_device_import() {
        assert_eq!(
            device_from_dlpack(DLDevice {
                device_type: K_DL_CPU,
                device_id: 0
            })
            .unwrap(),
            Device::Cpu
        );
        assert_eq!(
            device_from_dlpack(DLDevice {
                device_type: K_DL_MUSA,
                device_id: 2
            })
            .unwrap(),
            Device::Musa(2)
        );
    }

    #[test]
    fn test_device_import_rejects_foreign() {
        // kDLCUDA 等其他设备类型拒绝（v0.4 仅 musapy 内部 round-trip）。
        let err = device_from_dlpack(DLDevice {
            device_type: K_DL_CUDA,
            device_id: 0,
        })
        .unwrap_err();
        match err {
            crate::error::MusapyError::Interop(InteropError::UnsupportedProtocol(msg)) => {
                assert!(msg.contains("kDLMUSA"), "message should guide: {}", msg);
            }
            other => panic!("expected UnsupportedProtocol, got {:?}", other),
        }
    }

    #[test]
    fn test_device_import_rejects_negative_id() {
        let err = device_from_dlpack(DLDevice {
            device_type: K_DL_MUSA,
            device_id: -1,
        })
        .unwrap_err();
        assert!(matches!(
            err,
            crate::error::MusapyError::Interop(InteropError::DlpackImport(_))
        ));
    }
}
