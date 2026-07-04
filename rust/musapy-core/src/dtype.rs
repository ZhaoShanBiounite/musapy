//! 数据类型与类型提升（ADR L1-4, L1-5, L1-14）
//!
//! 职责：
//!   1. Dtype 枚举：15 种数据类型
//!   2. 类型属性：category / precision_width / element_size / name
//!   3. promote()：JAX 风格类型提升（ADR L1-5）+ GPU narrow 策略（ADR L1-14）
//!
//! 类型提升规则（ADR L1-14）：
//!   - all_gpu=false：使用 JAX 标准提升表（正确性优先）
//!   - all_gpu=true：同类别取最窄输入（性能优先），跨类别用 JAX 规则
//!   - 冲突对 f16+bf16：无论哪种模式都升到 f32（避免精度损失）

use crate::error::{DtypeError, Result};
use std::fmt;

// ============================================================
// 1. Dtype 枚举（ADR L1-4）
// ============================================================

/// musapy 支持的 15 种数据类型。
///
/// 科学计算用户需要 complex（FFT、信号处理）和 bfloat16（数值实验对比），
/// 缺失会迫使 NumPy fallback，体验割裂（ADR L1-4 rationale）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Complex64,
    Complex128,
}

impl Default for Dtype {
    /// 默认 dtype 是 float32（ADR L0-7：启动默认 ms.float32）。
    fn default() -> Self {
        Dtype::Float32
    }
}

// 类别常量（内部使用）
const CAT_BOOL: u8 = 0;
const CAT_INT: u8 = 1;
const CAT_UINT: u8 = 2;
const CAT_FLOAT: u8 = 3;
const CAT_COMPLEX: u8 = 4;

impl Dtype {
    /// 类型类别。
    ///
    /// bool / int / uint / float / complex 五类。
    /// 同类别在 GPU narrow 策略下取最窄输入（ADR L1-14）。
    pub fn category(self) -> u8 {
        match self {
            Dtype::Bool => CAT_BOOL,
            Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 => CAT_INT,
            Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 | Dtype::Uint64 => CAT_UINT,
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16 => CAT_FLOAT,
            Dtype::Complex64 | Dtype::Complex128 => CAT_COMPLEX,
        }
    }

    /// 精度位宽（complex 用其分量的位宽）。
    ///
    /// 用于类型提升计算。Complex64 = 两个 float32，精度位宽 32。
    pub fn precision_width(self) -> u16 {
        match self {
            Dtype::Bool => 8,
            Dtype::Int8 | Dtype::Uint8 => 8,
            Dtype::Int16 | Dtype::Uint16 | Dtype::Float16 | Dtype::Bfloat16 => 16,
            Dtype::Int32 | Dtype::Uint32 | Dtype::Float32 | Dtype::Complex64 => 32,
            Dtype::Int64 | Dtype::Uint64 | Dtype::Float64 | Dtype::Complex128 => 64,
        }
    }

    /// 单个元素占用的字节数。
    ///
    /// 用于 Buffer 分配大小计算。
    pub fn element_size(self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::Int8 | Dtype::Uint8 => 1,
            Dtype::Int16 | Dtype::Uint16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Int32 | Dtype::Uint32 | Dtype::Float32 => 4,
            Dtype::Int64 | Dtype::Uint64 | Dtype::Float64 | Dtype::Complex64 => 8,
            Dtype::Complex128 => 16,
        }
    }

    /// 字符串名称（用于 __repr__ 和调试输出）。
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bool => "bool",
            Dtype::Int8 => "int8",
            Dtype::Int16 => "int16",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Uint8 => "uint8",
            Dtype::Uint16 => "uint16",
            Dtype::Uint32 => "uint32",
            Dtype::Uint64 => "uint64",
            Dtype::Float16 => "float16",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "float64",
            Dtype::Bfloat16 => "bfloat16",
            Dtype::Complex64 => "complex64",
            Dtype::Complex128 => "complex128",
        }
    }

    /// 是否为浮点类型（含 bfloat16）。
    pub fn is_floating(self) -> bool {
        matches!(
            self,
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16
        )
    }

    /// 是否为整数类型（含 bool）。
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Dtype::Bool
                | Dtype::Int8
                | Dtype::Int16
                | Dtype::Int32
                | Dtype::Int64
                | Dtype::Uint8
                | Dtype::Uint16
                | Dtype::Uint32
                | Dtype::Uint64
        )
    }

    /// 是否为复数类型。
    pub fn is_complex(self) -> bool {
        matches!(self, Dtype::Complex64 | Dtype::Complex128)
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================
// 2. 辅助函数：按位宽查找 dtype
// ============================================================

/// 返回指定位宽的 int 类型。
fn int_of_width(w: u16) -> Dtype {
    match w {
        8 => Dtype::Int8,
        16 => Dtype::Int16,
        32 => Dtype::Int32,
        64 => Dtype::Int64,
        _ => unreachable!("int width must be 8/16/32/64, got {}", w),
    }
}

/// 返回指定位宽的 float 类型。
///
/// 位宽 16 时返回 Float16（不是 Bfloat16）。
fn float_of_width(w: u16) -> Dtype {
    match w {
        16 => Dtype::Float16,
        32 => Dtype::Float32,
        64 => Dtype::Float64,
        _ => unreachable!("float width must be 16/32/64, got {}", w),
    }
}

/// 返回指定位宽的 complex 类型（位宽指分量位宽）。
fn complex_of_width(w: u16) -> Dtype {
    match w {
        32 => Dtype::Complex64,
        64 => Dtype::Complex128,
        _ => unreachable!("complex component width must be 32/64, got {}", w),
    }
}

// ============================================================
// 3. JAX 标准类型提升（ADR L1-5）
// ============================================================

/// JAX 风格类型提升（ADR L1-5）。
///
/// 规则总结：
/// - bool + X = X
/// - 同类别（int+int, uint+uint, float+float, complex+complex）：取更宽
/// - float 同宽不同类型（f16+bf16）：升到 f32
/// - int + uint：结果总是 int，位宽足以容纳两者（同宽升一级）
/// - exact(int/uint) + float：float 至少 f32，位宽 = max(int_w, float_w, 32)
/// - exact + complex：complex，分量位宽 = max(int_w, complex_w, 32)
/// - float + complex：complex，分量位宽 = max(float_w, complex_w)
fn jax_promote(a: Dtype, b: Dtype) -> Result<Dtype> {
    if a == b {
        return Ok(a);
    }
    if a == Dtype::Bool {
        return Ok(b);
    }
    if b == Dtype::Bool {
        return Ok(a);
    }

    let (ca, pa) = (a.category(), a.precision_width());
    let (cb, pb) = (b.category(), b.precision_width());

    // 同类别
    if ca == cb {
        // float 同宽不同类型冲突（f16 + bf16）
        if ca == CAT_FLOAT && pa == pb {
            return Ok(Dtype::Float32);
        }
        // 否则取更宽
        return Ok(if pa >= pb { a } else { b });
    }

    // int + uint（不同类别，但都是 exact 整数）
    if (ca == CAT_INT && cb == CAT_UINT) || (ca == CAT_UINT && cb == CAT_INT) {
        let int_w = if ca == CAT_INT { pa } else { pb };
        let uint_w = if ca == CAT_INT { pb } else { pa };
        let target = if int_w > uint_w {
            // int 更宽，能容纳 uint
            int_w
        } else {
            // int <= uint：升一级（同宽升一级，int 更窄也升到 uint 的两倍）
            (uint_w * 2).min(64)
        };
        return Ok(int_of_width(target));
    }

    // exact(int/uint) + float
    let exact_float = |exact_w: u16, float_w: u16| -> Dtype {
        let target = exact_w.max(float_w).max(32);
        float_of_width(target)
    };
    if (ca == CAT_INT || ca == CAT_UINT) && cb == CAT_FLOAT {
        return Ok(exact_float(pa, pb));
    }
    if (cb == CAT_INT || cb == CAT_UINT) && ca == CAT_FLOAT {
        return Ok(exact_float(pb, pa));
    }

    // exact(int/uint) + complex
    let exact_complex = |exact_w: u16, complex_w: u16| -> Dtype {
        let target = exact_w.max(complex_w).max(32);
        complex_of_width(target)
    };
    if (ca == CAT_INT || ca == CAT_UINT) && cb == CAT_COMPLEX {
        return Ok(exact_complex(pa, pb));
    }
    if (cb == CAT_INT || cb == CAT_UINT) && ca == CAT_COMPLEX {
        return Ok(exact_complex(pb, pa));
    }

    // float + complex
    if ca == CAT_FLOAT && cb == CAT_COMPLEX {
        let target = pa.max(pb);
        return Ok(complex_of_width(target));
    }
    if ca == CAT_COMPLEX && cb == CAT_FLOAT {
        let target = pa.max(pb);
        return Ok(complex_of_width(target));
    }

    // 理论上不会到达（所有类别组合已覆盖）
    Err(DtypeError::Unsupported(format!(
        "cannot promote {} with {}",
        a, b
    )).into())
}

// ============================================================
// 4. 公开 API：promote（ADR L1-5 + L1-14）
// ============================================================

/// 类型提升（ADR L1-5, L1-14）。
///
/// `all_gpu=false` 时使用 JAX 标准提升表（正确性优先）。
/// `all_gpu=true` 时使用 GPU narrow 优先策略（性能优先）：
/// - 同类别（int+int, uint+uint, float+float, complex+complex）：取最窄输入
/// - 冲突对 f16+bf16：升到 f32（避免精度损失）
/// - 跨类别（int+uint, int+float, float+complex 等）：用 JAX 规则
///
/// 详见 ADR L1-14 的扩展表。
pub fn promote(a: Dtype, b: Dtype, all_gpu: bool) -> Result<Dtype> {
    if a == b {
        return Ok(a);
    }

    let jax = jax_promote(a, b)?;

    if !all_gpu {
        return Ok(jax);
    }

    // GPU narrow 策略
    let (ca, pa) = (a.category(), a.precision_width());
    let (cb, pb) = (b.category(), b.precision_width());

    if ca != cb {
        // 跨类别：用 JAX 规则（含 int+uint 的溢出保护）
        return Ok(jax);
    }

    // 同类别
    // 冲突对：f16 + bf16 → f32（JAX 结果）
    let is_float_conflict = ca == CAT_FLOAT
        && pa == pb
        && ((a == Dtype::Float16 && b == Dtype::Bfloat16)
            || (a == Dtype::Bfloat16 && b == Dtype::Float16));
    if is_float_conflict {
        return Ok(jax); // f32
    }

    // 取更窄的输入
    Ok(if pa <= pb { a } else { b })
}

// ============================================================
// 5. 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Dtype 属性 ---

    #[test]
    fn default_is_float32() {
        assert_eq!(Dtype::default(), Dtype::Float32);
    }

    #[test]
    fn category_correct() {
        assert_eq!(Dtype::Bool.category(), CAT_BOOL);
        assert_eq!(Dtype::Int32.category(), CAT_INT);
        assert_eq!(Dtype::Uint32.category(), CAT_UINT);
        assert_eq!(Dtype::Bfloat16.category(), CAT_FLOAT);
        assert_eq!(Dtype::Complex64.category(), CAT_COMPLEX);
    }

    #[test]
    fn precision_width_correct() {
        assert_eq!(Dtype::Bool.precision_width(), 8);
        assert_eq!(Dtype::Int8.precision_width(), 8);
        assert_eq!(Dtype::Float16.precision_width(), 16);
        assert_eq!(Dtype::Bfloat16.precision_width(), 16);
        assert_eq!(Dtype::Float32.precision_width(), 32);
        assert_eq!(Dtype::Complex64.precision_width(), 32); // 分量位宽
        assert_eq!(Dtype::Complex128.precision_width(), 64);
    }

    #[test]
    fn element_size_correct() {
        assert_eq!(Dtype::Bool.element_size(), 1);
        assert_eq!(Dtype::Int8.element_size(), 1);
        assert_eq!(Dtype::Float16.element_size(), 2);
        assert_eq!(Dtype::Bfloat16.element_size(), 2);
        assert_eq!(Dtype::Float32.element_size(), 4);
        assert_eq!(Dtype::Int64.element_size(), 8);
        assert_eq!(Dtype::Complex64.element_size(), 8); // 2 * f32
        assert_eq!(Dtype::Complex128.element_size(), 16); // 2 * f64
    }

    #[test]
    fn name_correct() {
        assert_eq!(Dtype::Bool.name(), "bool");
        assert_eq!(Dtype::Float32.name(), "float32");
        assert_eq!(Dtype::Bfloat16.name(), "bfloat16");
        assert_eq!(Dtype::Complex128.name(), "complex128");
    }

    #[test]
    fn is_floating_integer_complex() {
        assert!(Dtype::Float32.is_floating());
        assert!(Dtype::Bfloat16.is_floating());
        assert!(!Dtype::Int32.is_floating());

        assert!(Dtype::Int32.is_integer());
        assert!(Dtype::Bool.is_integer());
        assert!(!Dtype::Float32.is_integer());

        assert!(Dtype::Complex64.is_complex());
        assert!(!Dtype::Float32.is_complex());
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(Dtype::Float32.to_string(), "float32");
        assert_eq!(Dtype::Bfloat16.to_string(), "bfloat16");
    }

    // --- promote: 同类型 ---

    #[test]
    fn promote_same_type() {
        assert_eq!(promote(Dtype::Float32, Dtype::Float32, false).unwrap(), Dtype::Float32);
        assert_eq!(promote(Dtype::Int64, Dtype::Int64, true).unwrap(), Dtype::Int64);
    }

    // --- promote: bool ---

    #[test]
    fn promote_bool_with_anything() {
        assert_eq!(promote(Dtype::Bool, Dtype::Float32, false).unwrap(), Dtype::Float32);
        assert_eq!(promote(Dtype::Int64, Dtype::Bool, false).unwrap(), Dtype::Int64);
        assert_eq!(promote(Dtype::Bool, Dtype::Complex128, false).unwrap(), Dtype::Complex128);
    }

    // --- promote: ADR L1-14 扩展表（JAX 模式，all_gpu=false）---

    #[test]
    fn jax_float_narrow_to_wide() {
        // f16 + f32 → f32（JAX 取宽）
        assert_eq!(promote(Dtype::Float16, Dtype::Float32, false).unwrap(), Dtype::Float32);
        // bf16 + f32 → f32
        assert_eq!(promote(Dtype::Bfloat16, Dtype::Float32, false).unwrap(), Dtype::Float32);
    }

    #[test]
    fn jax_f16_plus_bf16_is_f32() {
        // f16 + bf16 → f32（同宽冲突）
        assert_eq!(promote(Dtype::Float16, Dtype::Bfloat16, false).unwrap(), Dtype::Float32);
        assert_eq!(promote(Dtype::Bfloat16, Dtype::Float16, false).unwrap(), Dtype::Float32);
    }

    #[test]
    fn jax_float_int() {
        // f32 + i32 → f32
        assert_eq!(promote(Dtype::Float32, Dtype::Int32, false).unwrap(), Dtype::Float32);
        // f16 + i8 → f32（float 至少 f32）
        assert_eq!(promote(Dtype::Float16, Dtype::Int8, false).unwrap(), Dtype::Float32);
        // f64 + i64 → f64
        assert_eq!(promote(Dtype::Float64, Dtype::Int64, false).unwrap(), Dtype::Float64);
    }

    #[test]
    fn jax_int_signed_unsigned() {
        // i32 + u32 → i64（signed+unsigned 同宽升一级，防溢出）
        assert_eq!(promote(Dtype::Int32, Dtype::Uint32, false).unwrap(), Dtype::Int64);
        // i8 + u8 → i16
        assert_eq!(promote(Dtype::Int8, Dtype::Uint8, false).unwrap(), Dtype::Int16);
        // i32 + u8 → i32（int 更宽，能容纳）
        assert_eq!(promote(Dtype::Int32, Dtype::Uint8, false).unwrap(), Dtype::Int32);
        // i8 + u32 → i64（uint 更宽，升到 i64）
        assert_eq!(promote(Dtype::Int8, Dtype::Uint32, false).unwrap(), Dtype::Int64);
    }

    #[test]
    fn jax_complex() {
        // f32 + c64 → c64
        assert_eq!(promote(Dtype::Float32, Dtype::Complex64, false).unwrap(), Dtype::Complex64);
        // f64 + c64 → c128
        assert_eq!(promote(Dtype::Float64, Dtype::Complex64, false).unwrap(), Dtype::Complex128);
        // c64 + c128 → c128
        assert_eq!(promote(Dtype::Complex64, Dtype::Complex128, false).unwrap(), Dtype::Complex128);
        // i32 + c64 → c64
        assert_eq!(promote(Dtype::Int32, Dtype::Complex64, false).unwrap(), Dtype::Complex64);
    }

    // --- promote: ADR L1-14 扩展表（GPU narrow 模式，all_gpu=true）---

    #[test]
    fn gpu_f16_plus_f32_is_f16() {
        // f16 + f32 → f16（narrow priority）
        assert_eq!(promote(Dtype::Float16, Dtype::Float32, true).unwrap(), Dtype::Float16);
        assert_eq!(promote(Dtype::Float32, Dtype::Float16, true).unwrap(), Dtype::Float16);
    }

    #[test]
    fn gpu_bf16_plus_f32_is_bf16() {
        // bf16 + f32 → bf16（narrow priority）
        assert_eq!(promote(Dtype::Bfloat16, Dtype::Float32, true).unwrap(), Dtype::Bfloat16);
    }

    #[test]
    fn gpu_f16_plus_bf16_is_f32() {
        // f16 + bf16 → f32（同宽冲突 → JAX）
        assert_eq!(promote(Dtype::Float16, Dtype::Bfloat16, true).unwrap(), Dtype::Float32);
    }

    #[test]
    fn gpu_f32_plus_f64_is_f32() {
        // f32 + f64 → f32（narrow priority）
        assert_eq!(promote(Dtype::Float32, Dtype::Float64, true).unwrap(), Dtype::Float32);
    }

    #[test]
    fn gpu_f32_plus_i32_is_f32() {
        // f32 + i32 → f32（跨类，JAX）
        assert_eq!(promote(Dtype::Float32, Dtype::Int32, true).unwrap(), Dtype::Float32);
    }

    #[test]
    fn gpu_i32_plus_i64_is_i32() {
        // i32 + i64 → i32（int narrow priority）
        assert_eq!(promote(Dtype::Int32, Dtype::Int64, true).unwrap(), Dtype::Int32);
    }

    #[test]
    fn gpu_i32_plus_u32_is_i64() {
        // i32 + u32 → i64（跨类 signed+unsigned，JAX 溢出保护）
        assert_eq!(promote(Dtype::Int32, Dtype::Uint32, true).unwrap(), Dtype::Int64);
    }

    #[test]
    fn gpu_f32_plus_complex64_is_complex64() {
        // f32 + c64 → c64（跨类，JAX narrow）
        assert_eq!(promote(Dtype::Float32, Dtype::Complex64, true).unwrap(), Dtype::Complex64);
    }

    #[test]
    fn gpu_complex64_plus_complex128_is_complex64() {
        // c64 + c128 → c64（complex narrow priority）
        assert_eq!(promote(Dtype::Complex64, Dtype::Complex128, true).unwrap(), Dtype::Complex64);
    }

    #[test]
    fn gpu_bool_plus_f32_is_f32() {
        // bool + f32 → f32（跨类，JAX）
        assert_eq!(promote(Dtype::Bool, Dtype::Float32, true).unwrap(), Dtype::Float32);
    }

    // --- promote: 对称性 ---

    #[test]
    fn promote_is_symmetric_jax() {
        let pairs = [
            (Dtype::Int32, Dtype::Uint32),
            (Dtype::Float16, Dtype::Bfloat16),
            (Dtype::Float64, Dtype::Complex64),
            (Dtype::Int8, Dtype::Float16),
        ];
        for (a, b) in pairs {
            assert_eq!(
                promote(a, b, false).unwrap(),
                promote(b, a, false).unwrap(),
                "JAX promote not symmetric for {} + {}",
                a,
                b
            );
        }
    }

    #[test]
    fn promote_is_symmetric_gpu() {
        let pairs = [
            (Dtype::Float16, Dtype::Float32),
            (Dtype::Bfloat16, Dtype::Float32),
            (Dtype::Float16, Dtype::Bfloat16),
            (Dtype::Int32, Dtype::Int64),
            (Dtype::Complex64, Dtype::Complex128),
        ];
        for (a, b) in pairs {
            assert_eq!(
                promote(a, b, true).unwrap(),
                promote(b, a, true).unwrap(),
                "GPU promote not symmetric for {} + {}",
                a,
                b
            );
        }
    }
}