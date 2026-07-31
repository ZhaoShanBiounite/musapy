# 已实现算子参考

> **版本**: v0.2-alpha (Phase 1 + Phase 2)  
> **最后更新**: 2026-07-31  
> **设备支持**: CPU / MUSA GPU (musa:\<id\>)

---

## 目录

- [概述](#概述)
- [Binary 算子](#binary-算子)
  - [add](#add)
  - [sub](#sub)
  - [mul](#mul)
  - [div](#div)
  - [pow](#pow)
- [Unary 算子](#unary-算子)
  - [sin](#sin)
  - [cos](#cos)
  - [exp](#exp)
  - [log](#log)
  - [abs](#abs)
  - [sign](#sign)
  - [neg](#neg)
- [Ternary-Scalar 算子](#ternary-scalar-算子)
  - [clamp](#clamp)
- [Cast 算子](#cast-算子)
  - [astype](#astype)
- [Python 运算符映射](#python-运算符映射)
- [类型提升规则](#类型提升规则)
- [广播规则](#广播规则)
- [Kernel ABI 说明](#kernel-abi-说明)
- [性能参考](#性能参考)

---

## 概述

musapy v0.2 实现了 **14 个 elementwise 算子**，全部支持：

- ✅ N 维 stride-aware 执行（非连续内存安全）
- ✅ NumPy 广播（Binary 算子）
- ✅ 自动类型提升（Binary 算子）
- ✅ `out=` 参数（原地写入）
- ✅ CPU fallback + MUSA GPU kernel 双路径
- ✅ float32 / float64 计算精度

| 分类 | 算子 | 数量 |
|------|------|------|
| Binary | add, sub, mul, div, pow | 5 |
| Unary | sin, cos, exp, log, abs, sign, neg | 7 |
| Ternary-Scalar | clamp | 1 |
| Cast | astype | 1 |
| **合计** | | **14** |

---

## Binary 算子

Binary 算子接受两个 Array 输入，支持 **NumPy 广播** 和 **自动类型提升**。

### 通用签名

```python
ms.<op>(a: Array, b: Array, out: Array | None = None) -> Array
```

### 通用行为

1. **广播**: 输入 shape 按 NumPy 规则对齐（见[广播规则](#广播规则)）
2. **类型提升**: 输入 dtype 不同时自动 cast 到公共 dtype（见[类型提升规则](#类型提升规则)）
3. **out= 校验**: shape / dtype / device 必须与输出一致，否则抛出异常
4. **别名检测**: `out` 不得与输入共享 buffer（抛出 `MemoryError`）

### 通用异常

| 异常 | 条件 |
|------|------|
| `ShapeError` | 广播不兼容 / out shape 不匹配 |
| `DtypeError` | 结果 dtype 不在 f32/f64 白名单 |
| `DeviceError` | 输入设备不一致 / out 设备不匹配 |
| `MemoryError` | out 与输入别名 |

---

### add

逐元素加法。

```python
ms.add(a, b, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = a[i] + b[i]` |
| 运算符 | `a + b` (`__add__`) |
| Kernel 符号 | `musapy_add_f32_v2`, `musapy_add_f64_v2` |
| FLOP/element | 1 |

**示例:**

```python
>>> a = ms.array([1.0, 2.0, 3.0])
>>> b = ms.array([4.0, 5.0, 6.0])
>>> ms.add(a, b).tolist()
[5.0, 7.0, 9.0]

>>> # 广播
>>> a = ms.array([[1.0], [2.0]])   # (2, 1)
>>> b = ms.array([10.0, 20.0])     # (2,)
>>> ms.add(a, b).shape
(2, 2)
```

---

### sub

逐元素减法。

```python
ms.sub(a, b, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = a[i] - b[i]` |
| 运算符 | `a - b` (`__sub__`) |
| Kernel 符号 | `musapy_sub_f32_v2`, `musapy_sub_f64_v2` |
| FLOP/element | 1 |

**示例:**

```python
>>> ms.sub(ms.array([5.0, 3.0]), ms.array([1.0, 2.0])).tolist()
[4.0, 1.0]
```

---

### mul

逐元素乘法。

```python
ms.mul(a, b, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = a[i] * b[i]` |
| 运算符 | `a * b` (`__mul__`) |
| Kernel 符号 | `musapy_mul_f32_v2`, `musapy_mul_f64_v2` |
| FLOP/element | 1 |

**示例:**

```python
>>> ms.mul(ms.array([2.0, 3.0]), ms.array([4.0, 5.0])).tolist()
[8.0, 15.0]
```

---

### div

逐元素除法。

```python
ms.div(a, b, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = a[i] / b[i]` |
| 运算符 | `a / b` (`__truediv__`) |
| Kernel 符号 | `musapy_div_f32_v2`, `musapy_div_f64_v2` |
| FLOP/element | 1 |
| 注意 | 除以零产生 `inf`（IEEE 754） |

**示例:**

```python
>>> ms.div(ms.array([10.0, 9.0]), ms.array([2.0, 3.0])).tolist()
[5.0, 3.0]
```

---

### pow

逐元素幂运算。

```python
ms.pow(a, b, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = a[i] ^ b[i]` |
| 运算符 | `a ** b` (`__pow__`) |
| Kernel 符号 | `musapy_pow_f32_v2`, `musapy_pow_f64_v2` |
| FLOP/element | ~8（超越函数） |
| 注意 | 负底数 + 非整数指数 → `NaN` |

**示例:**

```python
>>> ms.pow(ms.array([2.0, 3.0]), ms.array([3.0, 2.0])).tolist()
[8.0, 9.0]

>>> # 运算符形式
>>> (ms.array([4.0]) ** ms.array([0.5])).tolist()
[2.0]
```

---

## Unary 算子

Unary 算子接受单个 Array 输入，输出 shape 不变。支持 stride-aware 执行（非连续输入安全）。

### 通用签名

```python
ms.<op>(a: Array, out: Array | None = None) -> Array
```

### 通用约束

| 约束 | 说明 |
|------|------|
| Dtype 白名单 | 仅 `float32` / `float64`（整数输入需先 `astype`） |
| Shape | 输出 = 输入（无广播） |
| Stride-aware | 支持非连续输入（如 broadcast 视图） |

---

### sin

逐元素正弦。

```python
ms.sin(a, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = sin(a[i])` |
| Kernel 符号 | `musapy_sin_f32_v2`, `musapy_sin_f64_v2` |
| FLOP/element | ~8 |
| 值域 | [-1, 1] |

**示例:**

```python
>>> import math
>>> ms.sin(ms.array([0.0, math.pi / 2])).tolist()
[0.0, 1.0]
```

---

### cos

逐元素余弦。

```python
ms.cos(a, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = cos(a[i])` |
| Kernel 符号 | `musapy_cos_f32_v2`, `musapy_cos_f64_v2` |
| FLOP/element | ~8 |
| 值域 | [-1, 1] |

**示例:**

```python
>>> ms.cos(ms.array([0.0, math.pi])).tolist()
[1.0, -1.0]
```

---

### exp

逐元素自然指数。

```python
ms.exp(a, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = e^(a[i])` |
| Kernel 符号 | `musapy_exp_f32_v2`, `musapy_exp_f64_v2` |
| FLOP/element | ~8 |
| 值域 | (0, +∞) |

**示例:**

```python
>>> ms.exp(ms.array([0.0, 1.0])).tolist()
[1.0, 2.7182817...]
```

---

### log

逐元素自然对数。

```python
ms.log(a, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = ln(a[i])` |
| Kernel 符号 | `musapy_log_f32_v2`, `musapy_log_f64_v2` |
| FLOP/element | ~8 |
| 定义域 | (0, +∞) |
| 注意 | `log(0) = -inf`, `log(负数) = NaN` |

**示例:**

```python
>>> import math
>>> ms.log(ms.array([1.0, math.e])).tolist()
[0.0, 1.0]
```

---

### abs

逐元素绝对值。

```python
ms.abs(a, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = |a[i]|` |
| 运算符 | `abs(a)` (`__abs__`) |
| Kernel 符号 | `musapy_abs_f32_v2`, `musapy_abs_f64_v2` |
| FLOP/element | 1 |

**示例:**

```python
>>> ms.abs(ms.array([-3.0, 0.0, 5.0])).tolist()
[3.0, 0.0, 5.0]

>>> abs(ms.array([-1.0, 2.0])).tolist()
[1.0, 2.0]
```

---

### sign

逐元素符号函数。

```python
ms.sign(a, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = 1 if a[i]>0, -1 if a[i]<0, 0 if a[i]==0` |
| Kernel 符号 | `musapy_sign_f32_v2`, `musapy_sign_f64_v2` |
| FLOP/element | 1 |
| 值域 | {-1, 0, 1} |

**示例:**

```python
>>> ms.sign(ms.array([-10.0, 0.0, 7.0])).tolist()
[-1.0, 0.0, 1.0]
```

---

### neg

逐元素取反。

```python
ms.neg(a, out=None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = -a[i]` |
| 运算符 | `-a` (`__neg__`) |
| Kernel 符号 | `musapy_neg_f32_v2`, `musapy_neg_f64_v2` |
| FLOP/element | 1 |

**示例:**

```python
>>> ms.neg(ms.array([1.0, -2.0, 0.0])).tolist()
[-1.0, 2.0, 0.0]

>>> (-ms.array([3.0, -4.0])).tolist()
[-3.0, 4.0]
```

---

## Ternary-Scalar 算子

### clamp

逐元素截断到 [lo, hi] 区间。

```python
ms.clamp(a: Array, lo: float, hi: float, out: Array | None = None) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | `out[i] = min(max(a[i], lo), hi)` |
| 参数 | `lo`, `hi` 为 Python float（f64），按输入 dtype 内部转换 |
| Kernel 符号 | `musapy_clamp_f32_v2`, `musapy_clamp_f64_v2` |
| FLOP/element | 2（两次比较） |
| Dtype 白名单 | float32 / float64 |

**示例:**

```python
>>> ms.clamp(ms.array([-5.0, 0.5, 10.0]), 0.0, 1.0).tolist()
[0.0, 0.5, 1.0]

>>> # 2D
>>> a = ms.array([[-1.0, 5.0], [0.5, 20.0]])
>>> ms.clamp(a, 0.0, 10.0).tolist()
[[0.0, 5.0], [0.5, 10.0]]
```

---

## Cast 算子

### astype

显式 dtype 转换（Array 方法）。

```python
a.astype(dtype: Dtype) -> Array
```

| 属性 | 值 |
|------|-----|
| 语义 | 逐元素 `static_cast<Dst>(src[i])` |
| 目标 dtype | `float32`, `float64` |
| 源 dtype | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64 |
| 同 dtype | 返回深拷贝（要求连续布局） |
| Kernel 符号 | `musapy_cast_{src}_{dst}_v2`（18 个组合） |

**支持的 cast 矩阵:**

| 源 ↓ \ 目标 → | float32 | float64 |
|---------------|---------|---------|
| int8 | ✅ | ✅ |
| int16 | ✅ | ✅ |
| int32 | ✅ | ✅ |
| int64 | ✅ | ✅ |
| uint8 | ✅ | ✅ |
| uint16 | ✅ | ✅ |
| uint32 | ✅ | ✅ |
| uint64 | ✅ | ✅ |
| float32 | ✅ (copy) | ✅ |
| float64 | ✅ | ✅ (copy) |

**示例:**

```python
>>> a = ms.array([1, 2, 3], dtype=ms.int64)
>>> b = a.astype(ms.float32)
>>> b.dtype
float32
>>> b.tolist()
[1.0, 2.0, 3.0]

>>> # 保持 shape
>>> c = ms.array([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float32)
>>> c.astype(ms.float64).shape
(2, 2)
```

---

## Python 运算符映射

| Python 表达式 | 调用 | 说明 |
|--------------|------|------|
| `a + b` | `ms.add(a, b)` | 支持广播 |
| `a - b` | `ms.sub(a, b)` | 支持广播 |
| `a * b` | `ms.mul(a, b)` | 支持广播 |
| `a / b` | `ms.div(a, b)` | 支持广播 |
| `a ** b` | `ms.pow(a, b)` | 支持广播 |
| `-a` | `ms.neg(a)` | 逐元素取反 |
| `abs(a)` | `ms.abs(a)` | 逐元素绝对值 |

> **注意**: 运算符形式不支持 `out=` 参数，始终分配新 Array。

---

## 类型提升规则

Binary 算子在输入 dtype 不同时自动提升：

```
promote(dtype_a, dtype_b, all_gpu) -> result_dtype
```

### 提升表

| dtype_a | dtype_b | result | 说明 |
|---------|---------|--------|------|
| float32 | float32 | float32 | 无需 cast |
| float64 | float64 | float64 | 无需 cast |
| float32 | float64 | float64 | f32 → f64 |
| int8 | float32 | float32 | i8 → f32 |
| int16 | float32 | float32 | i16 → f32 |
| int32 | float32 | float32 | i32 → f32 |
| int64 | float32 | **float64** | int64 参与 → 提升到 f64 |
| int8 | float64 | float64 | i8 → f64 |
| int16 | float64 | float64 | i16 → f64 |
| int32 | float64 | float64 | i32 → f64 |
| int64 | float64 | float64 | i64 → f64 |
| uint8 | float32 | float32 | u8 → f32 |
| uint16 | float32 | float32 | u16 → f32 |
| uint32 | float32 | float32 | u32 → f32 |
| uint64 | float32 | **float64** | uint64 参与 → 提升到 f64 |
| int32 | int64 | float64 | 纯整数 → f64 |

### 设计原则（ADR-002-D1）

- 计算 kernel 仅实例化 f32/f64 → 整数输入必须先 cast
- `int64`/`uint64` 参与时结果至少为 f64（避免精度丢失）
- 提升后对输入执行内部 `astype`，再调用 same-dtype kernel
- Kernel 矩阵 O(N) 而非 O(N³)

---

## 广播规则

Binary 算子支持 NumPy 风格广播（`broadcast.rs`）：

### 规则

1. 若两数组维度数不同，在较小 shape 前面补 1
2. 逐维比较：相等 → 通过；其中一个为 1 → 拉伸；否则 → `ShapeError`
3. 输出 shape = 各维取最大值

### 示例

| Shape A | Shape B | 输出 | 说明 |
|---------|---------|------|------|
| (3,) | (3,) | (3,) | 无广播 |
| (3,1) | (4,) | (3,4) | 经典广播 |
| (1,4) | (3,1) | (3,4) | 双向拉伸 |
| () | (n,) | (n,) | 标量 + 向量 |
| () | () | () | 标量 + 标量 |
| (2,1,3) | (4,1) | (2,4,3) | 高维广播 |
| (2,3) | (4,) | ❌ | 3≠4，不兼容 |

### 实现

- 广播通过 **strides=0** 实现（零拷贝，不实际复制数据）
- Kernel 使用 `offset_nd()` 按 stride 寻址
- 输出始终为连续布局（C-order）

---

## Kernel ABI 说明

### v2 ABI（stride-aware）

所有 Phase 2 kernel 使用统一的 stride-aware ABI：

**Binary:**

```c
void musapy_{op}_{dtype}_v2(
    const T* a, const T* b, T* c,
    int ndim, const size_t* shape,
    const ssize_t* a_strides, const ssize_t* b_strides,
    musaStream_t stream
);
```

**Unary:**

```c
void musapy_{op}_{dtype}_v2(
    const T* a, T* c,
    int ndim, const size_t* shape,
    const ssize_t* a_strides,
    musaStream_t stream
);
```

### 参数传递

- `shape` / `strides` 打包为 `NdMeta` / `NdMetaUnary` 结构体
- 按值传入 kernel（自动进入 GPU constant memory）
- 最大维度: `MUSAPY_MAX_NDIM = 32`

### 寻址

```c
// 线性索引 → N 维偏移
size_t offset_nd(size_t linear_idx, const size_t* shape,
                 const ssize_t* strides, int ndim);
```

- 广播维度 stride=0 → 所有线程读同一元素
- 输出始终 stride=连续（idx 即 offset）

---

## 性能参考

**测试环境**: MTT S4000 (arch mp_22, 56 CUs, 47.9 GB VRAM)  
**数据规模**: 1,000,000 elements × float32 (3.81 MB/array)

| 算子 | 延迟 (ms) | 吞吐 (GElem/s) | GFLOPS |
|------|-----------|----------------|--------|
| add | 0.491 | 2.036 | 2.036 |
| sub | 0.443 | 2.255 | 2.255 |
| mul | 0.439 | 2.280 | 2.280 |
| div | 0.435 | 2.300 | 2.300 |
| pow | 0.446 | 2.243 | 17.946 |
| sin | 0.382 | 2.618 | 20.943 |
| cos | 0.378 | 2.646 | 21.166 |
| exp | 0.377 | 2.654 | 21.229 |
| log | 0.515 | 1.940 | 15.521 |
| abs | 0.368 | 2.714 | 2.714 |
| sign | 0.366 | 2.733 | 2.733 |
| neg | 0.359 | 2.782 | 2.782 |
| clamp | 0.356 | 2.809 | 5.617 |

**聚合指标:**

| 指标 | 值 |
|------|-----|
| 聚合吞吐 | 119.52 GFLOPS |
| 持续 kernel 发射率 | 10,532 launches/s |
| 等效内存带宽 | 126.47 GB/s |
| 峰值显存占用 | 32.2 MB |

> 复现: `python benchmark/bench_musa_utilization.py --size 1000000 --iters 100`

---

## 相关文档

- [架构决策记录 (ADR)](ADR-zh.md) — 设计原则与决策
- [v0.2 实现计划](v0.2-alpha-plan-zh.md) — Phase 规划
- [快速上手](getting-started.md) — 安装与基础用法
