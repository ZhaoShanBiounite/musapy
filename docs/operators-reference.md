# 已实现算子参考

> **版本**: v0.2-alpha  
> **设备**: CPU fallback + MUSA GPU kernel 双路径  
> **Kernel 精度**: i64 / f32 / f64（小整数 cast 到 i64）

---

## 算子总览（28 个）

| 分类 | 算子 | 数量 |
|------|------|------|
| Binary elementwise | add, sub, mul, div, pow | 5 |
| Unary elementwise | sin, cos, exp, log, abs, sign, neg | 7 |
| Ternary-scalar | clamp | 1 |
| Comparison | gt, lt, ge, le, eq, ne | 6 |
| Reduction | sum, prod, max, min, mean | 5 |
| Arg-reduction | argmax, argmin | 2 |
| Scan | cumsum | 1 |
| Cast | astype | 1 |
| **合计** | | **28** |

---

## Elementwise 算子

### 签名

```python
# Binary
ms.add(a, b, out=None) -> Array    # +, -, *, /, ** 同理
# Unary
ms.sin(a, out=None) -> Array       # cos, exp, log, abs, sign, neg 同理
# Ternary-scalar
ms.clamp(a, lo: float, hi: float, out=None) -> Array
```

### 实现要点

- N 维 stride-aware（非连续内存安全），广播通过 stride=0 实现
- Binary 支持 NumPy 广播 + 自动类型提升
- Dtype 白名单：f32 / f64（整数输入先 cast）
- 输出始终连续布局（C-order）

### Kernel 符号

```
musapy_{op}_{f32|f64}_v2
```

Binary ABI:
```c
void musapy_{op}_{dtype}_v2(
    const T* a, const T* b, T* c,
    int ndim, const size_t* shape,
    const ssize_t* a_strides, const ssize_t* b_strides,
    musaStream_t stream);
```

Unary ABI:
```c
void musapy_{op}_{dtype}_v2(
    const T* a, T* c,
    int ndim, const size_t* shape,
    const ssize_t* a_strides,
    musaStream_t stream);
```

### Python 运算符映射

| 表达式 | 调用 |
|--------|------|
| `a + b` | `ms.add(a, b)` |
| `a - b` | `ms.sub(a, b)` |
| `a * b` | `ms.mul(a, b)` |
| `a / b` | `ms.div(a, b)` |
| `a ** b` | `ms.pow(a, b)` |
| `-a` | `ms.neg(a)` |
| `abs(a)` | `ms.abs(a)` |

---

## Comparison 算子

### 签名

```python
ms.gt(a, b, out=None) -> Array   # lt, ge, le, eq, ne 同理
```

### 实现要点

- 输入：f32 / f64（整数先 cast）
- 输出：**bool**（1 byte/element）
- 支持广播，语义与 NumPy 一致
- Kernel 符号：`musapy_{op}_{f32|f64}_v2`，输出 `uint8_t*`

---

## Reduction 算子

### 签名

```python
ms.sum(a, axis=None, keepdims=False, out=None) -> Array
ms.prod(a, axis=None, keepdims=False, out=None) -> Array
ms.max(a, axis=None, keepdims=False, out=None) -> Array
ms.min(a, axis=None, keepdims=False, out=None) -> Array
ms.mean(a, axis=None, keepdims=False, out=None) -> Array
ms.argmax(a, axis=None, out=None) -> Array
ms.argmin(a, axis=None, out=None) -> Array
ms.cumsum(a, axis=None, out=None) -> Array
```

### 实现要点

- **axis=None**：全局缩减，视为 1D（kernel_ndim=1, strides=[1]），输出 0-dim scalar
- **axis=int**：沿指定轴缩减，支持负索引
- **keepdims**：仅影响输出 Layout shape，kernel 不感知
- **Kernel 策略**：one-thread-per-output-element，每线程循环 axis_len 次累加（correctness-first）
- **NdMetaReduce 结构体按值传入 kernel**（非 host 指针）

### Compute dtype 规则（ADR-002-D3）

| 算子 | 整数输入 | 浮点输入 | 输出 dtype |
|------|---------|---------|-----------|
| sum/prod/max/min/cumsum | cast → i64 | 保持 | 同 compute dtype |
| mean | cast → f64 | 保持 | 同 compute dtype |
| argmax/argmin | cast → i64 | 保持 | **恒 i64**（索引） |

### Kernel 符号（23 个）

```
musapy_{sum|prod|max|min}_{i64|f32|f64}_v2     # 12
musapy_mean_{f32|f64}_v2                        #  2
musapy_{argmax|argmin}_{i64|f32|f64}_v2         #  6
musapy_cumsum_{i64|f32|f64}_v2                  #  3
```

Reduction ABI:
```c
void musapy_{op}_{dtype}_v2(
    const T* a, T* c,
    int ndim, size_t in_shape[MUSAPY_MAX_NDIM],
    ssize_t in_strides[MUSAPY_MAX_NDIM],
    int axis, size_t axis_len, size_t out_size,
    musaStream_t stream);
```

Arg-reduction ABI（输出 int64_t*）:
```c
void musapy_{argmax|argmin}_{dtype}_v2(
    const T* a, int64_t* c,
    int ndim, size_t in_shape[MUSAPY_MAX_NDIM],
    ssize_t in_strides[MUSAPY_MAX_NDIM],
    int axis, size_t axis_len, size_t out_size,
    musaStream_t stream);
```

---

## Cast 算子

### 签名

```python
a.astype(dtype) -> Array
```

### 实现要点

- 目标 dtype：float32 / float64 / int64
- 源 dtype：int8~uint64 / float32 / float64
- 同 dtype 返回深拷贝
- Kernel 符号：`musapy_cast_{src}_{dst}_v2`（25 个组合）
- Stride-aware（支持非连续输入）

---

## 类型提升规则

Binary 算子输入 dtype 不同时自动提升：

| 条件 | 结果 |
|------|------|
| f32 + f32 | f32 |
| f64 + f64 | f64 |
| f32 + f64 | f64 |
| int(≤32) + f32 | f32 |
| int64/uint64 + float | f64 |
| 纯整数 + 纯整数 | f64 |

设计原则：kernel 仅实例化 f32/f64/i64，整数输入必须先 cast。

---

## 广播规则

1. 维度数不同时，较小 shape 前面补 1
2. 逐维：相等 → 通过；其一为 1 → 拉伸；否则 → `ShapeError`
3. 输出 shape = 各维取最大值
4. 实现：stride=0 零拷贝，kernel 按 stride 寻址

---

## 异常

| 异常 | 条件 |
|------|------|
| `ShapeError` | 广播不兼容 / out shape 不匹配 / axis 越界 |
| `DtypeError` | dtype 不在白名单 |
| `DeviceError` | 输入设备不一致 |
| `MemoryError` | out 与输入别名 |

---

## 性能参考

**环境**: MTT S4000, mp_22, 56 CUs, 47.9 GB VRAM  
**规模**: 1M elements × f32

| 类别 | 平均延迟 | 聚合吞吐 |
|------|---------|---------|
| elementwise (13 ops) | 0.72 ms | 73 GFLOPS |
| comparison (6 ops) | 0.31 ms | 19 GFLOPS |
| reduction global (8 ops) | ~210 ms | — (naive 1-thread) |
| reduction axis (256×256) | ~1.4 ms | — |

> 全局 reduction 慢是预期的：naive kernel 对 1M 元素只启动 1 个线程循环累加。  
> 复现: `python benchmark/bench_musa_utilization.py --size 1000000 --iters 50`

---

## 相关文档

- [ADR](ADR-zh.md) — 架构决策
- [v0.2 计划](v0.2-alpha-plan-zh.md) — Phase 规划
