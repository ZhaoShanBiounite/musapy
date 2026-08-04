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
- **Kernel 策略（P2 起三路选择）**：
  - `axis_len ≤ 16` 或 argmax/argmin → naive one-thread-per-output
  - `16 < axis_len ≤ 1024`（sum/prod/max/min/mean）→ 小 axis 并行
    （每输出 32..256 线程组，warp shuffle + smem 两级归约）
  - `axis_len > 1024` → 两阶段并行（partial 每线程 4 元素 + final）
- **cumsum**：work-efficient 分层扫描，**单轴容量上限 256³ ≈ 16.7M 元素**，
  超限报错（P0 修复：此前 axis_len > 65536 结果错误 + smem 越界）
- **NdMetaReduce 结构体按值传入 kernel**（非 host 指针）

### Compute dtype 规则（ADR-002-D3）

| 算子 | 整数输入 | 浮点输入 | 输出 dtype |
|------|---------|---------|-----------|
| sum/prod/max/min/cumsum | cast → i64 | 保持 | 同 compute dtype |
| mean | cast → f64 | 保持 | 同 compute dtype |
| argmax/argmin | cast → i64 | 保持 | **恒 i64**（索引） |

### Kernel 符号（76 个，P6 清理后）

```
musapy_{sum|prod|max|min}_{i64|f32|f64}_v2                  # 12（naive）
musapy_mean_{f32|f64}_v2                                     #  2（naive）
musapy_{argmax|argmin}_{i64|f32|f64}_v2                      #  6（naive）
musapy_{sum|prod|max|min}_small_axis_{i64|f32|f64}_v2        # 12（小 axis）
musapy_mean_small_axis_{f32|f64}_v2                          #  2（小 axis）
musapy_{sum|prod|max|min}_partial_{i64|f32|f64}_v2           # 12（两阶段 P1）
musapy_mean_partial_{f32|f64}_v2                             #  2（两阶段 P1）
musapy_{sum|prod|max|min}_final_{i64|f32|f64}_v2             # 12（两阶段 P2）
musapy_mean_final_{f32|f64}_v2                               #  2（两阶段 P2）
musapy_{argmax|argmin}_partial_{i64|f32|f64}_v2              #  6（两阶段 P1）
musapy_{argmax|argmin}_final_{i64|f32|f64}_v2                #  6（两阶段 P2）
musapy_cumsum_{i64|f32|f64}_v3                               #  3（分层扫描）
```

> P6（2026-08-04）清理：删除 3 个无调用者的死符号
> （`musapy_add_f32/f64_v1`、`musapy_mean_partial_i64_v2`），
> 全库 extern 符号 179 → 176。naive 值算子 14 个保留——门禁实测
> 在 axis_len ≤ 16 × 大 out_size 时优于小 axis 路径（最高 15.5×），
> argmax/argmin 在 axis_len ≤ 1024 段也只有 naive 实现。

Reduction ABI（naive / small_axis）:
```c
// naive：输入 T，输出 T
void musapy_{op}_{dtype}_v2(
    const T* a, T* c,
    int ndim, size_t in_shape[MUSAPY_MAX_NDIM],
    ssize_t in_strides[MUSAPY_MAX_NDIM],
    int axis, size_t axis_len, size_t out_size,
    musaStream_t stream);
// 小 axis：额外 group_size ∈ {32,64,128,256}
void musapy_{op}_small_axis_{dtype}_v2(..., int group_size, musaStream_t stream);
// 两阶段 partial：tiles_per_output = ceil(axis_len / 1024)
void musapy_{op}_partial_{dtype}_v2(..., size_t tiles_per_output, musaStream_t stream);
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

## Indexing 算子（v0.2 Phase 6.5-7）

### 签名

```python
ms.transpose(a, axes=None) -> Array          # 零拷贝视图
ms.permute(a, dims) -> Array                 # 零拷贝视图
ms.flip(a, axis) -> Array                    # 零拷贝视图（stride 取负）
ms.slice(a, specs) -> Array                  # 零拷贝视图
ms.contiguous(a) -> Array                    # 已连续零拷贝；否则 kernel 物化
ms.gather(a, indices, axis=0) -> Array       # copy，等价 np.take
ms.scatter(a, indices, values, axis=0) -> Array  # copy，返回新数组
```

### 实现要点

- view 算子零拷贝，仅修改 Layout；copy 算子分配新 buffer 走 kernel
- gather/scatter kernel 实例化 f32/f64/i32/i64（符号 `musapy_{op}_{dtype}_v2`）；
  其余 dtype 走 D2H→host→H2D fallback
- indices 固定 1D int64；CPU indices 自动 H2D 上传

### GPU 越界语义（P1 去同步，2026-08）

GPU 路径不在 host 端同步校验 indices，而是由 kernel 内检查：

1. 越界（含负数）元素跳过读/写，并通过 device 错误槽上报
   （atomicCAS 记录首个越界的展平位置与索引值，atomicOr 置标志）
2. 异常延迟到下一次流同步抛出（`tolist()`/`item()` 内部会同步），
   类型为 `ShapeError`，消息含算子上下文、越界值与位置
3. **流不毒化**：报错后同一流可继续使用；越界条目已被跳过，
   其余结果有效
4. CPU 路径与 mock 构建仍为同步校验、立即报错

因此 GPU 上 `ms.gather(a, idx).tolist()` 的报错位置与 numpy 风格一致
（取值/物化时抛出），但纯异步管线中错误可能延迟数个 op 才暴露。

---

## 类型提升规则

Binary 算子输入 dtype 不同时自动提升（两段式，见 ADR L1-14）：

| 条件 | 结果（CPU / 全 GPU） |
|------|----------------------|
| f32 + f64 | f64 / **f32**（GPU 窄优先） |
| f16 + bf16 | f32（同宽冲突 → JAX） |
| int/uint（任意位宽）+ float | **float 本身**（JAX 语义：整数不因位宽升级浮点；`i64 + f32 → f32`，对齐 v0.2 计划 §1.3） |
| 纯整数 + 纯整数 | 宽者（CPU）/ 窄者（GPU） |
| int + uint | 溢出保护升级（CPU/GPU 均 JAX 表） |
| int/float + complex | 宽 complex（CPU/GPU 均 JAX 表） |

设计原则：kernel 仅实例化 f32/f64/i64，整数输入必须先 cast；
`i64 + f32 → f32` 意味着 int64 输入会被 cast 成 f32（精度损失为计划
既定语义——"GPU 窄优先"）。

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
**规模**: 1M elements × f32（2026-08-04，P0–P5 优化后）

| 类别 | 平均延迟 | 备注 |
|------|---------|------|
| elementwise (13 ops) | ~0.054–0.066 ms | 受 ~45 µs launch 地板限制 |
| comparison (6 ops) | ~0.057 ms | 同上 |
| reduction 全局 (8 ops) | 0.084–0.142 ms | sum 0.085 / argmax 0.090 / cumsum 0.306 |
| reduction 2D (256×256) | 0.053–0.064 ms | 小 axis 并行路径 |
| gather(full) / scatter(full) | 0.178 / 0.240 ms | P1 去同步后 |
| contig(transp) / contig(flip) | 0.063 / 0.104 ms | P4 tiled kernel / u32 路径 |

**launch 地板（P3 坐实）**: 单次 kernel launch + sync ≈ 45 µs 固定开销
（driver 提交路径）。1M 规模的延迟读数 ≈ 地板 + kernel 执行；**≥4M 规模
才反映真实带宽**：elementwise 16M 620 GB/s、64M 655 GB/s（≈ DRAM 峰值
89%），转置 4M/16M 221/289 GB/s。

> 复现: `python benchmark/bench_musa_utilization.py --size 1000000 --iters 100`

---

## 相关文档

- [ADR](ADR-zh.md) — 架构决策
- [v0.2 计划](v0.2-alpha-plan-zh.md) — Phase 规划
