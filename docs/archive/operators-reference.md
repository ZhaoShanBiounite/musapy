# 已实现算子参考

> **版本**: v0.3-alpha  
> **设备**: CPU fallback + MUSA GPU kernel 双路径  
> **Kernel 精度**: i64 / f32 / f64 / c64 / c128（小整数 cast 到 i64；复数支持见各章节）

---

## 算子总览（v0.3，61 个 + 3 命名空间 + 索引语法）

| 分类 | 算子 | 数量 |
|------|------|------|
| Binary elementwise | add, sub, mul, div, pow | 5 |
| Unary elementwise | sin, cos, exp, log, abs, sign, neg | 7 |
| Ternary-scalar | clamp | 1 |
| Comparison | gt, lt, ge, le, eq, ne | 6 |
| Reduction | sum, prod, max, min, mean（axis=int/tuple） | 5 |
| Arg-reduction | argmax, argmin（axis=int/tuple, keepdims） | 2 |
| Scan | cumsum | 1 |
| Cast | astype（Array 方法） | 1 |
| Init | zeros, ones, full, eye, arange, linspace, zeros_like, ones_like | 8 |
| Indexing | transpose, permute, flip, slice, index_select, contiguous, gather, scatter | 8 |
| Linalg | matmul, dot, solve, lu, qr, svd | 6 |
| 命名空间 | ms.random（rand, randn, uniform, normal, bernoulli）· ms.fft（fft, ifft, rfft）· ms.sparse（csr_matrix, spmv, spmm） | 3 子模块 11 函数 |
| 高级索引 | a[mask]（boolean）· a[idx] / a[i0,i1,...]（fancy） | 语法扩展 |
| Complex 扩展 | elementwise（add/sub/mul/div/neg/abs/eq/ne）+ reduction（sum/mean/prod）+ cast（real→c64/c128） | dtype 扩展 |
| **合计** | | **50 个模块函数 + 11 命名空间函数（61）+ 3 命名空间 + 高级索引语法** |

> 复数 max/min/argmax/argmin **永久拒绝**（复数无全序，002-D3）。
> 数学库算子（linalg/random/fft/sparse）**GPU-only**（003-D4）。

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
ms.argmax(a, axis=None, keepdims=False, out=None) -> Array
ms.argmin(a, axis=None, keepdims=False, out=None) -> Array
ms.cumsum(a, axis=None, out=None) -> Array
```

### 实现要点

- **axis=None**：全局缩减，视为 1D（kernel_ndim=1, strides=[1]），输出 0-dim scalar
- **axis=int**：沿指定轴缩减，支持负索引
- **axis=tuple**（Phase 7 P7.1）：多轴归约。
  - sum/prod/max/min/mean：**逐轴迭代**（升序），全部轮 keepdims=true 保维，
    最后统一 squeeze 被归约轴（用户 keepdims=false 时）；重复轴报 ShapeError
  - argmax/argmin：**transpose+合并轴**（指定轴移到末尾、contiguous、reshape 合并
    为单轴后走单轴 kernel），索引为展平指定轴的**扁平索引**（NumPy 2.0+ 语义）；
    arg* 同步支持 keepdims（被归约轴处=1）
- **keepdims**：仅影响输出 Layout shape，kernel 不感知
- **Kernel 策略（P2 起三路选择）**：
  - `axis_len ≤ 16` 或 argmax/argmin → naive one-thread-per-output
  - `16 < axis_len ≤ 1024`（sum/prod/max/min/mean）→ 小 axis 并行
    （每输出 32..256 线程组，warp shuffle + smem 两级归约）
  - `axis_len > 1024` → 两阶段并行（partial 每线程 4 元素 + final）
- **复数**（Phase 7 P7.2）：sum/mean/prod 支持 complex64/128（naive 路径，
  显式 re/im 分量公式，CPU+MUSA 双路径）；**max/min/argmax/argmin 对复数抛
  DtypeError**（复数无全序，ADR-003 003-D5）
- **cumsum**：work-efficient 分层扫描，**单轴容量上限 256³ ≈ 16.7M 元素**，
  超限报错（P0 修复：此前 axis_len > 65536 结果错误 + smem 越界）
- **NdMetaReduce 结构体按值传入 kernel**（非 host 指针）

### Compute dtype 规则（ADR-002-D3）

| 算子 | 整数输入 | 浮点输入 | 复数输入 | 输出 dtype |
|------|---------|---------|---------|-----------|
| sum/prod/cumsum | cast → i64 | 保持 | 保持（sum/prod） | 同 compute dtype |
| max/min | cast → i64 | 保持 | **拒绝**（无全序） | 同 compute dtype |
| mean | cast → f64 | 保持 | 保持 | 同 compute dtype |
| argmax/argmin | cast → i64 | 保持 | **拒绝**（无全序） | **恒 i64**（索引） |

### Kernel 符号（107 个，含复数 sum/prod/mean 与 arg _mid_ 族）

```
musapy_{sum|prod}_{i64|f32|f64|c64|c128}_v2                    # 20（naive）
musapy_{max|min}_{i64|f32|f64}_v2                               #  6（naive）
musapy_mean_{f32|f64|c64|c128}_v2                               #  4（naive）
musapy_{argmax|argmin}_{i64|f32|f64}_v2                         #  6（naive）
musapy_{sum|prod}_small_axis_{i64|f32|f64|c64|c128}_v2          # 20（小 axis）
musapy_{max|min}_small_axis_{i64|f32|f64}_v2                    #  6（小 axis）
musapy_mean_small_axis_{f32|f64|c64|c128}_v2                    #  4（小 axis）
musapy_{sum|prod}_partial_{i64|f32|f64|c64|c128}_v2             # 20（两阶段 P1）
musapy_{max|min}_partial_{i64|f32|f64}_v2                       #  6（两阶段 P1）
musapy_mean_partial_{f32|f64|c64|c128}_v2                       #  4（两阶段 P1）
musapy_{sum|prod}_final_{i64|f32|f64|c64|c128}_v2               # 20（两阶段 P2）
musapy_{max|min}_final_{i64|f32|f64}_v2                         #  6（两阶段 P2）
musapy_mean_final_{f32|f64|c64|c128}_v2                         #  4（两阶段 P2）
musapy_{argmax|argmin}_partial_{i64|f32|f64}_v2                 #  6（两阶段 P1）
musapy_{argmax|argmin}_mid_{i64|f32|f64}_v2                     #  6（多级 partial 中间级，P2b）
musapy_{argmax|argmin}_final_{i64|f32|f64}_v2                   #  6（两阶段 P2）
musapy_cumsum_{i64|f32|f64}_v3                                  #  3（分层扫描）
```

> 复数（c64/c128）reduction：sum/prod/mean 支持（re/im 分量并行归约，
> 2026-08-08 优化 ~1900×）；max/min/argmax/argmin 对复数**永久拒绝**（无全序）。
> arg _mid_ 族为 P2b 多级 partial 的中间级（每级 ÷1024 递归，val/idx 双缓冲）。
> naive 值算子 14 个保留——门禁实测在 axis_len ≤ 16 × 大 out_size 时优于
> 小 axis 路径（最高 15.5×），argmax/argmin 在 axis_len ≤ 1024 段也只有 naive 实现。

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

- 目标 dtype：float32 / float64 / int64 / complex64 / complex128
- 源 dtype：int8~uint64 / float32 / float64（→ real 或 complex）；complex64 → complex128
  （宽度提升）；complex→real **不支持**（抛 DtypeError，无 cast kernel）
- 同 dtype 返回深拷贝
- Kernel 符号：`musapy_cast_{src}_{dst}_v2`（32 个：27 实数组合 + 5 复数组合）
  + fft 扩展 `musapy_cast_resize_{src}_{dst}_v2`（2 个，截断/补零）
- Stride-aware（支持非连续输入）

---

## Init 算子

### 签名

```python
ms.zeros(shape, dtype=None, device=None) -> Array           # 填充 0
ms.ones(shape, dtype=None, device=None) -> Array            # 填充 1
ms.full(shape, value, dtype=None, device=None) -> Array     # 填充常量
ms.eye(n, m=None, k=0, dtype=None, device=None) -> Array    # 单位阵（k 对角线偏移）
ms.arange(start, stop=None, step=1, dtype=None, device=None) -> Array
ms.linspace(start, stop, num=50, dtype=None, device=None) -> Array
ms.zeros_like(a) -> Array                                    # 继承输入 dtype/device
ms.ones_like(a) -> Array
```

### 实现要点

- dtype 解析（ADR-002-D5 / L0-7 链）：显式 `dtype=` > context > 全局默认 > **float32**；
  `arange` / `linspace` 从参数推断（全整数 → int64，含浮点 → float64），显式 `dtype=` 覆盖
- device 解析（L0-6 链）：首次创建需显式 `ms.set_default_device()` 或 `device=` 参数（L0-9，
  否则抛 `DeviceNotConfiguredError`）
- `*_like` 继承输入 Array 的 dtype/device（L3-18）
- kernel 为「写值」kernel：fill 写常量、arange/linspace 由 idx→value、eye 条件写（`i==j`）

### Kernel 符号

```
musapy_fill_{f32|f64|i64|i32|i16|i8|u64|u32|u16|u8}      # 10
musapy_arange_{f32|f64|i64|i32}                           #  4
musapy_linspace_{f32|f64}                                 #  2
musapy_eye_{f32|f64|i64|i32}                              #  4
```

---

## Indexing 算子（v0.2 Phase 6.5-7）

### 签名

```python
ms.transpose(a, axes=None) -> Array          # 零拷贝视图
ms.permute(a, dims) -> Array                 # 零拷贝视图
ms.flip(a, axis) -> Array                    # 零拷贝视图（stride 取负）
ms.slice(a, specs) -> Array                  # 零拷贝视图
ms.index_select(a, axis, index) -> Array     # 零拷贝视图（按 axis 取 index 处子集）
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

## 高级索引（v0.3 Phase 8，ADR-002-D4）

### 签名

```python
a[mask]          # boolean mask（等形或前 md 维广播）→ copy
a[idx]           # fancy 单索引（1D/N-D 索引数组）→ copy
a[i0, i1, ...]   # 多索引坐标配对（索引形状广播）→ copy
a[[0, 1, 2]]     # Python list 索引（自动转 int64）→ copy
```

### 实现要点

- **boolean mask**（`boolean_mask`）：mask 与 a 的**前 md 维**左对齐广播
  （NumPy 语义），输出 `(n_true,) + a.shape[md:]`，按 C 序取 true 位置子块展平拼接
- **fancy indexing**（`adv_index`）：单/多索引数组坐标配对 + 索引形状右对齐广播 +
  N-D 索引数组 + 负索引（kernel/ops 内转正）；输出恒为 copy
- **越界抛 Python 内置 `IndexError`**（非 MusapyError 子类，NumPy 兼容；
  L3-6 单继承限制下直接映射 `pyo3::PyIndexError`）
- `__getitem__` 识别 PyArray / ndarray / list 索引；混合 basic+fancy
  （`a[1:, [0,2]]`）抛 `NotImplementedError`（v0.4 推迟）
- 实现路径：GPU 侧目前走 **host fallback**（mcc 不支持指针数组 kernel 参数，
  error 999 探针证实）；`musapy_adv_gather_*_v2`/`musapy_nonzero_*_v2` kernel
  已声明，后续优化接入

---

## FFT 算子（v0.3 Phase 5，ADR-003 003-D5/D7）

### 签名

```python
ms.fft.fft(a, n=None, axis=-1, norm=None, out=None) -> Array   # 复数 FFT
ms.fft.ifft(a, n=None, axis=-1, norm=None, out=None) -> Array  # 逆变换（backward 缩放 1/N）
ms.fft.rfft(a, n=None, axis=-1, norm=None, out=None) -> Array  # 实输入 → 输出 (..., N//2+1)
```

### 实现要点

- 实现走 **muFFT**（`mufftExecC2C/Z2Z/R2C/D2Z`）；plan 经
  `math_handle::with_mufft_plan` 按 `MufftPlanSpec::OneD` 池化复用
- **GPU-only**（003-D4）：CPU 设备上调用抛 `DeviceError`
- **本轮范围（axis=-1 起步）**：只支持沿最后一维；`axis != -1` 抛 `ShapeError`
  （fftn/多轴推迟到 v0.3 后期）
- `n` 截断/补零：`n < last_dim` 截断、`n > last_dim` 补零（resize kernel）
- `norm`：`"backward"`（默认）/ `"ortho"` / `"forward"`（NumPy 语义）
- 输入 dtype：real f32/f64（内部扩 complex，`re=x, im=0`）或 complex64/128；
  输出恒 complex（real f32→c64 / f64→c128）
- 2D+ 输入沿 axis=-1 逐行执行（Plan1d batch=1，逐行偏移指针）
- mock 模式：mufft stub 用 naive O(N²) DFT 数值仿真（无 GPU CI 对照 np.fft）

## Sparse 算子（v0.3 Phase 6，ADR-003 003-D4/D7）

### 签名

```python
csr = ms.sparse.csr_matrix((data, indices, indptr), shape=None, dtype=None)  # CsrMatrix
y = csr @ vec          # spmv（vec 可为 ms.Array / ndarray / list）
C = csr @ dense        # spmm（dense 2D）
A = csr.toarray()      # 物化稠密 Array
ms.sparse.spmv(csr, vec) / ms.sparse.spmm(csr, dense)  # 显式函数形式
```

### 实现要点

- 实现走 **muSPARSE 泛型 API**（`musparseCreateCsr` + `CreateDnVec/DnMat` +
  `SpMV/SpMM`）；两段式（`temp_buffer=NULL` 查询 size → `get_workspace` → 计算）
- **GPU-only**（003-D4）：CPU 设备上构造/运算抛 `DeviceError`
- **`shape=(rows, cols)` 必须显式提供**（nnz>0 时无法从 device 端推断 cols，
  缺省会抛 ValueError）；nnz=0 时可省略（默认全零矩阵）
- **本轮范围**：只做 `csr_matrix`（`coo_matrix`/coo→csr 推迟）；data dtype
  f32/f64，indices/indptr 须 int32（`MUSPARSE_INDEX_32I`，0-based）
- `@` 右侧：ms.Array 走 device 直连；ndarray/list 经 `tolist()` → `ms.array`
  构造临时 device Array（dtype 沿用 mat）
- `nnz==0` 空矩阵早退输出全零；`toarray()` 走 D2H→host 构建→H2D（正确性优先）
- mock 模式：musparse stub 用 host CSR 循环数值仿真（无 GPU CI 对照 NumPy）

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

- [ADR](../adr/ADR-zh.md)) — 架构决策
- [v0.2 计划](./v0.2-alpha-plan-zh.md)) — Phase 规划
