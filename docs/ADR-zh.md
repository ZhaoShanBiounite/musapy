# musapy 架构决策记录（ADR）

> **状态**：草案
> **最后更新**：2025-01-15
> **范围**：musapy v1.0 设计（Python + Rust + MUSA 科学计算库）

本文档记录 musapy 的全部架构决策，按 5 层组织。每个决策有稳定 ID（`L<层>-<编号>`），
便于在代码、issue、未来 ADR 中引用。

---

## 目录

- [Layer 0：定位](#layer-0定位)
- [Layer 1：核心抽象](#layer-1核心抽象)
- [Layer 2：模块契约](#layer-2模块契约)
- [Layer 3.1：错误模型](#layer-31错误模型)
- [Layer 3.2：内存生命周期](#layer-32内存生命周期)
- [Layer 3.3：互操作](#layer-33互操作)
- [Layer 3.4：可观测性](#layer-34可观测性)
- [Layer 4：演进策略](#layer-4演进策略)
- [附录：决策索引](#附录决策索引)

---

## Layer 0：定位

### L0-1：主要受众

**决策**：科研用户优先（C），HPC 用户作为演进目标（A）。

**依据**：当前 MUSA 生态成熟度不足以支撑 ML 用户（B）。科研用户最能从 MUSA 上的
CuPy 等价物受益。HPC 用户在生态成熟后跟进。

**影响**：
- "显式 device" 是 feature 而非负担（HPC 思维方式）
- CPU fallback 通过独立的 `musapy-cpu` crate（opt-in），不进主库
- PyTorch 兼容（`__torch_function__`、autograd）推迟到 v3+

### L0-2：主要场景

**决策**：单卡加速起步，大规模离线计算作为演进目标。

**影响**：
- v1 不包含分布式（MCCL）和 ShardedArray
- 分布式是 v2+ 范围
- 大规模离线（Dask/Spark-GPU 风格）是远期目标

### L0-3：生态位

**决策**：独立可用库。不是 PyTorch 后端。不优先 PyTorch 兼容。

**影响**：
- DLPack 互操作会做，但与 PyTorch 的跨库验证推迟
- `__torch_function__` 和 autograd 集成是 v2+/v3+
- musapy 是产品，不是其他框架的基础设施

### L0-4：技术栈

**决策**：Python + Rust + MUSA。锁定，不可协商。

**依据**：
- Python：用户面向 API，NumPy/SciPy 兼容接口
- Rust：内存安全，PyO3 FFI，`Result` 错误处理
- MUSA：摩尔线程 GPU 栈（MUBLAS/MUDNN/MUSPARSE/MURAND/MUSOLVER/MUFFT/MCCL）

### L0-5：库别名

**决策**：推荐 `import musapy as ms`。不强制。不鼓励 `from musapy import *`。

**依据**：`ms` 简短、易记、不冲突。公共 API 通过 `ms.` 前缀访问，避免命名空间污染。
不强制别名允许用户偏好。

### L0-6：Device 解析 —— 5 级优先级链

**决策**：device 参数在 API 中始终存在，但其值通过 5 级链解析：

| 优先级 | 来源 | 说明 |
|---|---|---|
| 1 | 函数调用 `device=` 参数 | 最高，永远赢 |
| 2 | `with ms.device(...)` context | 覆盖全局 |
| 3 | 输入 Array 的 device（ufunc 风格） | `a + b` 跟 a 走 |
| 4 | 全局默认 device（`ms.set_default_device`） | 进程级，thread-local |
| 5 | 启动时 auto-probe | 有 MUSA 用 musa:0，否则 cpu |

**关键**：每一级可被上级覆盖。每一次解析都可追溯。

### L0-7：Dtype 解析 —— 与 device 对称

**决策**：dtype 遵循相同的 5 级解析链：

| 优先级 | 来源 |
|---|---|
| 1 | `dtype=` 参数 |
| 2 | `with ms.dtype(...)` context |
| 3 | 输入 Array 的 dtype（类型提升结果） |
| 4 | 全局默认 dtype（thread-local） |
| 5 | 启动默认 `ms.float32` |

**注意**：dtype 的 auto-probe 无意义（不像 device 有硬件探测），所以级 5 固定
`float32`。`DeviceNotConfigured` 只对 device 抛，dtype 永远有兜底。

### L0-8：Feedback 原则

**决策**：每次 device/dtype 解析必须产生可追溯的 `DeviceResolution` /
`DtypeResolution` 记录，附加到 Array，包含来源级别 + 来源位置。

**示例**：
```python
>>> a = ms.array([1,2,3])
>>> a.device
Device(musa:0)  # resolved from: global_default (musa:0), set via mp.set_default_device() at <stdin>:2
```

**依据**：这不是 perf 优化，是正确性保证。分布式调试时这是唯一能定位"为什么数据
跑到错设备"的手段。

### L0-9：首次创建必须显式设 device

**决策**：如果从未调用 `ms.set_default_device()`，第一次 `ms.array()` 抛
`DeviceNotConfigured`，而不是静默用 auto-probe。

**依据**：对科研用户，数据在 cpu 还是 musa 是天壤之别。静默选错会让结果对比失去
意义。

### L0-10：跨设备操作 —— strict policy

**决策**：跨设备操作（如 `a` 在 musa:0 + `b` 在 musa:1）抛 `DeviceMismatch`。
用户必须显式 `.to(device)`。

**依据**：隐式跨设备迁移是 perf 陷阱。显式 `.to()` 让数据移动可见。

### L0-11：默认 device 模型 —— 混合方案（thread-local + thread-safe runtime）

**决策**：
- 默认 device/dtype：**thread-local 栈**（per-thread 隔离，零锁）
- Runtime 基础设施（句柄表、内存池、stream 池）：**thread-safe**
  （RwLock/DashMap/AtomicPtr）
- 新线程在 `start()` 时继承父线程当前的默认（值快照，之后解耦）
- 不做广播 API。Worker 启动时自己读 config

**依据**：HPC 工作负载分析显示 thread-local 在 8 个维度中 6 个胜过全局共享。详见
设计笔记中的 ADR-L0-11 详细对比。

---

## Layer 1：核心抽象

### L1-1：Device 标识

**决策**：字符串（`"musa:0"`）和 `Device` 对象都支持。

```python
ms.array([1,2,3], device="musa:0")           # 字符串
ms.array([1,2,3], device=ms.Device.musa(0))  # 对象
```

### L1-2：不可用 device —— 启动即报错

**决策**：在只有 1 张 GPU 的机器上 `ms.set_default_device("musa:5")` 立即抛
`DeviceUnavailable`，不延迟到首次 op。

### L1-3：设备能力查询

**决策**：暴露 `device_count`、`arch`（计算能力）、`total_memory`（单设备）、
`total_memory_all_devices`（聚合）。

```python
ms.device_summary()
# musa:0 — MTT S4000, arch=mp_22, 47.9 GB VRAM, 56 CUs
# musa:1 — MTT S4000, arch=mp_22, 47.9 GB VRAM, 56 CUs
```

### L1-4：Phase 1 dtype 集

**决策**：15 种 dtype，预留扩展位。

```
bool, int8, int16, int32, int64,
uint8, uint16, uint32, uint64,
float16, float32, float64, bfloat16,
complex64, complex128
```

**依据**：科研用户需要复数（FFT、信号处理）和 bfloat16（数值实验对比）。缺了就
fallback 到 NumPy，体验割裂。

### L1-5：类型提升 —— JAX 风格 type-based

**决策**：用 JAX 的 type-based 提升表。不做 value-based 推断（避免 NumPy 的
`int8 + 1 → int64` 陷阱）。

### L1-6：bfloat16 on CPU

**决策**：CPU 上 bf16 运算自动提升到 f32 计算，再 round 回 bf16。

**依据**：CPU 无 bf16 硬件支持。明确文档化，非隐式。

### L1-7：默认 stream

**决策**：每设备一个 default stream，由 runtime 持有。`ms.array(...)` 无显式
stream 时绑定到该设备的 default stream。不暴露 null stream。

**依据**：null stream 的隐式同步语义是 CUDA 历史包袱。新库不应继承。default
stream 避免强制每个 op 都 `with stream:`。

### L1-8：`out=` 参数的 stream 语义

**决策**：op 在 `out` 的 stream 上执行。runtime 自动为输入 stream 插入 wait。

```python
with ms.stream(s1): a = ms.array(...)
with ms.stream(s2):
    b = ms.array(...)
    c = ms.empty(...)
    ms.add(a, b, out=c)   # 在 s2 执行，自动 wait s1（为 a）
```

**依据**：符合"out 是结果容器"的直觉。比报错更友好。debug 模式记录所有自动
插入的 wait（feedback 原则）。

### L1-9：Stream 优先级

**决策**：通过 `ms.Stream(device, priority=...)` 暴露。对标 MUSA stream 优先级。

### L1-10：Buffer 读写引用分离

**决策**：
- `Arc<Buffer>`：可写，唯一所有权语义
- `BufferRef(Arc<Buffer>)`：只读共享视图
- op 输入自动降级为 `BufferRef`；输出是新 `Buffer`

**依据**：使 kernel 能用 `__restrict__`（编译器可假设无别名）。编译期 aliasing 检测
（同一 `BufferRef` 不能同时作输入和 `out`）。

### L1-11：0-dim Array

**决策**：不特殊标量路径。`shape=[]` 就是 0-dim。MUSA runtime 自动优化。但
`.item()` / `__float__` / `__int__` 显式触发 `stream.synchronize` + D2H 拷贝。

```python
a = ms.array(3.14, device="musa:0")  # 0-dim 在 GPU
a + 1                                  # OK，不同步，结果是 0-dim GPU Array
float(a)                               # 触发同步 + D2H
```

### L1-12：执行模型 —— eager + lazy 钩子

**决策**：eager 执行为主。op 函数内部用 `OpBuilder`，将参数解析（一次）与
kernel launch（可重放）分离。为未来 MUSA Graphs capture 保留 lazy 钩子，不破坏 API。

**约束**：所有 op 函数必须 **capture-safe** —— 执行阶段不读 host 端可变状态。

### L1-13：Device policy —— strict 默认

**决策**：默认 policy 是 `strict`：跨设备 op 抛 `DeviceMismatch`。

"Musa > CPU" 层级 **仅** 用于 `auto` 默认 device 探测偏好。**不影响** op 行为。

### L1-14：GPU 精度对齐（dtype policy）

**决策**：两段式规则（默认行为，不需要 opt-in）：

| 场景 | 规则 |
|---|---|
| 全 GPU 运算（所有输入在 MUSA） | 结果 dtype = 输入 dtype 中 **最窄** 的（f16 > bf16 > f32 > f64，性能优先） |
| 含 CPU 运算（任一输入在 CPU） | 走 JAX 标准提升表（正确性优先） |

**同位宽冲突规则**：bf16 + f16（都 16-bit）→ JAX 提升 → f32（避免精度损失）。

**扩展表**（musapy 官方类型提升规范）：

| 输入组合（全 GPU） | 结果 | 理由 |
|---|---|---|
| f16 + f32 | f16 | 窄优先 |
| bf16 + f32 | bf16 | 窄优先 |
| f16 + bf16 | f32 | 同位宽冲突 → JAX |
| f32 + f64 | f32 | 窄优先 |
| f32 + i32 | f32 | 整型提升到浮点（JAX），GPU 窄 → f32 |
| f32 + i64 | f32 | 同上——整数**不因位宽升级浮点**（i64+f32 仍为 f32，v0.2 计划 §1.3 一致） |
| i32 + i64 | i32 | 整型窄优先 |
| i32 + u32 | i64 | JAX（有符号+无符号可能溢出） |
| f32 + complex64 | complex64 | 复数窄优先 |
| complex64 + complex128 | complex64 | 窄优先 |
| bool + f32 | f32 | bool 提升到浮点 |

### L1-15：Green Context

**决策**：v1 不用 Green Context。thread-local 默认 + thread-safe runtime 已够用。
Green Context 推迟到 v2+。

**依据**：Green Context 文档稀少。先用成熟的 thread-local + thread-safe 方案。
Green Context 是优化项，非必需。

### L1-16：OpBuilder 与 MUSA Graphs

**决策**：OpBuilder 的 lazy 钩子用 MUSA Graphs API（不自建 DAG）。

**影响**：所有 op 必须 capture-safe（参数解析与 kernel launch 可分离）。

---

## Layer 2：模块契约

### L2-1：Build System

**决策**：
- `maturin` + Cargo workspace + `mcc` 编译 `.mu` → `.o` → `libmusapy_kernels.a`
- ABI 版本嵌入符号名：`musapy_mul_f32_v1`
- runtime 启动时校验 kernel ABI
- MUSA SDK 探测：`MUSA_HOME` env + pkg-config 双重探测
- MUSA Runtime 版本（来自 musart_version.h）与运行时 ABI 版本兼容性矩阵检查

**禁止**：
- build 脚本含运行时逻辑
- 硬编码 MUSA 路径

### L2-2：MUSA Kernels（`kernels/*.mu`）

**职责**：纯并行计算 kernel。线程网格逻辑。无分支数学。设备端内存访问（只读输入、
只写输出）。

**允许依赖**：`musa_runtime.h`、`include/` 头文件、MUSA intrinsic。

**禁止**：
- 内存分配/释放（`malloc`、`musaMalloc`）
- 主机端代码（`printf`、文件 I/O）
- 错误返回（kernel 返回 `void`）
- 调度逻辑（grid/block 大小决策）
- 运行时类型分支（`if dtype==...` —— 必须模板实例化）
- 跨设备操作

**接口契约**：纯 C，无状态：`extern "C" void musapy_<op>_<dtype>_v<abi>(...)`。
所有指针 `__restrict__`（由 ops 层 alias 检测保证）。

### L2-3：Core Runtime（`rust/musapy-core`）

**职责**：
- 数据结构定义与不变式维护（Array / Buffer / BufferRef / Device / Dtype /
  Stream / Layout / DeviceResolution）
- 内存生命周期（RAII + stream-ordered dealloc）
- 线程安全全局基础设施（设备表、内存池、stream 池、MUBLAS handle 表）
- thread-local 默认 device/dtype 栈
- DLPack 互操作

**允许依赖**：仅 MUSA runtime API（**不**调用 MUBLAS/MUDNN 等计算库）。标准
Rust crate。

**禁止**：
- 任何算子实现
- 调用 MUBLAS/MUDNN/MCCL
- 算子调度
- 修改 Array 的 device/dtype 字段（只读使用）
- 直接管理 Python 对象/GIL

**线程安全分层**：
```
全局只读（启动后不变）：
  设备表、能力、ABI 版本 → 无需锁

全局可变（thread-safe）：
  内存池 → RwLock<MemoryPool>
  Stream 池 → DashMap<DeviceId, Arc<Stream>>
  MUBLAS handle 表 → thread_local<Handle>（handle 本身不线程安全）

thread-local：
  默认 device 栈 → RefCell<Vec<Device>>
  默认 dtype 栈 → RefCell<Vec<Dtype>>
  当前 stream 栈 → RefCell<Vec<Arc<Stream>>>
```

### L2-4：Ops 层 —— capture-safe 约束

**决策**：所有 op 函数必须 capture-safe：
- 参数解析（shape/dtype/device 检查）只执行一次
- kernel launch 可重放
- 执行阶段不读 host 端可变状态
- 参数解析与 kernel launch 在 `OpBuilder` 中分离

### L2-5：Ops 层 —— alias 检测

**决策**：同一 `BufferRef` 不能同时作 op 输入和 `out` 参数。违例抛 `AliasDetected`
错误。不自动 copy。

**依据**：使 kernel 能用 `__restrict__`。通过 Buffer/BufferRef 类型分离实现编译期保证。

### L2-6：PyO3 绑定 —— stream-aware DLPack export

**决策**：`__dlpack__(stream)` 导出：
1. 记录当前 array 的 pending write event
2. 若消费方传了 stream，让它 wait 我们的 event
3. capsule 持有 event 引用（防止 buffer 在 event 完成前被释放）

### L2-7：Python 前端 —— context 组合

**决策**：`ms.device()` / `ms.dtype()` / `ms.stream()` context manager 对称且可组合。
支持任意嵌套和元组简写：

```python
with ms.device("musa:0"), ms.stream(s1), ms.dtype(ms.float16):
    ...
```

### L2-8：Python 前端 —— 导入方式

**决策**：不鼓励 `from musapy import *`。不强制 `import musapy as ms`。公共 API
通过 `musapy.` 前缀访问。文档示例用 `ms` 别名。

---

## Layer 3.1：错误模型

### L3-1：双层检测

**决策**：
- **Launch 错误**（参数非法、句柄无效）：op 排队后立即 `musaGetLastError` 检查。
  在 op 调用处报。
- **执行错误**（越界、NaN）：延迟到 `stream.synchronize()`。从 stream 的 pending
  队列取 op 上下文报。

### L3-2：OpContext 归因

**决策**：每个 op 排队时记录 `OpContext` 到 stream 的 pending 队列：

```rust
pub struct OpContext {
    pub op_name: &'static str,        // "matmul"
    pub input_shapes: Vec<Shape>,
    pub input_devices: Vec<Device>,
    pub input_dtypes: Vec<Dtype>,
    pub output_shape: Shape,
    pub stream_id: u64,
    pub python_frame: Option<PythonFrame>,  // 仅 debug 模式
    pub timestamp: Instant,
}
```

synchronize 报错时，找最后一个未完成的 op（最可能是根因），附在错误消息里。

### L3-3：Poison 恢复

**决策**：
- op 执行失败标记 stream `poisoned: AtomicBool`
- poisoned stream 上所有后续 op 立即返回 `PoisonedStream`（不再排队）
- v1 提供 `stream.reset()`（标注 `@experimental`）：销毁 stream + 使该 stream 上
  所有 buffer 失效。**不保证** context 一致性。生产环境应重启进程。
- `ms.reset_device()` 推迟到 v2+

### L3-4：capture 模式错误

**决策**：
- 参数错误 + launch 错误：capture 时立即报（musaGraphAddNode 会校验）
- 执行错误：`graph.replay()` 时报，归因到 graph node，映射回 OpContext

### L3-5：异常层级深度

**决策**：浅继承，两层层级：

```
MusapyError
├── DeviceError
├── DtypeError
├── ShapeError
├── MemoryError
├── StreamError
├── KernelError
└── InteropError
```

### L3-6：Python 内置异常继承

**决策**：部分继承 Python 内置：

```
MusapyError(Exception)
├── DeviceError(MusapyError, RuntimeError)
├── DtypeError(MusapyError, TypeError)
├── ShapeError(MusapyError, ValueError)
├── MemoryError(MusapyError)                    # 见 L3-7
├── StreamError(MusapyError, RuntimeError)
├── KernelError(MusapyError, RuntimeError)
└── InteropError(MusapyError, RuntimeError)
```

**依据**：科研用户从 NumPy 迁移，`except ValueError` 是肌肉记忆。部分继承保持兼容。

### L3-7：OutOfMemoryError —— 不继承内置

**决策**：`OutOfMemoryError(MusapyError)` **不**继承 Python 内置 `MemoryError`。

**依据**：GPU 显存不足与 Python 堆内存不足语义不同。混在一起会误导用户。

**完整异常层级**：

```
MusapyError(Exception)
├── DeviceError(MusapyError, RuntimeError)
│   ├── DeviceNotConfiguredError
│   ├── DeviceMismatchError
│   └── DeviceUnavailableError
├── DtypeError(MusapyError, TypeError)
│   └── UnsupportedDtypeError
├── ShapeError(MusapyError, ValueError)
│   └── ShapeMismatchError
├── MemoryError(MusapyError)
│   ├── OutOfMemoryError
│   └── AliasDetectedError
├── StreamError(MusapyError, RuntimeError)
│   ├── PoisonedStreamError
│   └── SyncCycleError
├── KernelError(MusapyError, RuntimeError)
│   ├── LaunchFailedError
│   └── KernelFailedError
└── InteropError(MusapyError, RuntimeError)
    ├── DlpackExportError
    └── UnsupportedProtocolError
```

---

## Layer 3.2：内存生命周期

### L3-8：内存池 —— 3 层结构

**决策**：

| 层 | 职责 |
|---|---|
| L1 MUSA runtime | `musaMallocAsync` / `musaFreeAsync`，stream-ordered |
| L2 musapy BufferPool | per-device 池，按 size class 分桶复用 |
| L3 Buffer（用户层） | RAII 句柄，Drop 时归还到池（不立即 free） |

**GC 策略**：B（定期 + LRU）。默认：每 60 秒或显式 `ms.gc(device)`，释放超过 5 分钟
未用的 buffer。

**用户可配**：`ms.set_memory_policy("aggressive" | "lazy" | "manual")`

**实现状态（Phase C-lite, 2025-07）**：L2 BufferPool 已实现（`buffer_pool.rs`），
仅默认路径编译（`#[cfg(not(feature = "stream-ordered"))]`）。设计参数：
- SizeClass = round_up_pow2(size)，最小 512 bytes
- 每设备缓存上限 512 MB，超出 fallback 到 deferred-free（L3-11）
- 复用时若 stream 不同，wait on stored event（跨 stream 安全）
- 复用时 `actual_size >= requested_size` 校验（同 size-class 内可能有更小条目）
- GC 策略（LRU 淘汰、`ms.gc()`）尚未实现，当前仅容量上限控制

### L3-9: Stream-Ordered Dealloc — 条件实现（feature gate）

**决策**: v1 同时支持两种路径，用 Cargo feature gate + runtime probe 选择：

| 构建模式 | feature | alloc/free API | 适用 SDK |
|---|---|---|---|
| 默认 | （无） | musaMalloc / musaFree + deferred-free 队列 | 3.x / 4.x / 5.x |
| stream-ordered | `stream-ordered` | musaMallocAsync / musaFreeAsync | 5.x+ |

**比较**: MUSA Runtime 3.x/4.x 的 libmusart.so 不含 musaMallocAsync/musaFreeAsync
符号（3.1.0/3.3.5/4.3.7 实测确认）。MUSA SDK 5.1.0 Release Notes 明确"新增支持
Stream Ordered Memory Allocator API"（对标 CUDA 12.8），但 5.x 目前受限发布。
为保证单代码库兼容所有版本，用 feature gate 控制 async API 的链接声明，
runtime probe 做双重保险。

**实测版本矩阵（2025-01）**：

| MUSA Runtime | musaMallocAsync | musaFreeAsync | musaMalloc/Free |
|---|---|---|---|
| 3.1.0 | 头文件有声明，.so 无符号 | 头文件有声明，.so 无符号 | ✅ 可用 |
| 3.3.5 | 头文件有声明，.so 无符号 | 头文件有声明，.so 无符号 | ✅ 可用 |
| 4.3.7 | C++ inline 包装（转 musaMallocFromPoolAsync） | 仅声明，无实现 | ✅ 可用 |
| 5.1.0 | ✅ 完整 | ✅ 完整 | ✅ 可用 |

**未来**: 待 5.x 公开普及后，将 `stream-ordered` 改为 default feature，
或直接删除 feature gate，统一走 stream-ordered 路径。

### L3-10：dealloc stream 选择策略

**决策**：策略 **b**（最后使用 stream）。Buffer 的 `dealloc_stream` 可变，跨 stream
使用时更新。

**优化**：`read_events` 只存尚未被 `dealloc_stream` wait 过的 event。wait 后 pop。
Vec 通常 0-1 个元素。

**Phase C-lite 同 stream 优化（2025-07）**：Buffer 新增 `last_write_stream_id: AtomicU64`。
当读写操作在同一 stream 上时（单 stream 常见场景）：
- `wait_last_write_on`：同 stream 直接 return Ok（跳过 musaStreamWaitEvent）
- `record_write`：同 stream 连续写跳过 Event::new/Record（同 stream 隐式有序）
- `record_read`：读写同 stream 跳过 event 创建
实测减少 ~6 次 driver 调用/op，小数组延迟降低 ~39%。

```rust
impl Drop for Buffer {
    fn drop(&mut self) {
        let dealloc = self.dealloc_stream.lock().unwrap().clone();
        for ev in self.read_events.lock().unwrap().drain(..) {
            dealloc.wait_event(&ev);
        }
        if let Some(ev) = self.last_write_event.lock().unwrap().take() {
            dealloc.wait_event(&ev);
        }
        unsafe { musaFreeAsync(self.ptr, dealloc.raw()); }
    }
}
```

### L3-11: Deferred-Free — 默认路径

**决策**: deferred-free 是默认构建路径，适用于所有 SDK 版本（3.x/4.x/5.x）。
stream-ordered（L3-9）作为可选 feature，5.x 环境可启用。

**工作流程**：
1. `Buffer::drop` 不立即 free，而是 `(ptr, events)` 入 deferred-free 全局队列
2. 入队前在 `dealloc_stream` 上 wait 所有 read/write events（策略 b 保证）
3. `Stream::synchronize` 成功后，批量 reclaim：对队列中所有 buffer 调用 `musaFree(ptr)`

**安全保证**：
- synchronize 保证流上所有 op 完成
- 入队前已 wait events，synchronize 后 events 一定已完成
- 所以 reclaim 时 buffer 一定不再被任何流使用

**与 L3-9 的关系**：L3-11 是 L3-9 不可用时的 fallback，也是当前默认路径。
启用 `stream-ordered` feature 后，Buffer 走 L3-9 路径，deferred-free 队列
不再被使用（但代码保留，feature 关闭时自动恢复）。

**与 L3-8 BufferPool 的关系（Phase C-lite）**：默认路径下 `Buffer::drop` 先尝试
归还 BufferPool 复用；池满（超 512 MB/设备）时才 fallback 到 deferred-free 队列。
即：BufferPool 是热路径，deferred-free 是冷路径兜底。

**Capability probe**：启动期探测 `musaDeviceGetAttribute(MUSA_DEV_ATTR_MEMORY_POOLS_SUPPORTED)`。
即使编译了 `stream-ordered` feature，probe 也作为双重保险——如果运行时不支持，
回退到 deferred-free。

### L3-12：DLPack 生命周期

**决策**：DLPack capsule 持有 `Arc<Buffer>`。引用计数保证 buffer 在 capsule 释放前
不被回收。capsule 释放 → Arc 减 → 可能触发 Drop → 走正常 stream-ordered free 流程。

### L3-13：Poison 恢复 —— 保守策略

**决策**：`stream.reset()` 销毁该 stream 上所有 buffer，标记失效。后续访问抛
`BufferInvalidated`。

**依据**：MUSA 错误是 sticky 的。无法精确判定哪些 buffer 受影响。保守策略避免
use-after-poison。

### L3-14：Graph placeholder 语义

**决策**：
- capture 期间 Array 标记 `is_graph_placeholder: true`
- 普通 op 接收 placeholder 抛 `GraphNotReplayed`（capture 外不能用）
- `graph.replay()` 后 placeholder buffer 被填充，Array 转为普通
- v1 replay 同步

### L3-15: Minimal Verification Test

**决策**: 在版本1（v1）实现之前，使用MUSA硬件运行一个最小的跨流分配/使用/释放测试，以验证流顺序释放在实际MUSA硬件上是否可行。

**现状(2025-01)**: 已在 MUSA Runtime 3.1.0 / 3.3.5 / 4.3.7 上确认 stream-ordered API
不可用（musaFreeAsync 无实现），走 deferred-free fallback。5.1.0 环境待验证。

**测试**: 5.x 环境可用后，运行 stream-ordered 验证测试（见 v0.1-alpha-plan 2.1 节）。
通过则可启用 `stream-ordered` feature 作为推荐构建方式。

---

## Layer 3.3：互操作

### L3-16：DLPack MUSA 设备类型

**决策**：自定义 `kDLMUSA`（用 DLPack reserved 范围，如值 100）。稳定后向上游 DLPack
申请官方枚举值。

### L3-17：`__array_ufunc__` 跨设备

**决策**：strict policy 下，`np.add(musapy_array_on_gpu, numpy_array_on_cpu)` 抛
`DeviceMismatch` + 明确修复消息：

```
ms.DeviceMismatch: np.add() received inputs on different devices
  musapy Array on musa:0
  numpy.ndarray on cpu
  fix: convert numpy array to musapy first, e.g. np.add(a, ms.array(b, device=a.device))
  or:  convert musapy array to numpy first, e.g. np.add(a.cpu().numpy(), b)
```

### L3-18：`__array_function__` v1 范围

**决策**：v1 只支持高频函数：`concatenate`、`stack`、`split`、`zeros_like`、
`ones_like`、`where`。其余 fallback 到 NumPy（触发 `.cpu()` 同步）。

预留扩展空间供未来添加。

### L3-19：`__array__` 隐式同步

**决策**：`np.array(musapy_array)` 触发 `stream.synchronize` + D2H 拷贝。不发
warning。文档标注为同步操作。

### L3-20：DLPack v1 实现

**决策**：v1 实现 DLPack 协议 + round-trip 验证（musapy 导出 → musapy 导入）。
跨库验证（torch_musa）推迟。

### L3-21：PyTorch 互操作

**决策**：v1 **不**验证与 PyTorch/torch_musa 的跨库互操作。`__torch_function__` 和
autograd 集成推迟到 v2+/v3+。

### L3-22：CuPy 互操作

**决策**：v1 **不**做专门的 CuPy 互操作。CuPy 是 CUDA-only，与 MUSA GPU 物理不兼容。
文档说明："如需与 CuPy 混用，用显式 `.cpu()` + `cp.asarray()`"。

### L3-23：跨设备/跨框架互操作

**决策**：v1 聚焦纯摩尔线程环境。不做跨设备或跨框架互操作验证。

### L3-24：互操作错误处理

**决策**：

| 错误场景 | 谁报错 | 错误类型 |
|---|---|---|
| DLPack export 时 buffer 已失效 | musapy | `InteropError.DlpackExport` |
| DLPack import 时 capsule 格式不对 | musapy | `InteropError` |
| 消费方在 event 完成前访问 buffer | 消费方 | 消费方自己的错 |
| `__array_ufunc__` 跨设备 | musapy | `DeviceMismatch` |
| `__array__` 时 stream 已 poison | musapy | `PoisonedStream` |

Debug 模式：capsule deleter 中 assert Arc strong count == 1，否则 panic。

---

## Layer 3.4：可观测性

### L3-25：Profiling —— 不自建

**决策**：musapy **不**自建 profiler。所有 profiling 需求用摩尔线程的 `msys profile`
（Moore Perf System）和 Moore Perf Compute（MCU）。

**依据**：Moore Perf System 已提供 kernel 时间轴、stream 泳道、API 追踪、GPU 指标。
Moore Perf Compute 提供 Roofline、kernel 级分析。musapy 不应重复造轮子。

**文档**：引导用户运行 `msys profile -t musa -o report.msys-rep python script.py`。

**OpContext**（来自 L3-2）仍记录，用于错误归因，**不**用于 profiling。

### L3-26：Debug 模式 —— 运行时 flag

**决策**：单一二进制，运行时 flag。`ms.set_debug(True)` 或 `MUSAPY_DEBUG=1` env var
或 `with ms.debug():` context。

**Debug 模式启用**：
- OpContext 记录 `python_frame`
- sync DAG 完整环检测（DFS）
- Buffer alias 检测 + 详细 dump
- Arc count assert（L3-24）
- 释放后的 buffer 填 `0xDEADBEEF`（use-after-free 可视化）
- op 参数完整 dump 到 log

**实现**：Rust `if debug` 分支，编译器消除 release 路径。debug 关闭时零开销。

### L3-27：Array 命名

**决策**：`name` 存在 Array 层（不在 Buffer）。`Array.name: Option<String>`。

**依据**：
- 同一 buffer 可能有多个视图（slice/transpose），每个视图名字不同更合理
- Buffer 是热路径数据结构，String 字段影响缓存局部性
- Array 数量 << Buffer 数量，开销可忽略

```python
a = ms.array(..., device="musa:0", name="weights.layer1")
# 或
a.name = "weights.layer1"
```

### L3-28：内存/stream 状态查询

**决策**：
- `ms.memory_summary(device)` / `ms.stream_summary()` / `ms.device_summary()`：用
  atomic 计数器（零锁），适合频繁监控
- `ms.memory_detail(device)`：遍历 BufferPool，明确标注有开销

**维护的 atomic 计数器**：
- `allocated_bytes`、`allocated_buffers`
- `cached_bytes`、`cached_buffers`
- `peak_bytes`、`peak_timestamp`

---

## Layer 4：演进策略

### L4-1：v1 ops 范围

**决策**：v1 实现：

| op 类别 | v1 | 来源 |
|---|---|---|
| elementwise（add/sub/mul/div/sin/cos/exp/log/pow/abs/sign/clamp） | ✅ | 自定义 `.mu` kernel |
| reduction（sum/max/min/mean/argmax/argmin/cumsum/prod） | ✅ | 自定义 `.mu` kernel |
| init（zeros/ones/arange/linspace/fill/eye） | ✅ | 自定义 `.mu` kernel |
| linalg（matmul/lu/qr/svd/solve） | ✅ | muBLAS + muSOLVER |
| random（rand/randn/uniform/normal/bernoulli） | ✅ | muRAND |
| fft（fft/fftn/ifft/rfft） | ✅ | muFFT |
| sparse（csr_matrix/coo_matrix/spmv/spmm） | ✅ | muSPARSE |
| indexing（slice/gather/scatter/transpose/permute/flip） | ✅ | 自定义 `.mu` kernel |
| broadcast | ✅ | 通过 elementwise 的 strides=0 |
| comparison（==/!=/</>/<=/>=/argmax） | ✅ | 自定义 `.mu` kernel |

**v1 不实现**：
- nn（muDNN: conv/pool/activation/batch_norm/softmax）—— v2+
- distributed（MCCL: all_reduce/all_gather/broadcast/send/recv）—— v2+

### L4-2：v1 排除项

| 项 | 推迟到 |
|---|---|
| 分布式（MCCL） | v2 |
| MUSA Graphs capture 实现 | v2（v1 只保留 OpBuilder 钩子） |
| PyTorch 互操作验证 | v2+ |
| `__torch_function__` / autograd | v2+/v3+ |
| Green Context | v2+ |
| `ms.reset_device()` | v2+ |
| kernel fusion / JIT | v3+ |
| Autotuning | v3+ |
| CPU fallback crate（`musapy-cpu`） | v2+（如果 adoption 需要） |
| ShardedArray | v2+ |
| StreamedArray | v2+ |

### L4-3：向后兼容政策

**决策**：SemVer + musapy 特殊说明：

| 变更类型 | 政策 |
|---|---|
| Python 公共 API 签名 | minor 不破坏；major 可破坏 |
| Rust crate 公共 API | 同上，Cargo SemVer 兼容 |
| Kernel ABI（符号名） | minor 不破坏（新 kernel 用 `_v2` 后缀，旧符号保留）；major 可破坏 |
| 默认行为变更（device/dtype policy） | minor 不改默认，加 opt-in；major 可改默认 |
| 实验性 API | 标注 `@experimental`，minor 可破坏，release notes 必须说明 |
| 错误消息格式 | 不保证兼容（用户不应 parse 错误消息） |

**实验性 API 毕业流程**：
- experimental → stable candidate（1 个 minor 版本）→ stable（下个 minor）
- candidate 阶段：收集反馈，完善 API
- stable 时：API 签名冻结，走标准兼容政策

### L4-4：Deprecation 流程

**决策**：

| 阶段 | 行为 |
|---|---|
| 1. 标注 deprecated（vX.Y） | API 仍可用，发 `DeprecationWarning`，文档显示替代方案 |
| 2. 保留 1 个 major 周期（v(X+1) 仍可用） | 继续发 warning，不删除 |
| 3. 删除（v(X+2)） | 真正移除 |

用 Python 标准库 `DeprecationWarning`（不自定义）。用户用标准
`warnings.filterwarnings` 静音。

### L4-5：预发布序列

**决策**：

| 版本 | 范围 |
|---|---|
| v0.1-alpha | 核心运行时（Device/Dtype/Stream/Array/Buffer） |
| v0.2-alpha | 基础 ops（elementwise/reduction/init/indexing/broadcast/comparison） |
| v0.3-alpha | 数学库 ops（linalg/random/fft/sparse） |
| v0.4-beta | 互操作 + 错误模型 + 可观测性完整 |
| v1.0-rc1 | 全部 v1 范围，公开测试 |
| v1.0 | 正式版 |

### L4-6：MCCL + Graphs

**决策**：MCCL 分布式和 MUSA Graphs capture 实现都推迟到 v2。

**依据**：
- MCCL 需要多 GPU 测试环境（≥2 张 MUSA GPU）
- MUSA Graphs API 成熟度需在真实硬件上验证
- v1.0 应尽快落地供用户使用
- 两者都是相对独立的模块，可作为 v1.1 或 v2.0 minor/major 添加

---

## 附录：决策索引

### 按层

| 层 | 决策数 | 范围 |
|---|---|---|
| Layer 0 | 11 | L0-1 至 L0-11 |
| Layer 1 | 16 | L1-1 至 L1-16 |
| Layer 2 | 8 | L2-1 至 L2-8 |
| Layer 3.1 | 7 | L3-1 至 L3-7 |
| Layer 3.2 | 8 | L3-8 至 L3-15 |
| Layer 3.3 | 9 | L3-16 至 L3-24 |
| Layer 3.4 | 4 | L3-25 至 L3-28 |
| Layer 4 | 6 | L4-1 至 L4-6 |
| **合计** | **69** | |

### 按状态

| 状态 | 数量 | ID |
|---|---|---|
| 已接受（最终） | 69 | 全部 |
| 实验性 | 1 | L3-3（`stream.reset()`） |
| 推迟到 v2+ | 11 | L1-15、L3-3（reset_device 部分）、L4-2 各项 |

### 关键交叉引用

- **Device policy**：L0-6、L0-9、L0-10、L1-13、L3-17
- **Dtype policy**：L0-7、L1-4、L1-5、L1-14
- **Stream 模型**：L1-7、L1-8、L1-9、L3-1、L3-2、L3-10
- **内存生命周期**：L1-10、L2-3、L3-8 至 L3-15
- **错误模型**：L3-1 至 L3-7、L3-24
- **互操作**：L2-6、L3-12、L3-16 至 L3-24
- **Capture-safety**：L1-12、L1-16、L2-4、L3-4、L3-14

---

## 变更记录

| 日期 | 变更 | 影响的 ADR ID |
|---|---|---|
| 2025-01-15 | 初始草案，全部 69 决策 | 全部 |

---

## 使用本 ADR 的说明

1. **引用决策**：在代码注释、issue、PR 中用稳定 ID，如 `ADR-L1-10` 或 `L1-10`。
2. **提议变更**：开新 ADR（如 ADR-002）supersede 特定决策。**不要**直接编辑本文件
   做变更。
3. **实验性项**：在单独的 `EXPERIMENTAL.md` 追踪毕业进度。
4. **实现追踪**：每个决策在 musapy repo 有对应 tracking issue。

---

*ADR 结束*
