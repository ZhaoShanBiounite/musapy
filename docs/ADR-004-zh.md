# musapy 架构决策记录 —— ADR-004：v0.4-beta 互操作与可观测性补充决策

> **状态**：草案（决策待 v0.4 实现定稿后转已接受）
> **最后更新**：2026-08-08
> **范围**：musapy v0.4-beta —— DLPack / NumPy 协议互操作 + 错误模型 + 可观测性完整
> **关系**：本文档是 [ADR-zh.md](./ADR-zh.md)（主 ADR，69 决策）的补充，与 [ADR-002-zh.md](./ADR-002-zh.md)
> （v0.2 补充，5 决策）、[ADR-003-zh.md](./ADR-003-zh.md)（v0.3 补充，7 决策）同级。按主 ADR「使用本 ADR 的说明」第 2 条，
> v0.4 新增决策单独成文，**不**直接编辑主 ADR。每个决策标注它扩展（extends）或澄清（clarifies）的主 ADR ID。

本文档记录主 ADR 未覆盖、但 v0.4 实现必须确定的决策。决策 ID 采用 `004-D<编号>` 形式，
便于在代码、issue、PR 中引用（如 `ADR-004-D1`）。

---

## 目录

- [004-D1：DLPack 依赖策略与 kDLMUSA 取值](#004-d1dlpack-依赖策略与-kdlmusa-取值)
- [004-D2：DLPack dtype 映射（Bool round-trip 有损）](#004-d2dlpack-dtype-映射bool-round-trip-有损)
- [004-D3：DLPack 生命周期与所有权](#004-d3dlpack-生命周期与所有权)
- [004-D4：NumPy 协议落地边界](#004-d4numpy-协议落地边界)
- [004-D5：错误模型扩展（DeviceMismatch + InteropError 落地）](#004-d5错误模型扩展devicemismatch--interoperror-落地)
- [004-D6：可观测性补全范围](#004-d6可观测性补全范围)
- [004-D7：v0.3 欠账与算子扩展排期](#004-d7v03-欠账与算子扩展排期)
- [004-D8：dtype 补全的可裁剪边界](#004-d8dtype-补全的可裁剪边界)
- [变更记录](#变更记录)

---

## 004-D1：DLPack 依赖策略与 kDLMUSA 取值

**扩展 / 澄清**：L3-16（DLPack MUSA 设备类型）、L3-20（DLPack v1 实现）

**决策**：

- v0.4 **不引入外部 DLPack 依赖**（系统无独立 `dlpack.h`，仅 torch 自带）。vendor 最小
  DLPack C 结构（`DLManagedTensor` / `DLTensor` / `DLDevice` / `DLDataType`，约 150 行）
  放 `musapy-core/src/dlpack.rs`，`repr(C)` 布局与 dlpack.h 一致。
- `kDLMUSA = 100`（DLPack reserved 范围，沿用 L3-16 的数值约定，不再另行商议）。
  稳定后向上游 DLPack 申请官方枚举值的动作保持 L3-16 原样。
- 结构体定义属 musapy-core（与 musa_ffi.rs 同层：FFI 基础设施）；capsule 构造/解析与
  PyO3 方法在 musapy-python（interop.rs）。

**依据**：L3-16 已定 kDLMUSA=100；vendor 避免把 torch 头文件路径变成构建依赖，保持
musapy 零第三方互操作依赖。

---

## 004-D2：DLPack dtype 映射（Bool round-trip 有损）

**扩展 / 澄清**：L3-20（round-trip 验证）、L1-4（dtype 系统）

**决策**：

- Dtype ↔ DLDataType 映射：
  - int/uint → `kDLInt`/`kDLUInt`，bits = 8/16/32/64
  - float16/32/64 → `kDLFloat`，bits = 16/32/64
  - bfloat16 → `kDLBfloat`，bits = 16
  - complex64/128 → `kDLComplex`，bits = 64/128（re/im 交错，位宽含双倍）
  - bool → `kDLUInt`，bits = 8（DLPack 无 bool 类型）
- **Bool round-trip 有损**：导出为 `kDLUInt/8`；导入 `kDLUInt/8` 按 PyTorch 惯例映射为
  **Uint8**（跨库一致性优先），Bool 需 `astype('b1')` 显式转回。文档化限制，不改协议。
- 向量 lanes ≠ 1 一律拒绝（`InteropError::UnsupportedProtocol`）。

**依据**：DLPack 规范无 bool；PyTorch 同款映射是事实标准，跨库互操作时行为一致比
musapy 内部保真更重要（L3-20 跨库验证虽推迟，映射先对齐惯例）。

---

## 004-D3：DLPack 生命周期与所有权

**扩展 / 澄清**：L3-20（DLPack v1 实现）

**决策**：

- **导出**：`Array.__dlpack__(stream=None)` 构造 `DLManagedTensor`，`manager_ctx` 持有
  buffer 的 `Arc` 计数（`Arc::into_raw`）；capsule deleter 调用时 `Arc::from_raw` 归还。
  buffer 不因导出而释放（capsule 未销毁前内存有效）。
- **导入**：`ms.from_dlpack(obj)` 调用 `obj.__dlpack__()` 取 capsule；解析 `DLTensor`
  并校验 device/dtype/shape。MUSA 设备 → 零拷贝接管（新 `BufferRef` 指向同一内存）；
  CPU 设备 → 构造 host Array（指针即 host 内存）。deleter 语义：musapy 接管后主动调用
  原 deleter 释放结构（数据所有权随 DLPack 协议转移）。
- **跨 device**：导出时若 Array 在 GPU 而消费者期望 CPU（如 `np.array`），由消费方
  （`__array__`）触发 sync + D2H，DLPack 导出本身不改 device。
- stream 同步：`__dlpack__(stream)` 可选参数记录生产者 stream；默认导出不强制 sync
  （协议要求消费者在 device 上等待，musapy 内部 round-trip 由现有事件机制保证）。

**依据**：DLPack 协议的所有权模型（deleter 负责释放）；Arc 计数避免 GPU buffer 在
capsule 存活期间被 pool 回收。

---

## 004-D4：NumPy 协议落地边界

**扩展 / 澄清**：L3-17（`__array_ufunc__`）、L3-18（`__array_function__`）、L3-19（`__array__`）

**决策**：

- `__array__(dtype=None, copy=None)`：`np.array(ms_arr)` 触发 `stream.synchronize` + D2H，
  返回 np.ndarray。**不发 warning**（L3-19），文档标注为同步操作。
- `__array_ufunc__`：strict policy。同 device 输入内委托既有 ops（`__array_ufunc__` 内
  直接走 `musapy_ops`，不经 NumPy）；跨 device 抛 **`DeviceMismatch`**（继承 `MusapyError`）
  + 修复消息（L3-17 的格式：列出两侧 device 与两条修复路径）。`out=` 参数支持。
- `__array_function__`：v1 范围 `concatenate` / `stack` / `split` / `zeros_like` /
  `ones_like` / `where`，其余 fallback NumPy（触发 `.cpu()` 同步）。预留扩展空间（L3-18）。
- 便捷方法：`.cpu()`（D2H 拷贝，返回 CPU Array）、`.numpy()`（→ np.ndarray，含 sync）。
- **既有 dunder 优先**：`__eq__` 等已实现返回 Array 的 dunder 不受 `__array_ufunc__`
  影响（NumPy 对 `__eq__` 走 `__array_ufunc__`，本库 `__eq__` 返回 Array 属协议允许的
  `NotImplemented` 分支，需全量回归把关）。

**依据**：L3-17/18/19 的设计原文；严格模式避免静默隐式拷贝造成性能陷阱。

---

## 004-D5：错误模型扩展（DeviceMismatch + InteropError 落地）

**扩展 / 澄清**：L3-5（错误模型）、L3-17（跨设备错误）

**决策**：

- 新增 **`DeviceMismatch`** 异常类（继承 `MusapyError`），错误消息按 L3-17 固定格式。
- `InteropError` 落地：补 `DlpackImport` 变体（与既有 `DlpackExport` 对称）；Python 侧
  `InteropError` 类已注册（v0.3），映射补全 import/协议错误。
- DLPack 导入失败、未知 dtype、未知 device 类型一律抛 `InteropError`（不 panic、不静默降级）。

**依据**：L3-5 错误分层 + v0.3 既有的 `InteropError` 注册（error.rs 预留变体）。

---

## 004-D6：可观测性补全范围

**扩展 / 澄清**：L3-26（Debug 模式）、L3-28（内存/stream 状态查询）

**决策**：

- Debug 模式（`ms.set_debug(True)` / `MUSAPY_DEBUG=1` / `with ms.debug():`）补全
  L3-26 未实现项：sync DAG 完整环检测（DFS）、buffer alias 检测 + dump、Arc count assert、
  释放后 buffer 填 `0xDEADBEEF`、op 参数完整 dump 到 log。`python_frame` 已实现（v0.1），
  保持。Rust `if debug` 分支，release 零开销不变。
- 新增 `ms.memory_detail(device)`（遍历 BufferPool，标注开销）与 `ms.stream_summary()`；
  与既有 `memory_summary` / `device_summary` 共用 atomic 计数器（L3-28）。
- **不自建 profiler**（L3-25）：性能分析引导用 `msys profile`，OpContext 仅错误归因。

**依据**：L3-25/26/28 原文；v0.1 起仅实现 python_frame，其余项随 L4-5「可观测性完整」
在 v0.4 补全。

---

## 004-D7：v0.3 欠账与算子扩展排期

**扩展 / 澄清**：ADR-003 003-D6（dot）、002-D4（高级索引推迟项）、v0.3 plan §1.2 范围外表

**决策**：

- **v0.3 欠账纳入 v0.4**（此前标注"v0.3 后期"，随 v0.3 收尾平移）：
  - fftn / ifftn / rfftn / irfft / fftfreq（多轴 FFT）
  - coo_matrix / coo→csr 归并（muSPARSE）
  - 高级索引 GPU kernel 接入（`musapy_adv_gather_*_v2` / `musapy_nonzero_bool_v2`
    替换 host fallback；若 mcc 指针数组 error 999 无绕行方案则保持 host fallback 并标注）
  - 混合 basic+fancy 索引（`a[1:, [0,2]]`，当前抛 PyTypeError，v0.4 实现 NumPy 语义）
- **算子扩展**：eigh（musolver syevd，仅实数；复数因无 C/Z 变体另行评估）、
  batch matmul（`mublasSgemmStridedBatched`，3D+ 广播）、N-D 广播 dot（003-D6 v0.4+ 评估项）。
- **保持排除**：sort/argsort（muThrust parallel_for 未确认）、跨库 DLPack 验证（L3-20）、
  PyTorch/CuPy 互操作（L3-21/22）、MUSA Graphs / MCCL（v2）、kernel fusion / JIT（v3+）。

**依据**：v0.3 release note 排期与 plan §1.2 表；排除项按 L3-20/21/22、L4-6。

---

## 004-D8：dtype 补全的可裁剪边界

**扩展 / 澄清**：L1-4（dtype 系统）、ADR-002（计算白名单）

**决策**：

- v0.4 尝试补全：f16/bf16 创建/读取/cast/运算全链路（当前 4 处 `PyNotImplementedError`）、
  cast complex→real、complex 算子扩展（pow、unary exp/log/sin/cos/sign）、标量比较
  `a > 2.0`（mask 便捷路径）。
- **可裁剪边界**：上述为 Phase 8 可裁剪项。若 v0.4 主线（互操作 + 可观测性 + 欠账）耗时
  超预算，dtype 补全随 v0.4.x 版本发布，不阻塞 0.4.0-beta 主线。
- 复数 max/min/argmax/argmin 永久拒绝不变（002-D3）。

**依据**：dtype 缺口盘点（v0.3 代码注释标记）；范围管理优先于面面俱到。

---

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-08 | 起草 ADR-004（004-D1~D8），随 v0.4-beta 计划制定 |

*与 [v0.4-beta-plan-zh.md](./v0.4-beta-plan-zh.md) 同步维护；英文版要点待 v0.4 决策转已接受时
补 ADR-004.md。*
