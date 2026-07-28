# musapy 架构决策记录 —— ADR-002：v0.2 算子套件补充决策

> **状态**：草案
> **最后更新**：2026-07-28
> **范围**：musapy v0.2-alpha 基础算子套件（elementwise / comparison / broadcast / reduction / init / indexing）
> **关系**：本文档是 [ADR-zh.md](./ADR-zh.md)（主 ADR，69 决策）的补充。按主 ADR「使用本 ADR 的说明」第 2 条，
> v0.2 新增决策单独成文，**不**直接编辑主 ADR。每个决策标注它扩展（extends）或取代（supersedes）的主 ADR ID。

本文档记录主 ADR 未覆盖、但 v0.2 实现必须确定的 5 个决策。决策 ID 采用 `002-D<编号>` 形式，
便于在代码、issue、PR 中引用（如 `ADR-002-D3`）。

---

## 目录

- [002-D1：binary 算子启用类型提升](#002-d1binary-算子启用类型提升)
- [002-D2：broadcast = elementwise strides=0](#002-d2broadcast--elementwise-strides0)
- [002-D3：reduction 语义（axis / keepdims / 累加 dtype）](#002-d3reduction-语义axis--keepdims--累加-dtype)
- [002-D4：indexing 语义（view-vs-copy / 负索引 / step）](#002-d4indexing-语义view-vs-copy--负索引--step)
- [002-D5：init 算子的 dtype 与 device 解析](#002-d5init-算子的-dtype-与-device-解析)
- [变更记录](#变更记录)

---

## 002-D1：binary 算子启用类型提升

**扩展**：L1-5（JAX 类型提升）、L1-14（GPU 精度对齐）
**取代**：v0.1-alpha 的「binary 算子要求两输入 dtype 相等」约束（该约束是 v0.1 的实现简化，从未是 ADR 决策）

**决策**：v0.2 起，binary elementwise 与 comparison 算子**不再要求**两输入 dtype 相等。
结果 dtype 由 `promote(a, b, all_gpu)`（主 ADR L1-14 规则）计算，输入在运算前按需 cast 到结果 dtype。

**实现策略**（控制 kernel 复杂度）：
- elementwise/comparison kernel **只实例化同 dtype 版本**（`<T,T>→T`，comparison 为 `<T,T>→bool`），不为每种 `(in_a, in_b, out)` 三元组合做模板特化。
- 当输入 dtype ≠ 结果 dtype 时，ops 层先插入一个内部 `astype` cast（本身是一个 stride-aware elementwise op，见 002-D2），把输入 cast 到结果 dtype，再调用同 dtype kernel。
- 代价是多一次 cast pass；收益是 kernel 矩阵从 O(N³) 降到 O(N)，且复用 v0.1 已有的同 dtype kernel 路径。

**依据**：
- L1-5 已规定用 JAX 类型提升表，`promote()` 已在 musapy-core 实现并导出，但 v0.1 未启用。v0.2 是启用它的合适时机。
- 同 dtype 约束让用户必须手动 `astype`，违背 NumPy/CuPy 肌肉记忆（主 ADR L3-5 错误模型同样为此目标服务）。
- 「预 cast 再同 dtype kernel」是 CuPy 等库的常见折中，避免组合爆炸。

**影响**：
- comparison 算子结果恒为 `bool`；两输入先提升到共同输入 dtype 再比较。
- `ms.add(int32_array, float32_array)` 在 GPU 上 → L1-14 → `f32`，int32 输入被 cast 到 f32。
- cast 路径需要 `astype` 作为 v0.2 的内部基础设施（也作为公开 op `Array.astype()` 暴露）。
- CPU fallback 与 mock stub 同样遵循「先 cast 后同 dtype 计算」。

---

## 002-D2：broadcast = elementwise strides=0

**扩展**：L4-1（broadcast 行：「通过 elementwise strides=0 实现」）、L1-11（0-dim Array）

**决策**：broadcast 不是独立机制，而是 elementwise kernel 的 **per-operand stride** 能力。
广播形状按 **NumPy 规则**计算：从最右维对齐，每一维要么相等、要么其一为 1（或其中一个为 0-dim）。
某操作数在被广播的维上 stride 设为 **0**，kernel 读该维时索引不前进 → 元素被复制。

**kernel ABI 变更**（elementwise/comparison 通用）：

```c
// v0.1 ABI（仅 add，扁平连续）：
void musapy_add_f32_v1(const float* a, const float* b, float* c, size_t n, musaStream_t s);

// v0.2 ABI（stride-aware，N 维）：
void musapy_add_f32_v2(
    const float* a, const float* b, float* c,
    int ndim,
    const size_t* shape,     // 广播后的输出形状，长度 ndim
    const ssize_t* a_strides,// 以「元素」为单位，长度 ndim；广播维为 0
    const ssize_t* b_strides,
    musaStream_t s);
```

- 输出 `c` 恒为 **contiguous**（标准 row-major strides），不传 c_strides。
- stride 以**元素个数**为单位（非字节），kernel 内 `a[idx·stride]` 直接寻址。
- 输入可能本身非 contiguous（如 transpose view），其 strides 与广播 strides **组合**后传入。
- 符号版本 `_v2`：按 L4-3，`_v1` 符号在 v0.2 内保留（内部仍可用于 1D 连续快路径），新代码走 `_v2`。

**0-dim 标量广播**：`shape=[]` 的操作数广播到任意形状，所有维 stride=0（L1-11：标量是 0-dim Array，无特殊代码路径）。

**依据**：
- L4-1 已锁定 broadcast 用 strides=0，本决策给出可落地的 ABI。
- NumPy 广播规则是科研用户的既有直觉（L0-1 主要受众）。
- 输出恒 contiguous 简化下游（reduction/indexing 可假设输出连续）。

**影响**：
- Phase 1 必须先完成 ABI 改造，再用新 ABI 重写 `add`，然后其余 elementwise/comparison 算子直接复用。
- `common.h` 需新增 N 维索引→线性偏移的辅助（如 `offset_nd(idx, strides, ndim)`）。
- broadcast 形状计算 + stride 推导放在 ops 层（Rust），kernel 不含广播逻辑（符合 L2-2：kernel 无 dispatch）。

---

## 002-D3：reduction 语义（axis / keepdims / 累加 dtype）

**扩展**：L4-1（reduction 行）、L1-14（dtype policy，本决策对其在 reduction 场景做澄清）

**决策**：

| 项 | 规则 |
|---|---|
| `axis` | v0.2 支持 `None`（全 reduce → 0-dim）与 `int`（单轴）。`tuple` 多轴**推迟**到 v0.2 后期或 v0.3 |
| `keepdims` | `bool`，默认 `False`；`True` 时被 reduce 的维保留为长度 1 |
| 负 axis | 支持，`axis < 0` → `axis + ndim`（NumPy 行为）；越界抛 `ShapeError` |

**累加 / 输出 dtype 规则**（对齐 NumPy/CuPy，正确性优先）：

| 算子 | 整型输入 | 浮点输入 | 输出 dtype |
|---|---|---|---|
| `sum` / `prod` / `cumsum` | int64 累加 | 保持输入 dtype | 整型→int64；浮点→输入 dtype |
| `mean` | float64 累加 | float32→float32，float64→float64 | 恒为浮点 |
| `max` / `min` | 保持 | 保持 | 输入 dtype（无累加） |
| `argmax` / `argmin` | — | — | 恒为 int64（索引） |

- bool 输入：`sum` 按 int64 计数；`max/min` 返回 bool。
- 复数：v0.2 的 reduction **不支持**复数（`max/min/argmax/argmin` 对复数无全序），抛 `DtypeError`；`sum/mean` 复数支持推迟。

**依据**：
- 整型累加溢出是**正确性**问题，优先于 L1-14 的「窄优先」（后者针对 elementwise 的**性能**）。
- argmax/argmin 输出索引，int64 是 NumPy/CuPy 惯例。
- 复数无序，max/min 语义不成立，明确拒绝优于给出误导结果。

**影响**：
- reduction kernel 需 block-reduce（`common.h` 新增 block reduce 辅助）；`mean` = `sum / count`（count 为被 reduce 元素数）。
- axis reduce 需按该轴 stride 遍历；输出形状按 keepdims 推导。
- `cumsum` 是 scan，需独立 kernel（prefix sum），v0.2 先支持 `axis` 上的 cumsum。

---

## 002-D4：indexing 语义（view-vs-copy / 负索引 / step）

**扩展**：L4-1（indexing 行）、L3-27（Array 命名 —— 一个 Buffer 多个 view）

**决策**：

| 算子 | 实现 | view / copy | 需要 kernel |
|---|---|---|---|
| `transpose(a, axes=None)` | 重排 Layout strides | **view** | 否 |
| `permute(a, dims)` | transpose 的显式 dims 形式 | **view** | 否 |
| `flip(a, axis)` | stride 取负 + offset 调整 | **view** | 否 |
| `slice`（`a[i:j:k]`） | 调整 offset + strides + shape | **view** | 否 |
| `gather(a, indices, axis)` | 按索引取元素 | **copy** | 是（indexing.mu） |
| `scatter(a, indices, values, axis)` | 按索引写元素 | **copy**（返回新 Array） | 是（indexing.mu） |

**切片细则**（对齐 NumPy）：
- 负索引：`i < 0` → `i + dim_len`。
- step：支持正/负 step；负 step 等价于 flip + 正 step。
- 越界：start/stop **clamp** 到合法范围（NumPy 行为，不报错）；空切片合法（该维长度为 0）。
- v0.2 只做 **basic slicing**（整数 / slice）。**高级索引**（boolean mask indexing、fancy indexing 用整数数组）推迟到 v0.3+。

**view 语义保证**：
- view 与原 Array **共享 Buffer**（通过 `BufferRef`，符合 L1-10 读写引用分离），修改 view 即修改原数据。
- view 的 `name` 独立于原 Array（L3-27：name 在 Array 而非 Buffer 的动机正是多 view）。
- view 通常**非 contiguous**；因 elementwise 已 stride-aware（002-D2），view 可直接参与后续运算，无需先 copy。

**依据**：
- L3-27 明确 name 放在 Array 是因为「一个 buffer 可有多个 view（slice/transpose）」——本决策落实该暗示。
- view 零拷贝是性能关键（transpose/slice 不应触发 D2D 复制）。
- 高级索引语义复杂（布尔索引返回 copy、形状难预测），推迟以降低 v0.2 风险。

**影响**：
- transpose/permute/flip/slice 只需 Rust 侧 Layout 操作 + 新 `BufferRef`，**无 kernel、无内存分配**。
- gather/scatter 需 `indexing.mu`（新 `.mu` 文件，build.rs 需编译）。
- `Layout` 需补充 `transposed(axes)` / `sliced(range)` / `flipped(axis)` 等纯函数变换。

---

## 002-D5：init 算子的 dtype 与 device 解析

**扩展**：L0-6（device 5 级链）、L0-7（dtype 5 级链，level-5 = float32）、L0-9（首次创建需显式 device）、L3-18（`*_like`）

**决策**：

| 算子 | dtype 来源 | device 来源 |
|---|---|---|
| `zeros` / `ones` / `full` | L0-7 链：`dtype=` 参数 > `with ms.dtype()` > 全局默认 > **float32** | L0-6 链：`device=` 参数 > `with ms.device()` > 全局默认 > auto-probe |
| `eye` | 同 zeros/ones（无参数可推断 → L0-7 链 → float32） | 同上 |
| `arange` | **从参数推断**（NumPy 行为）：全整数参数→int64，含浮点参数→float64；显式 `dtype=` 覆盖 | 同上 |
| `linspace` | 默认 **float64**（NumPy 行为）；显式 `dtype=` 覆盖 | 同上 |
| `zeros_like` / `ones_like` | **继承输入 Array** 的 dtype（忽略全局默认） | **继承输入 Array** 的 device |

**首次创建规则**：`zeros/ones/full/eye/arange/linspace` 都是「首次创建」候选。若从未调用 `ms.set_default_device()`
且无 `device=` 参数、无 `with ms.device()` context，则按 L0-9 抛 `DeviceNotConfiguredError`（**不**静默 auto-probe）。

**依据**：
- zeros/ones/full/eye 无参数可推断 dtype，落到 L0-7 的 level-5（float32）——与库的 GPU float32 默认一致。
- arange/linspace 的 dtype 从参数推断是 NumPy 肌肉记忆（`ms.arange(3)` → int64，`ms.arange(3.0)` → float64）。
- `*_like` 继承输入是 L3-18 的明确范围，且符合「like 即同形同类型」直觉。
- L0-9 首次创建规则对 init 算子同样适用，保持一致的「显式 device」体验（L0-1）。

**影响**：
- init kernel 是「写值」kernel：zeros/ones/full 是 fill（按线性 idx 写常量）；arange/linspace 是 idx→value；eye 是条件写（`i==j`）。
- init 算子必须填充 `DeviceResolution`/`DtypeResolution`（L0-8 feedback），source 对应解析链层级。
- arange 的整数/浮点推断在 ops 层（Rust）完成，kernel 仍按确定的 dtype 实例化。

---

## 变更记录

| 日期 | 变更 | 影响的决策 ID |
|---|---|---|
| 2026-07-28 | 初始草案，5 个 v0.2 补充决策 | 002-D1 至 002-D5 |

---

## 与主 ADR 的交叉引用

- **类型提升**：002-D1 扩展 L1-5、L1-14
- **broadcast**：002-D2 扩展 L4-1、L1-11；kernel ABI 遵守 L2-1、L2-2、L4-3
- **reduction**：002-D3 扩展 L4-1；澄清 L1-14 在累加场景的适用边界
- **indexing**：002-D4 扩展 L4-1、L3-27；view 共享 Buffer 遵守 L1-10
- **init**：002-D5 扩展 L0-6、L0-7、L0-9、L3-18
- **错误模型**：全部决策的错误类型遵守 L3-5/L3-6/L3-7（ShapeError/DtypeError/DeviceError 等）
- **capture-safety**：全部新算子遵守 L1-12、L2-4（参数解析一次性、kernel launch 可重放）

---

*ADR-002 结束*
