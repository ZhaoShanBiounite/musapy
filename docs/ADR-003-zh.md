# musapy 架构决策记录 —— ADR-003：v0.3 数学库补充决策

> **状态**：草案（v0.3-alpha 实现尚未开始；决策随实现定稿，发布时转「已接受」）
> **最后更新**：2026-08-06
> **范围**：musapy v0.3-alpha 数学库 ops（linalg / random / fft / sparse）+ reduction 补全 + 高级索引
> **关系**：本文档是 [ADR-zh.md](./ADR-zh.md)（主 ADR，69 决策）的补充，与 [ADR-002-zh.md](./ADR-002-zh.md)
> （v0.2 补充，5 决策）同级。按主 ADR「使用本 ADR 的说明」第 2 条，v0.3 新增决策单独成文，
> **不**直接编辑主 ADR。每个决策标注它扩展（extends）或澄清（clarifies）的主 ADR / ADR-002 ID。

本文档记录主 ADR 未覆盖、但 v0.3 实现必须确定的 7 个决策。决策 ID 采用 `003-D<编号>` 形式，
便于在代码、issue、PR 中引用（如 `ADR-003-D2`）。其中涉及 SDK 事实的决策基于
**2026-08-06 对 /usr/local/musa（SDK 3.1.0）头文件的实测核对**。

---

## 目录

- [003-D1：MUSA-X FFI 层放置与 L2-3 边界澄清](#003-d1musa-x-ffi-层放置与-l2-3-边界澄清)
- [003-D2：句柄模型与生命周期（musolver 共享句柄）](#003-d2句柄模型与生命周期musolver-共享句柄)
- [003-D3：错误模型扩展（LinAlgError + 内置 IndexError）](#003-d3错误模型扩展linalgerror--内置-indexerror)
- [003-D4：CPU fallback 宿主库策略](#003-d4cpu-fallback-宿主库策略)
- [003-D5：复数语义](#003-d5复数语义)
- [003-D6：dot 算子补充](#003-d6dot-算子补充)
- [003-D7：数学库 Python API 形态与返回约定](#003-d7数学库-python-api-形态与返回约定)
- [变更记录](#变更记录)

---

## 003-D1：MUSA-X FFI 层放置与 L2-3 边界澄清

**扩展 / 澄清**：L2-3（Core Runtime 职责与禁止项）、L2-1（Build System）

**决策**：

| 组件 | 放置 | 理由 |
|---|---|---|
| `musa_x_ffi.rs`（5 库 extern 声明 + mock stub） | **musapy-core** | 与 musa_ffi.rs 同层；声明与句柄是基础设施 |
| `math_handle.rs`（handle/plan/generator/workspace 生命周期） | **musapy-core** | L2-3 职责表明列「MUBLAS handle 表」 |
| 计算调用（gemm/getrf/fft/spmv 等） | **仅 musapy-ops** | L2-3 禁止 core 做算子实现/调度 |
| MUSA-X 链接指令（`cargo:rustc-link-lib=mublas/musolver/murand/mufft/musparse`） | **musapy-ops/build.rs** | 链接随调用方；core 保持 musart-only |

**L2-3 边界澄清**：L2-3 的禁止项「调用 MUBLAS/MUDNN/MCCL」解释为**计算分发**
（发起 gemm 等数值运算）；extern 声明与句柄生命周期管理（Create/Destroy/SetStream/
版本查询）**不在禁止之列** —— L2-3 职责表自身就把「MUBLAS handle 表」列为 core 的
线程安全全局基础设施。musapy-core/build.rs 的 L2-3 注释（「不链接 mublas/mudnn/mccl」）
**继续有效**：core 不发出任何 MUSA-X 链接指令，extern 声明在 musapy-ops 发出链接后
于 musapy-python cdylib 最终链接期解析。

**依据**：
- 声明集中在 core 保证 mock stub 与真实 FFI 签名一致（沿用 musa_ffi.rs 的 L2-3 模式）。
- Rust 链接语义：`cargo:rustc-link-lib` 由依赖图中任一 crate 发出即对最终二进位生效，
  因此「声明在 core、链接在 ops」可行且职责清晰。
- 避免 musapy-core 依赖计算库 .so，保持 core 在无 MUSA-X 环境（仅 runtime）可编译。

**影响**：
- musapy-core 新增 musa_x_ffi.rs / math_handle.rs 两个模块；lib.rs 导出。
- musapy-ops/build.rs 扩展链接 + openblas 探测（见 003-D4）。
- 实现计划见 v0.3-alpha-plan Phase 1。

---

## 003-D2：句柄模型与生命周期（musolver 共享句柄）

**扩展**：L2-3（handle 表）、L3-9（deferred-free）、L3-10（dealloc stream 选择）

**决策**：

**每 device 4 类句柄**（懒创建、按 device 缓存、跨 op 复用）：

| 句柄 | 服务库 | 备注 |
|---|---|---|
| `mublasHandle_t` | muBLAS **+ muSOLVER** | 共享（见下） |
| `murandGenerator_t` | muRAND | Philox4x32_10 / DEFAULT 引擎 |
| `mufftHandle` | muFFT | plan 池，按 (shape, type, direction) 键缓存 |
| `musparseHandle_t` | muSPARSE | 泛型 API（SpMV/SpMM） |

**musolver 无独立句柄（SDK 3.1.0 实测）**：不存在 `musolverDnHandle_t` / `musolverDnCreate`
（与 cuSOLVER 不同）。全部例程形如 `musolverSgetrf(mublasHandle_t handle, ...)`，
返回 `mublasStatus_t`。因此 muBLAS 与 muSOLVER 共用同一 `mublasHandle_t`。

**生命周期规则**：
- 创建：首次使用时懒创建；创建失败抛 `DeviceError`。
- stream 绑定：每个 op 调用前 `mublasSetStream` / `musparseSetStream` / `murandSetStream`
  绑定当前 op 的 stream（沿用 v0.2「每 op 显式传 stream」惯例）。
  stream 类型无需转换：`MUstream` 与 `musaStream_t` 同为 `struct MUstream_st*`
  （musa.h / driver_types.h 实测）。
- 释放：device 注销或进程退出时经 **deferred_free 队列**延迟释放（L3-9 默认路径），
  dealloc stream 按 L3-10「最后使用 stream」策略。
- workspace：muSOLVER / muSPARSE 两段式（`buffer=nullptr` 查询 → 分配 → 复用），
  按 size 分桶缓存；workspace buffer 同样走 stream-ordered 释放。

**依据**：
- 2026-08-06 头文件级核对（musolver_functions.h / mublas internal headers / musparse-functions.h）。
- 句柄创建/销毁昂贵（驱动态资源），必须跨 op 复用；deferred_free 是 v0.1/v0.2 已验证的
  3.x SDK 释放路径（L3-9 实测矩阵）。

**影响**：
- math_handle.rs 提供统一 `with_mublas_handle(device, stream, |h| ...)` 风格 API。
- 泄漏回归：1e6 次句柄创建/销毁后 mem_stats 无增长（v0.3 计划 P1.7 / §14.2）。

---

## 003-D3：错误模型扩展（LinAlgError + 内置 IndexError）

**扩展**：L3-5（异常层级）、L3-6（Python 内置继承）

**决策**：

| 场景 | 异常 | 说明 |
|---|---|---|
| solve 奇异矩阵 / 分解失败（getrf info > 0 等） | **`LinAlgError`**（新增，继承 `MusapyError`） | 对齐 `numpy.linalg.LinAlgError` 语义 |
| 高级索引（mask / fancy）越界 | **Python 内置 `IndexError`** | NumPy 兼容；musapy 层级不新增此类 |
| 复数 max/min/argmax/argmin | `DtypeError`（维持 002-D3） | 复数无全序 |

**层级约束**：L3-5 的两层浅层级不变 —— `LinAlgError` 是 `MusapyError` 的直接子类。
**单继承**：PyO3 `create_exception!` 仅支持单继承；与 v0.2 实现一致
（L3-6 的多继承方案在 v0.2 已按单继承落地，本决策延续）。

**依据**：
- `numpy.linalg` 抛 `LinAlgError`、NumPy 索引越界抛内置 `IndexError` —— 肌肉记忆（L0-1）。
- 奇异检测在 ops 层完成（getrf 返回的 info / U 对角零检测），不依赖 kernel。

**影响**：
- musapy-core error.rs：新增 LinAlgError 变体；musapy-python error.rs：`create_exception!`
  注册 + `to_pyerr` 映射；越界走 `PyIndexError::new_err`。
- `python/musapy/__init__.py` 导出 `LinAlgError`；`_core.pyi` 同步。

---

## 003-D4：CPU fallback 宿主库策略

**澄清**：与 L4-2 的关系（L4-2 推迟的是**独立 `musapy-cpu` crate**，v2+）；扩展 L2-1（build 探测）

**决策（2026-08-07 修订）**：v0.3 及以后的数学库算子**一律 GPU-only，不建 CPU fallback**；
CPU 设备上调用抛 `DeviceError`（`linalg::require_musa`）。v0.2 及以前的算子
（elementwise/comparison/reduction/creation/indexing）CPU 设备支持**保持不变**。

修订前（已废弃）的决策：v0.3 数学库的 CPU fallback 是算子内嵌路径（linalg 走
OpenBLAS + 纯 Rust 朴素降级；fft/random/sparse 纯 Rust 实现）。废弃原因：
- mock 模式（`MUSAPY_MOCK_MUSA`）已提供无 GPU 环境下的 GPU 路径测试覆盖，
  CPU fallback 的测试价值冗余；
- OpenBLAS 系统依赖 + 双路径（openblas/朴素）分派显著增加维护与验证成本
  （本轮实施中即暴露 openblas 路径 2 个语义 bug）；
- 数学库（muBLAS/muSOLVER 等）本质是 GPU 库，CPU 侧无对等物。

**当前影响**：
- linalg.rs（matmul/dot/solve）：GPU-only；CPU 输入 → `DeviceError::Mismatch`
  （与 math_handle 对 CPU 的既有报错风格一致）。
- musapy-ops/build.rs：OpenBLAS 探测已移除（原产出 `musapy_openblas` cfg）。
- 测试矩阵：MUSA 真机 + mock（GPU stub）两条路径。
- 后续 Phase（qr/svd/eig/random/fft/sparse）按本策略实施，不再建 CPU 路径。

**实施澄清（2026-08-08，Phase 5 落地时确认）——算子族归属规则**：
「GPU-only 不建 CPU fallback」按**算子族**划分，不按 dtype/参数形态划分：

> v0.2 已有算子族（elementwise/comparison/reduction/creation/indexing）→ 无论后续
> 新增何种 dtype（如 Phase 5 的 complex64/128）或参数形态（如 Phase 7 的 axis=tuple、
> Phase 8 的 mask/fancy 索引），CPU 路径一律保留（003-D4 原文「CPU 设备支持保持不变」
> 即按算子族承诺）。v0.3 新增算子族（linalg/random/fft/sparse）→ 一律 GPU-only。

依据：
- 003-D4 原文列的就是算子族（elementwise/comparison/…），非 dtype 清单；
- 同一算子（如 `ms.add`）按 dtype 拆设备行为（f32 有 CPU、complex 抛错）割裂且
  打破 v0.2「CPU 支持保持不变」承诺；
- complex 的 CPU 路径是复用既有骨架的 dtype 分支（trait + 循环），非 OpenBLAS 双路径
  那种维护负担（003-D4 废弃 CPU fallback 的真正原因）。

**影响**：Phase 5 的 complex elementwise（v0.2 算子族的 dtype 扩展）保留 CPU 路径；
fft 套件（v0.3 新算子族）GPU-only（`fft_impl` 首段 `require_musa`）。Phase 7/8 照此执行。

---

## 003-D5：复数语义

**扩展**：002-D3（复数 reduction 推迟项，本决策兑现）、L4-1

**决策**：

**v0.3 复数落地范围**（complex64 / complex128）：

| 层 | 内容 |
|---|---|
| kernel | elementwise：binary add/sub/mul/div + unary neg/abs；reduction：sum/mean（实部虚部分别归约） |
| 数学库 | linalg complex（Cgemm/Zgemm、complex getrf/getrs/geqrf+cungqr/gesvd）、fft 全套 |
| Python 侧 | `ms.array` complex 创建（含 Python complex 字面量推断）、`tolist()` / `item()` complex 分支 |

**语义规则**：
- `max/min/argmax/argmin` 对复数**永久拒绝** → `DtypeError`（002-D3 的 v0.2 临时规则转正：复数无全序）。
- comparison：`eq/ne` 支持复数；`lt/gt/le/ge` 拒绝复数 → `DtypeError`（NumPy 行为）。
- 类型提升：沿用 `dtype.rs` 已实现的 CAT_COMPLEX 规则（JAX 风格）：float+complex →
  分量位宽取大的 complex（f32+c64→c64，f64+c64→c128）；int+complex → 精确提升。
- Python 复数字面量创建：`ms.array([1+2j])` 默认推断 **complex128**（NumPy 行为），
  `dtype=` 显式覆盖。
- `fft` 输出恒为 complex；`ifft` 归一化 1/N（NumPy 约定）。

**依据**：
- 002-D3 明确「sum/mean 复数支持推迟」，v0.3 兑现；fft（L4-1）以 complex 为硬前置。
- v0.2 实测：`ms.array` complex 创建与 tolist/item complex 分支均抛 NotImplementedError
  （ops.rs / array.rs），必须与 kernel 一并落地，否则 §1.3 成功定义不可达。

**影响**：
- elementwise.mu / reduction.mu 新增 complex 实例化（mcc 3.1.0 兼容性先做最小冒烟，见风险表）。
- musapy-python ops.rs / array.rs 解除 complex NotImplementedError 分支。
- creation.rs 的 fill/arange 等 init kernel 复数实例化**不在 v0.3 范围**（zeros complex
  可用 fill=0 的 cast 路径临时覆盖，正式版推迟）。

---

## 003-D6：dot 算子补充

**扩展**：L4-1（linalg 行原列表 matmul/lu/qr/svd/solve **不含 dot**，本决策补充）

**决策**：v0.3 新增 `ms.dot(a, b)`：

| 输入形状 | 语义 | 实现 |
|---|---|---|
| (n,) · (n,) | 内积 → **0-dim** | `mublasSdot/Ddot/Cdotu/Zdotu`（复数**不取共轭**，NumPy 语义 → dotu） |
| (m,n) · (n,k) | = matmul | 委托 matmul 路径（003-D7） |
| N-D 广播 dot | **不支持**（v0.4+ 评估） | NumPy N-D dot 为 sum-product over axes |

**依据**：dot 是高频算子；在 matmul/mublas 之上实现成本极低。L4-1 未列出属遗漏，
按主 ADR 使用说明第 2 条在本文档补充记录。

**影响**：linalg.rs + 导出 `ms.dot` + `Array.dot()`（不新增 dunder，`@` 是 matmul）。

---

## 003-D7：数学库 Python API 形态与返回约定

**扩展**：L4-1、L3-18（`__array_function__` v1 范围的算子命名惯例）

**决策**：

**API 形态**：

| 域 | 形态 | 示例 |
|---|---|---|
| linalg | **模块级函数** + `__matmul__` dunder | `ms.matmul(a, b)`、`a @ b`、`ms.solve(a, b)` |
| random / fft / sparse | **命名空间子模块** | `ms.random.rand(...)`、`ms.fft.fft(...)`、`ms.sparse.csr_matrix(...)` |

命名空间对齐 NumPy 的 `np.random` / `np.fft` / `scipy.sparse` 肌肉记忆（L0-1）。

**返回约定**（除注明外对齐 NumPy）：

| 算子 | 约定 |
|---|---|
| `ms.svd(a, full_matrices, compute_uv)` | `(u, s, vh)`，s 1D **降序**（NumPy 语义） |
| `ms.qr(a, mode)` | `(q, r)`，`mode='reduced'/'complete'`（NumPy 语义） |
| `ms.lu(a)` | `(lu, piv)`，对齐 **`torch.linalg.lu`** / LAPACK getrf 布局（piv 1-based int64）。注意 **NumPy 无 `np.lu`**，不以其为参照 |
| `ms.solve(a, b)` | 解 `a·x = b`；奇异 → `LinAlgError`（003-D3）；非方阵 → `ShapeError` |
| `ms.fft.ifft` | 归一化 1/N；支持 `norm='ortho'`（NumPy 约定） |
| `ms.fft.rfft` | 实输入 → complex 输出，形状 `N//2+1` |
| `ms.random.*` | `seed=` 每次调用可选；同 seed 逐元素可复现 |
| `ms.sparse.csr_matrix/coo_matrix` | 独立轻量类型（**非** Array 子类）；支持 `@`（spmv/spmm）与 `.toarray()` |

**依据**：
- lu 无 NumPy 对应物；`torch.linalg.lu` 的 `(LU, pivots)` 布局即 LAPACK getrf 原生输出，
  可零拷贝透传 musolver getrf 结果，实现代价最低。
- svd/qr/fft 的 NumPy 约定是用户预期基线（L0-1）。

**影响**：
- musapy-python 新增 random.rs / fft.rs / sparse.rs 子模块；python/musapy 新增
  random.py / fft.py / sparse.py 包装。
- `_core.pyi` 与 `__init__.py` 同步导出；测试按 §14.1 验收矩阵执行。

---

## 003-D8：分解类算子实施注记（2026-08-07 真机探针修正）

**扩展**：003-D7（lu/qr/svd 返回约定）实施过程中的 SDK 3.1.0 实测修正。

**背景**：Phase 3 实施前用 C 探针核对 muSOLVER 语义，两处结论与初始假设
（头文件浅读）相反，实施时已按实测修正：

| # | 探针发现 | 影响 |
|---|---|---|
| 1 | `musolver?gesvd` 的 **V 输出缓冲即 Vᵀ**（头文件 "stored as rows (transposed)"；探针 2：按 V' 列主序重建误差 1e-15，按 V 解释误差 4.1） | `vh` = strides `(1, ldv)` 视图，**非** `(n,1)` |
| 2 | **SINGULAR 模式 U 输出有 bug**：m>n 时 U 损坏（探针 3：6×4 下 UᵀU−I=4.5，OUTOFPLACE/INPLACE 均复现；同矩阵 ALL 模式误差 1e-15） | 一律 ALL/ALL + 薄视图切片（thin = 全尺寸缓冲的前 k 列/行跨步视图），绕开 SINGULAR |
| 3 | SINGULAR 模式 V 缓冲按 `min(m,n)` 紧凑写入（传更大 ldv 会写出错乱，探针 2 wide 3×5 实锤） | 与 #2 合并解决（不再使用 SINGULAR） |

**决策**：
- `ms.svd` 内部恒走 `MUBLAS_SVECT_ALL`（`compute_uv=False` 走 NONE/NONE）；
  thin 输出（u m×k / vh k×n）为全尺寸缓冲的跨步视图切片——**零额外拷贝**。
- 奇异值合理性校验（S 全部 ≥0 且有限）作为 gesvd info 可靠性兜底
  （getrf 的 info 失效已在 Phase 2 实锤，003-D3；gesvd 探针显示 info=0 正常写）。
- lu/qr 无新增模式问题：getrf/geqrf/orgqr 按头文件签名调用，列主序物化输入
  + 跨步视图输出（见 003-D3 实施注记）。

**影响**：linalg.rs `svd` 的实现约束；文档注明 SDK 3.1.0 SINGULAR 模式不可用
（升级 SDK 后可评估改回 SINGULAR 以省显存）。全部 SDK 3.1.0 限制/缺陷汇总见
[sdk-3.1.0-limitations.md](./sdk-3.1.0-limitations.md)。

---

## 003-D9：random 套件语义（seed / 生成串行化 / shape=None）

**扩展**：003-D7（`ms.random` 命名空间）、003-D4（GPU-only）

**决策**：

1. **seed 语义**：`seed=` 给定 → 每次调用前 `murandSetPseudoRandomGeneratorSeed`
   重置生成器，**同 seed 紧邻两次调用逐元素可复现**（验收标准）；无 seed →
   不重置（共享 generator 自然推进，连续调用产生不同序列）。`offset` 仅声明
   FFI（P4.1 交付物），v0.3 不暴露公开参数。
2. **生成串行化（真机探针 2026-08-07 实锤）**：generator 为按 device 缓存的
   共享资源。若每个 op 新建 MUSA 流（creation 骨架惯例），生成 kernel 跨流
   并发会与 seed 重置交错，**破坏 f64 Normal 的 seed 复现性**（f32/f64
   Uniform 因生成快未触发）；同流异步序列完全可复现（无需中间同步）。
   → random 算子无显式 stream context 时走 **per-device 缓存单一流**
   （random.rs `generation_stream`）；有用户 stream context 时仍遵循
   「每 op 显式传 stream」惯例，多流并发调用 random 的复现性由调用方保证。
3. **shape=None → 0-dim 标量数组**（NumPy 对齐；用户确认，plan-phase4）。
4. **实现路径**：`normal(loc, scale)` 用 `murandGenerateNormal` **原生
   mean/stddev 参数一步生成**（头文件核对发现，非仿射组合）；`uniform(low,
   high)` = rand·(high−low)+low 复用既有 elementwise（mul/add + 0-dim full
   标量广播，**零新增 kernel**）；`bernoulli(p)` = rand < p → **bool**
   （复用 comparison::lt）。GPU-only（003-D4，CPU 调用抛 DeviceError）。

**影响**：random.rs 的流管理；mock stub 状态化（seed 重置 + 计数器推进 +
Σ12u−6 近似正态），使无 GPU CI 可跑形状/复现性/分布统计用例；bench 记录
SDK 特征：randn f64 吞吐 ~3 GB/s（比 f32 慢 ~50×，见 sdk-3.1.0-limitations）。

---

## 变更记录

| 日期 | 变更 | 影响的决策 ID |
|---|---|---|
| 2026-08-06 | 初始草案，7 个 v0.3 补充决策（基于 SDK 3.1.0 头文件实测核对） | 003-D1 至 003-D7 |
| 2026-08-07 | Phase 3 实施：lu/qr/svd 落地，探针修正 gesvd V 输出 = Vᵀ、SINGULAR 模式 U 损坏 → ALL 模式切片（003-D8） | 003-D7、003-D8 |
| 2026-08-07 | Phase 4 实施：random 套件，seed/生成串行化/shape=None 语义 + Normal 原生 mean/stddev（003-D9） | 003-D7、003-D9 |
| 2026-08-08 | Phase 5 实施：complex 落地（elementwise add/sub/mul/div/neg/abs + comparison eq/ne、array 创建/tolist/item、real→complex cast 4 对 + c64→c128 提升）+ fft 套件（fft/ifft/rfft，axis=-1 起步，n 截断/补零，norm 三值，out=）。fftn/多轴与 irfft 推迟到 v0.3 后期。mcc 3.1.0 complex struct kernel 兼容性冒烟通过（sdk-3.1.0-limitations 风险解除）。mock mufft 用 naive O(N²) DFT 数值仿真 | 003-D4、003-D5、003-D7 |

---

## 与主 ADR / ADR-002 的交叉引用

- **FFI 层放置**：003-D1 扩展并澄清 L2-3、L2-1；构建探测沿用 L2-1 双探针
- **句柄生命周期**：003-D2 扩展 L2-3（handle 表）、L3-9（deferred-free）、L3-10（dealloc stream）
- **错误模型**：003-D3 扩展 L3-5/L3-6；延续 v0.2 的单继承实现
- **CPU fallback**：003-D4 澄清与 L4-2（musapy-cpu crate 推迟）的边界
- **复数**：003-D5 兑现 002-D3 推迟项；提升规则沿用 L1-5/L1-14（dtype.rs CAT_COMPLEX）
- **dot**：003-D6 补充 L4-1 linalg 行
- **API 形态**：003-D7 扩展 L4-1、L3-18
- **capture-safety**：全部新算子遵守 L1-12、L2-4（参数解析一次性、调用可重放），
  为 v2 的 Graphs capture 保留钩子（L4-2）

---

*ADR-003 结束*
