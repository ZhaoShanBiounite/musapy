# SDK 3.1.0 MUSA-X 已知限制与实测行为汇总

> **日期**:2026-08-07（随 v0.3 Phase 3 实施更新）
> **SDK**:MUSA 3.1.0（`/usr/local/musa` → `/usr/local/musa-3.1.0`,mp_21/mp_22,与 MTT S4000 匹配）
> **目的**:把散落在 ADR-003（003-D2/D3/D8）、v0.3 计划 §2/§16 风险表、
> sdk-3.1.0-musax-coverage.md、linalg.rs 注释中的 SDK 限制/怪异行为集中到本文件,
> 作为实现与部署的单一查询入口。
> **相关文档**:[sdk-3.1.0-musax-coverage.md](./sdk-3.1.0-musax-coverage.md)（符号覆盖清单）、
> [ADR-003-zh.md](./ADR-003-zh.md)（003-D2 句柄模型 / 003-D3 错误模型 / 003-D8 探针修正）、
> [v0.3-alpha-plan-zh.md](./v0.3-alpha-plan-zh.md) §16 风险登记表。

---

## 1. 行为缺陷（实测,影响实现语义）

| # | 缺陷 | 实测证据 | 影响 | musapy 处置 | 升级 SDK 后 |
|---|---|---|---|---|---|
| 1.1 | **`musolver?getrf` 从不写 info 输出** | 2026-08-07 真机 C 探针:非奇异与奇异矩阵 info 均保持预填值,status=0、ipiv 正常 | 无法用 LAPACK 标准 `info>0` 判奇异 | solve 走 **LU 对角 D2H 检查**(musaMemcpy2D 单次读回,U(k,k)==0 → `LinAlgError::Singular`,见 linalg.rs gpu_solve) | 可改回 info 判断(对角检查保留亦无害) |
| 1.2 | **`musolver?gesvd` 的 V 输出缓冲即 Vᵀ**(非 V) | 头文件 "stored as rows (transposed)";探针 2:按 V' 列主序重建误差 1e-15,按 V 解释 4.1 | vh 视图 strides 必须按 Vᵀ 设计 | `vh` = strides `(1, ldv)` 视图(003-D8;初始假设「V 输出就是 V」为误读,已修正) | 语义不变,沿用即可 |
| 1.3 | **`gesvd` SINGULAR 模式 U 输出损坏**(m>n 时) | 探针 3:6×4 下 UᵀU−I=4.5,status=0 无报错;OUTOFPLACE/INPLACE 均复现;同矩阵 ALL 模式误差 1e-15 | SINGULAR 模式不可用(尤其 tall 矩阵) | svd 一律 **ALL/ALL + 薄视图切片**(thin = 全尺寸缓冲前 k 列/行跨步视图,零额外拷贝,003-D8) | 验证修复后可评估改回 SINGULAR(省显存) |
| 1.4 | **`gesvd` SINGULAR 模式 V 按 `min(m,n)` 紧凑写入** | 探针 2:wide 3×5 传 ldv=n 输出错乱;ldv=k 时重建 1e-15 | 传大 ldv 会写出错乱 | 随 1.3 一并规避(不再使用 SINGULAR) | 同 1.3 |
| 1.5 | **`gesvd` info 语义弱**:正常返回 info=0(与 getrf 不同),但收敛失败时 info 可靠性存疑 | 探针:gesvd info 正常写 0 | 收敛失败不可直接检测 | **S 合理性校验兜底**(S 全部 ≥0 且有限,否则抛 `DeviceError`);局限:S 层面可观测 | 可加 info>0 → 收敛错误映射 |

---

## 2. 符号与例程缺口（coverage 核对结论）

| # | 缺口 | 影响范围 | 处置 |
|---|---|---|---|
| 2.1 | **`gesdd` 不存在**(S/D/C/Z 均无) | svd 无快速分解替代路径 | 用 gesvd(功能正确,慢于 gesdd 若存在) |
| 2.2 | **`orgqr` 仅 S/D**(实数);复数走 `cungqr` C/Z | qr 的复数支路 | Phase 5 复数落地时用 cungqr |
| 2.3 | **`syevd` 仅 S/D**(无 C/Z) | eigh 复数特征值无例程 | eigh 推迟 v0.4,复数另行评估 |
| 2.4 | **`musaMallocAsync`/`musaFreeAsync` 头文件有声明、.so 无符号**(3.1.0/3.3.5/4.3.7 实测;5.1.0 才完整) | stream-ordered 内存分配不可用 | deferred-free 队列默认路径(ADR-zh L3-9,feature gate + runtime probe 双保险) |
| 2.5 | **musparse 无独立 `_bufferSize` 符号** | SpMV/SpMM 两段式查询 | `buffer=nullptr` → size 查询 → 分配 → 计算(动作参数驱动) |
| 2.6 | **五个库均无静态库** | 无静态链接选项 | 一律动态链接 |
| 2.7 | **musolver 无独立句柄**(不存在 `musolverDnHandle_t`/`musolverDnCreate`) | 句柄模型与 cuSOLVER 不同 | muBLAS/muSOLVER 共享 `mublasHandle_t`(003-D2) |
| 2.8 | **murand/mufft `GetVersion` 无句柄参数** | API 形态差异(非缺陷) | 直接传 `int*` |

---

## 3. 运行期依赖与环境

| # | 依赖 | 说明 | 处置/记录 |
|---|---|---|---|
| 3.1 | **muFFT → `libmtfft-device-*.so`** | 按 arch 变体,lib 目录在位 | P1 冒烟验证加载路径;风险表中评级(中,中) |
| 3.2 | **muSOLVER → libomp**(OpenMP 运行库) | 需 **versionless `libomp.so`**;2026-08-07 部署时创建 `/usr/lib/x86_64-linux-gnu/libomp.so → libomp.so.5` | 部署环境需预检;缺失表现为运行期加载失败 |
| 3.3 | 宿主 OpenBLAS(v0.2 CPU fallback) | 缺失则降级纯 Rust 朴素实现 | v0.3+ 数学库算子 GPU-only 后仅影响 v0.2 算子 |
| 3.4 | **SDK 3.1.0 过老** | mutlass ≥4.3.4 / tilelang ≥5.2 等轮子不可用 | 风险表(已知,低);本计划不引入这些依赖;升级 SDK 列为 v0.3 后期可选 |
| 3.5 | mcc 3.1.0 编译 complex kernel | 兼容性未验证 | 风险表(中,中);Phase 5 先做最小 complex 冒烟再扩套件 |

---

## 4. 算子影响速查表

| musapy 算子 | 相关限制 | 当前处置（代码位置） |
|---|---|---|
| `solve` | 1.1(getrf 不写 info) | LU 对角 D2H 奇异检测(linalg.rs `gpu_solve`) |
| `lu` | 1.1 | 不做奇异检测(与 torch 一致:返回 LU/piv,奇异由用户判);共享 `gpu_getrf` |
| `qr` | 2.2(复数 orgqr) | 实数 S/D 直接可用;复数等 Phase 5 走 cungqr |
| `svd` | 1.2/1.3/1.4/1.5 | ALL 模式 + 薄视图切片 + S 合理性校验(linalg.rs `svd`) |
| `eigh`(v0.4) | 2.3(syevd 仅 S/D) | 推迟;复数另行评估 |
| random 套件(Phase 4) | 2.8 | 无实质影响 |
| fft 套件(Phase 5) | 3.1(libmtfft-device) | 部署环境预检 |
| sparse 套件(Phase 6) | 2.5 | 两段式查询 |
| 内存管理(全局) | 2.4(无 async alloc) | deferred-free 默认路径(ADR-zh L3-9) |
| 句柄管理(全局) | 2.7(共享句柄) | `math_handle::with_mublas_handle`(003-D2) |

---

## 5. 复测与升级指引

- **符号复测**:`./tools/check_musax_symbols.sh [SDK路径]`(当前 96/96 全 PASS;.so 与头文件一致)。
- **行为复测**:上述 1.x 结论来自实施期真机 C 探针(临时文件,存 `/tmp`,不随仓库保留);
  复现思路见 ADR-003 003-D8 的表项(现象 + 判据)。升级 SDK 后建议重跑:
  - getrf info 是否开始写入(1.1 → 可改回 info 判奇异);
  - SINGULAR 模式 U 是否修复(1.3 → 可评估改回 SINGULAR 省显存);
  - `musaMallocAsync`/`musaFreeAsync` 是否导出(2.4 → 可切 stream-ordered 默认路径)。
- **升级目标**:MUSA SDK ≥5.1.0(stream-ordered 完整)或至少修复 SINGULAR 的 4.x 版本。

---

*本文件与 ADR-003-zh.md 003-D8 同步维护;英文版要点见 ADR-003.md 003-D8。*
