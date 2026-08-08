# SDK 3.1.0 MUSA-X 已知限制与实测行为汇总

> **日期**:2026-08-08（随 v0.3 P0-P2 性能优化更新；§1.8-1.11 为 2026-08-07/08 benchmark 段级探针新增，§1.12-1.16 为 mcc 编译器限制）
> **SDK**:MUSA 3.1.0（`/usr/local/musa` → `/usr/local/musa-3.1.0`,mp_21/mp_22,与 MTT S4000 匹配）
> **目的**:把散落在 ADR-003（003-D2/D3/D8）、v0.3 计划 §2/§16 风险表、
> sdk-3.1.0-musax-coverage.md、linalg.rs 注释中的 SDK 限制/怪异行为集中到本文件,
> 作为实现与部署的单一查询入口。
> **相关文档**:[sdk-3.1.0-musax-coverage.md](../sdk-3.1.0-musax-coverage.md))（符号覆盖清单）、
> [ADR-003-zh.md](../adr/ADR-003-zh.md))（003-D2 句柄模型 / 003-D3 错误模型 / 003-D8 探针修正）、
> [v0.3-alpha-plan-zh.md](./v0.3-alpha-plan-zh.md)) §16 风险登记表、
> [benchmark/analysis-2026-08-07.md](../../benchmark/analysis-2026-08-07.md))（性能归因与实测数据）。

---

## 1. 行为缺陷（实测,影响实现语义）

| # | 缺陷 | 实测证据 | 影响 | musapy 处置 | 升级 SDK 后 |
|---|---|---|---|---|---|
| 1.1 | **`musolver?getrf` 从不写 info 输出** | 2026-08-07 真机 C 探针:非奇异与奇异矩阵 info 均保持预填值,status=0、ipiv 正常 | 无法用 LAPACK 标准 `info>0` 判奇异 | solve 走 **LU 对角 D2H 检查**(P0,2026-08-08 起为 `musapy_extract_diag` kernel 提取对角 + 单次连续 D2H,U(k,k)==0 → `LinAlgError::Singular`,见 linalg.rs gpu_solve) | 可改回 info 判断(对角检查保留亦无害) |
| 1.2 | **`musolver?gesvd` 的 V 输出缓冲即 Vᵀ**(非 V) | 头文件 "stored as rows (transposed)";探针 2:按 V' 列主序重建误差 1e-15,按 V 解释 4.1 | vh 视图 strides 必须按 Vᵀ 设计 | `vh` = strides `(1, ldv)` 视图(003-D8;初始假设「V 输出就是 V」为误读,已修正) | 语义不变,沿用即可 |
| 1.3 | **`gesvd` SINGULAR 模式 U 输出损坏**(m>n 时) | 探针 3:6×4 下 UᵀU−I=4.5,status=0 无报错;OUTOFPLACE/INPLACE 均复现;同矩阵 ALL 模式误差 1e-15 | SINGULAR 模式不可用(尤其 tall 矩阵) | svd 一律 **ALL/ALL + 薄视图切片**(thin = 全尺寸缓冲前 k 列/行跨步视图,零额外拷贝,003-D8) | 验证修复后可评估改回 SINGULAR(省显存) |
| 1.4 | **`gesvd` SINGULAR 模式 V 按 `min(m,n)` 紧凑写入** | 探针 2:wide 3×5 传 ldv=n 输出错乱;ldv=k 时重建 1e-15 | 传大 ldv 会写出错乱 | 随 1.3 一并规避(不再使用 SINGULAR) | 同 1.3 |
| 1.5 | **`gesvd` info 语义弱**:正常返回 info=0(与 getrf 不同),但收敛失败时 info 可靠性存疑 | 探针:gesvd info 正常写 0 | 收敛失败不可直接检测 | **S 合理性校验兜底**(S 全部 ≥0 且有限,否则抛 `DeviceError`);局限:S 层面可观测 | 可加 info>0 → 收敛错误映射 |
| 1.6 | **randn f64 生成吞吐 ~3 GB/s**(比 f32 慢约 50×) | bench_random.py 实测:10M 元素 randn f64 29.7ms vs f32 0.19ms;100M 260ms vs 2.7ms | f64 Normal 生成是性能瓶颈(Box-Muller 类实现) | 文档注明;f64 大批量 randn 需评估(如混用 f32 + 精度折衷);Uniform f64 无此问题(142 GB/s) | 升级 SDK 后重测 |
| 1.7 | **共享 generator 跨流并发破坏 seed 复现性**(f64 Normal 可复现性) | 真机探针 2026-08-07:每 op 新建流时 f64 Normal 同 seed 两次不等;同流异步序列完全可复现 | 与 musapy 的「每 op 新建流」惯例冲突 | random 算子走 per-device 缓存单一流(ADR-003 003-D9);多流并发调 random 由调用方保证 | 语义不随 SDK 变化 |
| 1.8 | **`musaMemcpy2D` 小 pitch D2H 非确定性**(同参数不同上下文随机出现) | 2026-08-07 C 探针:dpitch=8/width=8/D2H 同一调用实测随机出现 error 1 / 进程段错误 / 2.2s stall / 26ms / 正常;连续 8KB `musaMemcpy` 稳定 0.18ms(145× 差距) | 跨步小行 D2H 不可依赖;8KB 对角读回曾 26.5ms(逐行 ~26µs 驱动开销) | solve 奇异检测 P0(2026-08-08)改为设备端 `extract_diag` kernel + 连续 D2H,已绕开;新增跨步 D2H 场景须用 kernel 提取 | 验证修复后可评估直接跨步读回 |
| 1.9 | **`mublas?gemm` f64 封顶 ~160 GFLOPS**(f32 的 1/86) | 2026-08-08 探针:dgemm 512/1024/2048 = 97/156/159 GFLOPS;f32/f64 延迟比随规模恶化 14×→48×→88×(mp_22 无原生 FP64,软件仿真 + DGEMM kernel 调优不足) | 所有 f64 分解类算子(lu/qr/svd/solve 的 getrf/getrs)性能预期上限 ≈160 GFLOPS | 文档注明性能预期;f64 大批量计算需评估精度折衷走 f32 | 升级 SDK 后重测 |
| 1.10 | **`gesvd` NONE/NONE 仅省 ~25% 耗时**(compute_uv=False 收益有限) | 2026-08-08 探针:1024² f64 ALL 2695ms vs NONE 2016ms(0.75×);256² 同样 0.76× | svd(compute_uv=False)逃生通道收益有限——NONE 模式仍做大部分迭代工作 | svd 已支持 compute_uv=False;文档注明预期收益;eigh 等其他分解不依赖此路径 | 升级 SDK 后重测 |
| 1.11 | **`geqrf`/`orgqr` 实现低效**(f64 亦非仿真问题) | 2026-08-08 探针:geqrf 1024² f64 259ms vs getrf 17ms(同量级 O(n³) 工作量慢 15×);f32 qr(1024) 365ms 仅比 f64 快 1.2× | qr 算子延迟天花板 ≈ geqrf+orgqr 组合(~438ms@1024²) | qr 走 geqrf+orgqr(无替代);文档注明;大批量 qr 需评估 | 升级 SDK 后重测 |
| 1.12 | **mcc 不支持 `__shfl_down_sync` 的 struct 参数**(c64/c128 编译错误) | 探针证实(2026-08-08 复数归约):shfl 传 `muComplex`/`muDoubleComplex` 编译失败 | 复数归约无法直接 shuffle 复合值 | re/im 拆成两个独立标量分别 shuffle 归约(reduction.mu `cplx_*_v2`,~1900× 优化,repo.md §6.1) | 升级 mcc 后可评估复合值 shuffle |
| 1.13 | **mcc 不支持指针数组 kernel 参数**(`const int64_t* const*` 启动 error 999) | 2026-08-08 探针(高级索引):`adv_gather`/`nonzero` kernel 传索引指针数组启动报错 999 | GPU 端 fancy/boolean 索引无法直接收指针数组 | 高级索引走 host fallback(D2H→host→H2D),kernel 已声明留作后续接入(indexing.rs,repo.md §7) | 升级 mcc 后接入 GPU kernel |
| 1.14 | **musparse alpha/beta 标量指针宽度须按 dtype 传**(f32 路径传 f64 指针 → 输出全零) | 2026-08-08 真机:spmv/spmm 的 f32 分支若沿用 f64 alpha/beta 指针,输出恒 0 | alpha/beta 指针类型必须与 data dtype 匹配 | 按 dtype 传对应宽度标量指针(f32→f32*,f64→f64*,sparse.rs 注释) | 语义不随 SDK 变化 |
| 1.15 | **mcc 对 float4+shuffle 组合生成病态代码**(f32 reduction 显式 float4 变慢 ~47×) | 2026-08-08 探针(analysis-cplx-bw):f32 reduction 显式 float4+shuffle 组合 47× 变慢 | f32 归约无法用显式 float4 提速(LD.B128 与 shuffle 路径互斥) | f32 reduction 保持标量路径(~220 GB/s);若编译器升级后可翻倍(analysis-cplx-bw 注释) | 升级 mcc 后重测 |
| 1.16 | **mcc 对 REDUCE_ITEMS=8 的 unroll+边界检查生成病态代码**(~3500× 退化) | 2026-08-07 探针:REDUCE_ITEMS 4→8,64M sum 1.176→4106ms | 大 unroll 宽度 + 边界检查组合生成病态代码(与 1.15 同族) | REDUCE_ITEMS 回退 4;partial 带宽 ~220 GB/s 瓶颈在 shuffle+smem 路径本身 | 升级 mcc 后重测 |

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
| 3.5 | mcc 3.1.0 编译 complex kernel | Phase 5 已落地:complex elementwise/reduction/cast 全套编译并通过真机验证(见 1.12/1.15/1.16 的 mcc 限制规避) | 已解决;复数归约 re/im 分量并行,complex 仅支持已实现算子族 |

---

## 4. 算子影响速查表

| musapy 算子 | 相关限制 | 当前处置（代码位置） |
|---|---|---|
| `solve` | 1.1/1.8(getrf 不写 info;memcpy2D 不可靠) | extract_diag kernel + 连续 D2H 奇异检测(linalg.rs `gpu_solve`,P0) |
| `lu` | 1.1 | 不做奇异检测(与 torch 一致:返回 LU/piv,奇异由用户判);共享 `gpu_getrf` |
| `qr` | 1.11(geqrf/orgqr 低效)、2.2(复数 orgqr) | 实数 S/D 直接可用;性能预期见 1.11;复数等 Phase 5 走 cungqr |
| `svd` | 1.2/1.3/1.4/1.5/1.10 | ALL 模式 + 薄视图切片 + S 合理性校验(linalg.rs `svd`);compute_uv=False 仅省 ~25% |
| `eigh`(v0.4) | 2.3(syevd 仅 S/D) | 推迟;复数另行评估 |
| random 套件(Phase 4) | 2.8 | 无实质影响 |
| 复数 reduction(Phase 7) | 1.12(shfl struct)/1.15(float4+shuffle) | re/im 分量并行归约(reduction.mu `cplx_*_v2`) |
| 高级索引(Phase 8) | 1.13(指针数组 error 999) | host fallback;GPU kernel 已声明待接入 |
| fft 套件(Phase 5) | 3.1(libmtfft-device) | 部署环境预检 |
| sparse 套件(Phase 6) | 2.5、1.14(alpha/beta 宽度) | 两段式查询;按 dtype 传标量指针 |
| 内存管理(全局) | 2.4(无 async alloc) | deferred-free 默认路径(ADR-zh L3-9) |
| 句柄管理(全局) | 2.7(共享句柄) | `math_handle::with_mublas_handle`(003-D2) |

---

## 5. 复测与升级指引

- **符号复测**:`./tools/check_musax_symbols.sh [SDK路径]`(当前 96/96 全 PASS;.so 与头文件一致)。
- **行为复测**:上述 1.x 结论来自实施期真机 C 探针(临时文件,存 `/tmp`,不随仓库保留);
  复现思路见 ADR-003 003-D8 的表项(现象 + 判据)。升级 SDK 后建议重跑:
  - getrf info 是否开始写入(1.1 → 可改回 info 判奇异);
  - SINGULAR 模式 U 是否修复(1.3 → 可评估改回 SINGULAR 省显存);
  - `musaMallocAsync`/`musaFreeAsync` 是否导出(2.4 → 可切 stream-ordered 默认路径);
  - `musaMemcpy2D` 小 pitch D2H 是否稳定(1.8 → 可评估去掉 extract_diag kernel 中转);
  - dgemm f64 / gesvd NONE / geqrf+orgqr 性能是否改善(1.9/1.10/1.11 → 更新性能预期文档)。
- **升级目标**:MUSA SDK ≥5.1.0(stream-ordered 完整)或至少修复 SINGULAR 的 4.x 版本。

---

*本文件与 ADR-003-zh.md 003-D8 同步维护;英文版要点见 ADR-003.md 003-D8。*
