# SDK 3.1.0 MUSA-X 例程覆盖清单(v0.3 Phase 1 P1.1 交付物)

> **日期**:2026-08-06
> **SDK**:MUSA 3.1.0(`/usr/local/musa` → `/usr/local/musa-3.1.0`,mp_21/mp_22,与 MTT S4000 匹配)
> **依据**:[v0.3-alpha-plan-zh.md](./v0.3-alpha-plan-zh.md) §2 前置条件 + Phase 1 P1.1;
> [ADR-003-zh.md](./ADR-003-zh.md) 003-D2(句柄模型)

本清单是 v0.3 全部数学库阶段的符号级前置验证,分两级完成:

1. **头文件级核对(2026-08-06,提前完成)**:逐个核对 `/usr/local/musa/include/` 下
   `mublas.h`(含 `internal/`)、`musolver_functions.h`、`murand.h`、`mufft.h`、
   `musparse-functions.h`/`musparse-auxiliary.h` 的签名与常量。
2. **运行期符号验证(本文档)**:`tools/check_musax_symbols.sh` 用 `nm -D --defined-only`
   核对 5 个 `.so` 实际导出的符号。**结果:83/83 全部 PASS**,无头文件与 .so 不一致项。

## 库文件形态

| 库 | 文件 | 形态 |
|---|---|---|
| muBLAS | `libmublas.so` | 单一 .so(381 MB),SONAME 即本名 |
| muSOLVER | `libmusolver.so` → `libmusolver.so.1.0.0` | 版本后缀软链(630 MB) |
| muRAND | `libmurand.so` | 单一 .so(59 MB) |
| muFFT | `libmufft.so` | 单一 .so(580 KB);运行期依赖 `libmtfft-device-*.so`(按 arch 变体,lib 目录在位) |
| muSPARSE | `libmusparse.so` | 单一 .so(248 MB) |

五个库**均无静态库**(`_static.a` 不存在),musapy 一律动态链接。

## 逐库覆盖

### muBLAS(Phase 2:matmul/dot)

| 例程 | S/D/C/Z | 备注 |
|---|---|---|
| `mublasCreate/Destroy/SetStream/GetVersion` | — | Phase 1 生命周期,`GetVersion(handle, int*)` |
| `mublasSgemm/Dgemm/Cgemm/Zgemm` | ✅ | matmul 主路径 |
| `mublasSdot/Ddot/Cdotu/Zdotu` | ✅ | dot 用 **dotu 变体**(NumPy dot 对复数不取共轭,003-D6) |
| `mublasSgemmStridedBatched/DgemmStridedBatched` | ✅ | batch matmul 候选(v0.3 范围内视需要启用) |

### muSOLVER(Phase 2 solve / Phase 3 lu/qr/svd)

**关键差异(SDK 3.1.0 实测)**:无独立句柄 —— 不存在 `musolverDnHandle_t`/`musolverDnCreate`
(与 cuSOLVER 不同)。全部例程形如 `musolverSgetrf(mublasHandle_t handle, ...)`,
与 muBLAS **共享句柄**、共用 `mublasStatus_t`(ADR-003 003-D2)。

| 例程 | S | D | C | Z | 备注 |
|---|---|---|---|---|---|
| `getrf` + `getrf_bufferSize` | ✅ | ✅ | ✅ | ✅ | lu / solve 第一步 |
| `getrs` | ✅ | ✅ | ✅ | ✅ | solve(getrf 后) |
| `geqrf` + `geqrf_bufferSize` | ✅ | ✅ | ✅ | ✅ | qr 第一步 |
| `orgqr`(S/D)/ `cungqr`(C/Z) | ✅ | ✅ | ✅ | ✅ | qr 第二步,**复数用 cungqr** |
| `gesvd` + `gesvd_bufferSize`(S/D) | ✅ | ✅ | ✅ | ✅ | svd;**`gesdd` 不存在**,不可作替代 |
| `syevd` + `syevd_bufferSize` | ✅ | ✅ | — | — | eigh 候选(v0.4,范围管理推迟) |
| `potrf` / `gebrd` | ✅ | ✅ | ✅ | ✅ | 已存在,v0.3 不使用 |

### muRAND(Phase 4:random 套件)

| 例程 | 状态 | 备注 |
|---|---|---|
| `murandCreateGenerator(murandGenerator_t*, murandRngType_t)` | ✅ | 引擎枚举:`MURAND_RNG_PSEUDO_DEFAULT=400`、`MURAND_RNG_PSEUDO_PHILOX4_32_10` |
| `murandDestroyGenerator/SetStream/SetPseudoRandomGeneratorSeed/SetGeneratorOffset` | ✅ | |
| `murandGetVersion(int*)` | ✅ | **无句柄参数** |
| `murandGenerateUniform/UniformDouble/Normal/NormalDouble` | ✅ | f32/f64 分独立函数(f64 为 `*Double` 变体) |

### muFFT(Phase 5:fft 套件)

cufft 同构 API:

| 例程 | 状态 | 备注 |
|---|---|---|
| `mufftCreate(plan*)` / `mufftDestroy(plan)` | ✅ | |
| `mufftPlan1d(plan*, nx, mufftType, batch)` | ✅ | 含 batch 参数 |
| `mufftPlan2d/Plan3d/PlanMany` | ✅ | PlanMany 为通用 N 维 |
| `mufftSetStream(plan, musaStream_t)` | ✅ | 直接收 `musaStream_t` |
| `mufftGetVersion(int*)` | ✅ | 无句柄参数 |
| `mufftExecC2C/R2C/C2R/Z2Z/D2Z/Z2D` | ✅ | |

常量:`MUFFT_FORWARD = -1`、`MUFFT_INVERSE = 1`;`mufftResult` 来自 `mufftResult_t` 枚举。

### muSPARSE(Phase 6:sparse 套件)

泛型 API(cusparse generic 同构):

| 例程 | 状态 | 备注 |
|---|---|---|
| `musparseCreate/Destroy/SetStream(handle, MUstream)/GetVersion(handle, int*)` | ✅ | `MUstream` 与 `musaStream_t` 同为 `struct MUstream_st*`,无需转换 |
| `musparseCreateCsr/CreateCoo/CreateDnVec/CreateDnMat` | ✅ | 泛型矩阵/向量描述符 |
| `musparseDestroySpMat/DestroyDnVec/DestroyDnMat` | ✅ | |
| `musparseSpMV/SpMM` | ✅ | 两段式(`temp_buffer=nullptr` → size 查询 → 分配 → 计算,action 参数驱动,**无独立 `_bufferSize` 符号**) |

## 结论

- v0.3 计划(Phase 2–6)所需符号 **83/83 在位**,头文件与 .so 一致,
  v0.3 风险登记表「muSOLVER 运行期符号与头文件不一致」一栏维持**低**概率评级。
- 已知边界(非缺口,设计已吸收):`gesdd` 不存在(svd 用 gesvd);`orgqr` 仅 S/D
  (复数 `cungqr`);`syevd` 仅 S/D(eigh 本就推迟 v0.4);musolver 共享句柄。
- 复测命令:`./tools/check_musax_symbols.sh [SDK路径]`(退出码非 0 即有 MISS)。
