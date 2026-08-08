# Benchmark 分析（2026-08-08）：fft 套件 + complex 落地（v0.3 Phase 5）

分支 `feat/v0.3-musax-ffi`，**MTT S4000 真机**（mp_22, 56 CUs, 47.9 GB VRAM），
**release 构建**（`maturin develop --release`）。原始输出：`benchmark/bench_fft.py --iters 20`。

> 说明：`musapy_mock_musa` cfg 只在 SDK 缺失时发出；本机 SDK 3.1.0 一直存在，
> 故 mock stub（naive DFT）从未被编译——**本 Phase 全部验证均为真机执行**。

## 1. fft 真机性能（muFFT，axis=-1）

| n | fft f32(ms) | ifft f32(ms) | rfft f32(ms) | fft f64(ms) | ifft f64(ms) | rfft f64(ms) |
|---|---|---|---|---|---|---|
| 1,000,000 | 0.391 | 0.429 | 0.260 | 1.480 | 1.526 | 0.979 |
| 10,000,000 | 2.632 | 2.879 | 1.364 | 16.623 | 17.066 | 6.921 |
| 100,000,000 | 33.906 | 36.277 | 18.507 | 194.737 | 199.438 | 92.759 |

吞吐（f32：20.4→30.4→23.6 GB/s；f64：10.8→9.6→8.2 GB/s）。

### 归因与特征（muFFT/SDK 侧）

1. **f64 比 f32 慢 3.8-4×**：mp_22 无原生 FP64（软件仿真），与 linalg dgemm 的
   f64 特征一致（见 analysis-2026-08-07 §2.4）。f64 fft 吞吐 ~8-11 GB/s。
2. **吞吐天花板 ~30 GB/s**（f32），远低于 elementwise 的 696 GB/s——muFFT 1D
   性能由 FFT 计算/规约主导而非内存带宽；且 musapy 当前按「每行 Plan1d batch=1
   逐行执行」，多行未走 batched plan（batch 批量优化为后续可选）。
3. **规模非线性**：1M→10M 吞吐上升（launch 地板稀释减弱），10M→100M 回落
   （SDK FFT 规模特征，与 rand 100M 退化同族）。
4. **rfft 约快 2×**：输出 N//2+1 减半，天然省一半写带宽与计算。

### 2D 逐行

`fft(64×4096)` 6.432 ms vs `fft(4096)` 单行 0.166 ms → 逐行偏移正确，
实际耗时略优于线性（64×0.166≈10.6ms），有 plan 复用/调度效应。

## 2. 真机数值验证（对照 NumPy）

- `test_fft.py`（24 用例）：fft/ifft/rfft 数值对照 np.fft **rtol 1e-10（f64）/ 1e-5（f32）**；
  圆整性 ifft(fft(x))≈x；norm 三值；n 截断/补零；2D 逐行；错误路径——**全部真机通过**
- `test_complex.py`（18 用例）：complex 创建/运算/提升/视图对照 NumPy——**全部真机通过**
- 全量 pytest：**517 passed**（release 构建，8.75s）

## 3. complex 真机 kernel

mcc 3.1.0 编译 complex struct kernel（c64/c128 分量公式）兼容性在本机冒烟通过
（`/tmp/smoke_cplx.mu`，EXIT=0）；真机运行 add/sub/mul/div/neg/abs/eq-ne + cast/
resize/scale 全部数值正确（test_complex.py 对照 NumPy）。

## 4. 瓶颈归因（C 探针直连 mufft 分解，2026-08-08）

用 `gcc` 探针（`/tmp/probe_fft.c`、`/tmp/probe_fft2.c`）直连 muFFT，对比 musapy
包装开销与 mufft 本体，定位三类瓶颈：

### 4.1 🔴 musapy 侧（可优化，收益最大）：2D+ 逐行执行 Plan1d batch=1

| 场景 | 逐行（musapy 现状） | batched PlanMany 一次 Exec | 加速 |
|---|---|---|---|
| 64×16384 f32 C2C | 2.618 ms | 0.116 ms | **22.5×** |
| 64×4096 f32 C2C | 1.991 ms | 0.040 ms | **49.4×** |

- 每次 `mufftExecC2C` 有 ~31µs 固定 launch 地板（逐行 1.99/64 ≈ 31µs/行），
  64 行累积成 2ms；batched 一次 Exec 摊销到所有行
- musapy 当前按 `MufftPlanSpec::OneD { batch: 1 }` **逐行循环 Exec**
  （`fft.rs` 的 `for row in 0..outer`）；改为 `PlanMany`（batch=outer）一次 Exec，
  64×4096 场景预计 6.4ms → ~0.2ms（**~30× 端到端**）
- **修复方案**：`math_handle::MufftPlanSpec::Many` 已存在，fft.rs 在 outer>1 时
  走 Many + 单次 Exec（idist/odist = n，istride/ostride = 1）

### 4.2 🟡 musapy 侧（次优）：real→complex cast 与 ifft scale

| 开销（10M 元素） | f32 | f64 | 占对应 op |
|---|---|---|---|
| real→fft 的 cast（`cast_array`） | 215 µs | 914 µs | fft real32 的 8% |
| ifft 的 1/N scale kernel | 244 µs | — | ifft 的 9% |

- cast 已可复用 rfft 的 R2C 直连路径（rfft 10M = 1.37ms 无 cast，是 fft real
  2.63ms 的一半）——fft/ifft 的 real 输入若改走「cast 合并进 resize」可省 215µs
- scale 是独立 launch（~240µs 含地板），量级不大，优先级低

### 4.3 ⚫ SDK 侧（无解，文档化）：mufft 本体 ~30 GB/s 天花板

- 纯 `mufftExecC2C` 1D 10M f32 = 2.619 ms（30.5 GB/s 有效带宽，扣除 H2D 干扰）
- musapy 1D 包装开销 ≈ 0（2.417ms vs 探针 2.619ms，差量为探针含 memcpy 排队）
- f64 慢 3.8-4×（mp_22 FP64 软件仿真，同 dgemm 特征）——SDK 侧，升级 SDK 前无解

### 优化实施结果（2026-08-08，P-FFT-1/2 已实施）

| 项 | 提交前 | 提交后 | 结论 |
|---|---|---|---|
| 2D fft (64×4096) | 6.432 ms | **0.263 ms** | ✅ **24.5× 加速**（探针预测 22-49× 吻合） |
| 1D fft 10M | 2.632 ms | 2.636 ms | ✅ 无回归（1D 无 batched 收益，mufft 本体瓶颈） |
| 1D fft 100M | 33.906 ms | 33.903 ms | ✅ 无回归 |
| 数值 | — | batched 路径对照 np.fft 全绿 | ✅（2D/real/截断/补零） |

- **P-FFT-1**：`fft.rs` outer>1 时改走 `MufftPlanSpec::Many`（batch=outer）+ 单次
  Exec；mock mufft stub 同步支持 batch/stride/dist（`MockFftPlan` 扩展）
- **P-FFT-2**：real 输入 n≠last_dim 走 `cast_resize` 合并 kernel（cast 扩 complex +
  截断/补零一步），省一次 kernel launch + 中间 buffer；10M real→fft 的 cast 开销
  从 215µs 降为合并 kernel 单次传递
- **P-FFT-3**：ifft 的 1/N scale 为独立 launch（~240µs @10M，占 9%），量级小，
  且合并需改 mufft 执行后回读路径——评估后保留（低优先级，记录不实施）
- 门禁：pytest 517 passed（真机 release）· cargo check 双模式通过

### 优化优先级建议

| 优先级 | 项 | 预期收益 | 归属 |
|---|---|---|---|
| P-FFT-1 | 2D+ 改 batched PlanMany | **已实施**：2D 24.5× | musapy 侧 ✅ |
| P-FFT-2 | real 输入 cast 合并进 resize | **已实施**：省 1 kernel + 中间 buffer | musapy 侧 ✅ |
| P-FFT-3 | ifft scale 合并/精简 | 保留（~9%，低优先级） | musapy 侧 ⏸️ |
| P-FFT-4 | mufft 本体 30 GB/s / f64 4× | 无解 | SDK（记录） |

## 5. 待办与已知限制

| 项 | 状态 |
|---|---|
| P-FFT-1 batched PlanMany | 待实施（收益最大） |
| fftn / 多轴 / irfft | 推迟（用户确认 axis=-1 起步） |
| sdk-3.1.0-limitations：muFFT 30 GB/s 天花板、f64 慢 4× | 记入 SDK 限制表（可选） |
| bench_fft.py 并入 repo.md 全量报告 | 已并入（§4） |
