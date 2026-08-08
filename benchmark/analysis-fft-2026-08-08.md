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

## 4. 待办与已知限制

| 项 | 状态 |
|---|---|
| fftn / 多轴 / irfft | 推迟（用户确认 axis=-1 起步） |
| muFFT 吞吐优化（batched PlanMany 替代逐行） | 可选（当前 ~30 GB/s 天花板） |
| sdk-3.1.0-limitations：muFFT f64 慢 4×、吞吐规模退化 | 记入 SDK 限制表（可选） |
| bench_fft.py 并入 repo.md 全量报告 | v0.3 后期随 Phase 6/7 一并纳入 |
