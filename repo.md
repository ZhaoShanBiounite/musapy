# musapy Benchmark 数据报告（2026-08-08）

**环境**：MTT S4000（mp_22, 56 CUs, 47.9 GB VRAM）· 分支 `feat/v0.3-musax-ffi`
（release 构建）· 原始输出 `/tmp/results-2026-08-08-full/`（基线 `3417ef8`）与
`/tmp/results-2026-08-08-final/`（最终复测，见 §8）

**范围**：`bench_linalg.py` + `bench_musa_utilization.py`（1M/10M/64M 三档）+
`bench_random.py` + `bench_fft.py` + `bench_sparse.py`；排除 `bench_math_handles.py`
（句柄生命周期回归，非性能基准）。全部 exit=0。

---

## 1. linalg（matmul / dot / solve / lu / qr / svd，iters=20）

### 1.1 matmul — n×n 方阵，GFLOPS = 2·n³/t

| n | f32 延迟(ms) | f32 GFLOPS | f64 延迟(ms) | f64 GFLOPS |
|---|---|---|---|---|
| 128 | 0.103 | 40.8 | 0.407 | 10.3 |
| 256 | 0.130 | 258.7 | 0.763 | 44.0 |
| 512 | 0.184 | 1461.5 | 2.784 | 96.4 |
| 1024 | 0.299 | 7183.1 | 13.769 | 156.0 |
| 2048 | 1.237 | 13885.0 | 108.142 | 158.9 |

### 1.2 dot — (n,)·(n,) → 0-dim

| 规模 | 延迟(ms) | 吞吐(GE/s) | 带宽(GB/s) |
|---|---|---|---|
| 1,000,000 | 0.248 | 4.034 | 32.3 |
| 10,000,000 | 0.394 | 25.387 | 203.1 |

### 1.3 solve — f64（getrf + 奇异检测 + getrs）

| n | nrhs=1 (ms) | nrhs=4 (ms) |
|---|---|---|
| 64 | 3.190 | 1.936 |
| 256 | 4.937 | 4.775 |
| 1024 | 25.786 | 26.483 |

### 1.4 lu / qr / svd — f64 方阵

| n | lu(ms) | qr(ms) | svd(ms) |
|---|---|---|---|
| 64 | 0.911 | 15.383 | 31.698 |
| 256 | 3.174 | 83.323 | 299.495 |
| 1024 | 17.740 | 445.312 | 2702.679 |

---

## 2. utilization — elementwise / comparison / reduction（延迟 + 带宽）

### 2.1 @1M（小，launch 地板主导）

| 类别 | 算子 | 延迟(ms) | 带宽(GB/s) |
|---|---|---|---|
| elementwise | add/sub/mul/div | 0.054–0.056 | 215–221 |
| elementwise | sin/cos/exp/log/abs/sign/neg | 0.050–0.054 | 148–160 |
| elementwise | pow | 0.073 | 165 |
| elementwise | clamp | 0.059 | 135 |
| comparison | gt/lt/ge/le/eq/ne | 0.057–0.058 | 156–157 |
| reduction | sum/prod/max/min/mean | 0.084–0.086 | 46.6–47.5 |
| reduction | argmax/argmin | 0.088–0.090 | 44.2–45.2 |
| reduction | cumsum | 0.306 | 26.2 |

分类峰值：elementwise 221 GB/s · comparison 157 GB/s · reduction 47 GB/s

### 2.2 @10M（中，真实带宽）

| 类别 | 算子 | 延迟(ms) | 带宽(GB/s) |
|---|---|---|---|
| elementwise | add/sub/mul/div | 0.211–0.215 | 559–569 |
| elementwise | sin/cos/exp/log/abs/sign/neg | 0.158–0.163 | 490–506 |
| elementwise | pow | 0.251 | 479 |
| elementwise | clamp | 0.221 | 362 |
| comparison | 全部 6 个 | 0.219 | 410–411 |
| reduction | sum/prod/max/min/mean | 0.246–0.247 | 161.8–162.4 |
| reduction | argmax/argmin | 0.303–0.311 | 128.7–132.1 |
| reduction | cumsum | 1.730 | 46.3 |

分类峰值：elementwise 569 GB/s · comparison 411 GB/s · reduction 162 GB/s

### 2.3 @64M（大，饱和带宽）

| 类别 | 算子 | 延迟(ms) | 带宽(GB/s) |
|---|---|---|---|
| elementwise | add/sub/mul/div | 1.104–1.113 | 690–696 |
| elementwise | sin/cos/exp/log/abs/sign | 0.757–0.764 | 670–676 |
| elementwise | neg | 0.821 | 624 |
| elementwise | pow | 1.212 | 634 |
| elementwise | clamp | 1.172 | 437 |
| comparison | 全部 6 个 | 1.391–1.398 | 412–414 |
| reduction | sum/prod/max/min/mean | 1.170–1.175 | 217.8–218.7 |
| reduction | argmax/argmin | 1.550–1.589 | 161.1–165.2 |
| reduction | cumsum | 2.899 | 46.3 |

分类峰值：elementwise **696 GB/s** · comparison 414 GB/s · reduction 219 GB/s

### 2.4 2D reduction 专项（256×256，各规模相同，地板主导）

| 算子 | 延迟(ms) |
|---|---|
| sum(axis=0) | 0.060 |
| sum(axis=1) | 0.057 |
| sum(global) | 0.073 |
| mean(axis=1) | 0.057 |
| max(axis=0) | 0.059 |
| argmax(axis=1) | 0.141 |
| cumsum(axis=1) | 0.067 |

### 2.5 Indexing 专项

| 算子 | 1M (ms) | 10M (ms) | 64M (ms) | 64M 带宽(GB/s) |
|---|---|---|---|---|
| transpose(view) | 0.000 | 0.001 | 0.000 | 零拷贝 |
| flip(view) | 0.000 | 0.000 | 0.000 | 零拷贝 |
| slice(view) | 0.001 | 0.001 | 0.001 | 零拷贝 |
| gather(full) | 0.177 | 1.302 | 7.961 | 64.3 |
| scatter(full) | 0.240 | 1.538 | 9.206 | 111.2 |
| contig(flat) | 0.000 | 0.000 | 0.000 | 零拷贝 |
| contig(transp) | 0.063 | 0.345 | 1.600 | 319.9 |
| contig(flip) | 0.104 | 0.651 | 4.223 | 121.2 |

---

## 3. random（rand/randn 吞吐 + uniform/normal/bernoulli 延迟）

### 3.1 rand/randn — 吞吐 GB/s

| n | rand f32 | rand f64 | randn f32 | randn f64 |
|---|---|---|---|---|
| 1M | 64.2 | 100.8 | 45.3 | 1.39 |
| 10M | 283.0 | 256.1 | 234.3 | 2.74 |
| 100M | 136.7 | 119.6 | 134.0 | 3.04 |

### 3.2 uniform / normal / bernoulli — 延迟(ms)

| n | uniform(f64) | normal(f64) | bernoulli(f32) |
|---|---|---|---|
| 1M | 0.394 | 6.146 | 0.230 |
| 10M | 0.990 | 28.669 | 0.441 |
| 100M | 12.340 | 257.020 | 3.090 |

---

## 4. fft（fft/ifft/rfft，muFFT，axis=-1；2026-08-08 真机新增）

> 注：`musapy_mock_musa` cfg 仅在 SDK 缺失时发出，本机 SDK 3.1.0 常驻，
> mock naive-DFT stub 从未编译——fft 数据均为真机 mufft 执行。

### 4.1 延迟（ms）与吞吐

| n | fft f32 | ifft f32 | rfft f32 | fft f64 | ifft f64 | rfft f64 |
|---|---|---|---|---|---|---|
| 1M | 0.391 | 0.429 | 0.260 | 1.480 | 1.526 | 0.979 |
| 10M | 2.632 | 2.879 | 1.364 | 16.623 | 17.066 | 6.921 |
| 100M | 33.906 | 36.277 | 18.507 | 194.737 | 199.438 | 92.759 |

f32 吞吐 ~20-30 GB/s（天花板）；f64 慢 3.8-4×（mp_22 FP64 仿真，同 dgemm 特征）；
rfft 输出减半故约快 2×。2D+ 走 **batched PlanMany 单次 Exec**（P-FFT-1）：
`fft(64×4096)` 0.262 ms（对比逐行 Exec 6.432 ms，**24.5× 加速**）。详细归因见
[benchmark/analysis-fft-2026-08-08.md](benchmark/analysis-fft-2026-08-08.md)。

### 4.2 数值

真机对照 np.fft：fft/ifft/rfft rtol 1e-10（f64）/1e-5（f32）；ifft 圆整性、norm 三值、
n 截断/补零、2D batched 全部通过（test_fft.py，24 用例）。

---

## 5. sparse（csr_matrix + spmv/spmm/toarray，muSPARSE；2026-08-08 真机新增）

**范围**：只做 `csr_matrix`（coo 推迟）；data f32/f64，indices/indptr 须 int32；
GPU-only（003-D4）；`@` 收 ms.Array 直连 + ndarray/list 经 tolist 转 device。

### 5.1 延迟与吞吐（2000×2000 稀疏矩阵，density 扫描）

| density | nnz | spmv(ms) | spmm k=4(ms) | spmv 有效带宽(GB/s) |
|---|---|---|---|---|
| 0.01 | 40,000 | 0.65-0.67 | 0.055-0.064 | 0.5-0.7 |
| 0.10 | 400,000 | 0.81-0.83 | 0.104-0.163 | 3.9-5.8 |
| 0.50 | 2,000,000 | 0.97-1.12 | 0.329-0.644 | 16.5-21.5 |

spmv 低密度下延迟被固定开销主导（每调用 create/destroy 描述符 + 两段式查询），
带宽随 nnz 上升；spmm 快 ~10×（批量列处理高效）。详细归因见
[benchmark/analysis-sparse-2026-08-08.md](benchmark/analysis-sparse-2026-08-08.md)。

### 5.2 数值

真机对照 NumPy（f32/f64 × 多 shape × spmv/spmm/toarray 全绿）；空矩阵（nnz=0）
输出全零；错误路径（shape/dtype/CPU 拒绝）覆盖（test_sparse.py，19 用例）。

---

## 6. 健康状态

- Stream：pending=0，is_poisoned=False（全部 benchmark）
- 显存：linalg 最终 Allocated 45.3 MB（7 buffers）· Peak 160.4 MB · 无 deferred-free 残留
- bench_random 完整通过（大块 buffer 立即 free 修复 `3417ef8` 后 4/4 次成功）

---

## 7. 关键数字速查

| 指标 | 数值 |
|---|---|
| elementwise 峰值带宽（64M） | 696 GB/s |
| reduction 峰值带宽（64M） | 219 GB/s |
| comparison 峰值带宽（64M） | 414 GB/s |
| f32 matmul 峰值 | 13.9 TFLOPS（n=2048） |
| f64 matmul 封顶 | ~159 GFLOPS |
| solve(1024) | 25.8 ms |
| lu(1024) / qr(1024) / svd(1024) | 17.7 / 445 / 2703 ms |
| randn f64 吞吐 | ~3 GB/s（SDK 限制） |
| uniform(100M) / bernoulli(100M) | 12.3 / 3.1 ms |
| fft 1M/10M/100M（f32） | 0.39 / 2.64 / 33.9 ms |
| fft 2D 64×4096（batched） | 0.262 ms（逐行对比 6.43 ms，P-FFT-1 24.5×） |
| fft f64（10M） | 16.6 ms（mp_22 FP64 仿真，慢 4×） |
| spmv 2000² d=0.5 | ~1.0 ms / ~21 GB/s（CSR 随机访存） |

---

## 8. 最终全量复测注记（2026-08-08）

除 `bench_math_handles.py` 外的全部 benchmark 于 2026-08-08 重跑
（`bench_linalg.py` + `bench_musa_utilization.py` 1M/10M/64M 三档 +
`bench_random.py` + `bench_fft.py` + `bench_sparse.py`，全部 exit=0），
与 `3417ef8` 基线及 fft/sparse 提交时数据**逐项一致，无回归**：
matmul f32 13895 GFLOPS、elementwise 64M 696 GB/s、sum 64M 1.19 ms、
cumsum 64M 2.94 ms（16.7M 上限）、randn f64 ~3 GB/s、fft 2D 0.262 ms
（P-FFT-1 收益保持 24.5×）、spmv ~1 ms。原始输出 `/tmp/results-2026-08-08-final/`。
