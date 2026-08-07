# musapy Benchmark 数据报告（2026-08-08）

**环境**：MTT S4000（mp_22, 56 CUs, 47.9 GB VRAM）· 分支 `feat/v0.3-musax-ffi` @ `3417ef8`（release 构建）· 原始输出 `/tmp/results-2026-08-08-full/`

**范围**：`bench_linalg.py` + `bench_musa_utilization.py`（1M/10M/64M 三档）+ `bench_random.py`；排除 `bench_math_handles.py`（句柄生命周期回归，非性能基准）。全部 exit=0。

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

## 4. 健康状态

- Stream：pending=0，is_poisoned=False（全部 benchmark）
- 显存：linalg 最终 Allocated 45.3 MB（7 buffers）· Peak 160.4 MB · 无 deferred-free 残留
- bench_random 完整通过（大块 buffer 立即 free 修复 `3417ef8` 后 4/4 次成功）

---

## 5. 关键数字速查

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
