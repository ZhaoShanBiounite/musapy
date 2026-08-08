# Benchmark 运行说明

前置：release 构建（`maturin develop --release`），在项目 venv 内运行。

## 脚本清单

| 脚本 | 测量范围 |
|---|---|
| `bench_linalg.py` | matmul / dot / solve / lu / qr / svd |
| `bench_musa_utilization.py` | elementwise / comparison / reduction / indexing |
| `bench_random.py` | rand / randn / uniform / normal / bernoulli |
| `bench_fft.py` | fft / ifft / rfft |
| `bench_sparse.py` | csr_matrix + spmv / spmm |

`bench_math_handles.py` 是句柄生命周期回归，非性能基准，不在此列。

## 规模档位

- **小**（~1M 元素）：launch 地板主导，反映固定开销
- **中**（~10M 元素）：真实带宽区间
- **大**（~64M 元素）：带宽饱和

`bench_random.py` / `bench_fft.py` 内部固定扫描 1M/10M/100M 三档，一次运行覆盖全部。

## 命令

### 小

```bash
python benchmark/bench_linalg.py --iters 20 --max-n 512
python benchmark/bench_musa_utilization.py --size 1000000 --iters 100
python benchmark/bench_random.py --iters 20
python benchmark/bench_fft.py --iters 20
python benchmark/bench_sparse.py --iters 15 --n 2000
```

### 中

```bash
python benchmark/bench_linalg.py --iters 20 --max-n 1024
python benchmark/bench_musa_utilization.py --size 10000000 --iters 100
python benchmark/bench_random.py --iters 20
python benchmark/bench_fft.py --iters 20
python benchmark/bench_sparse.py --iters 15 --n 2000
```

### 大

```bash
python benchmark/bench_linalg.py --iters 20 --max-n 2048
python benchmark/bench_musa_utilization.py --size 64000000 --iters 100
python benchmark/bench_random.py --iters 20
python benchmark/bench_fft.py --iters 20
python benchmark/bench_sparse.py --iters 15 --n 2000
```
