#!/usr/bin/env python3
"""v0.3 Phase 6 sparse 性能基准（csr_matrix + spmv/spmm，muSPARSE）。

测量 GPU 上 sparse 算子（ADR-003 003-D4/D7）：
  - spmv：n×n 稀疏矩阵 @ 向量（吞吐口径 = 2·nnz·elem / t，稀疏 FLOPs 量级）
  - spmm：n×n 稀疏 @ n×k 稠密
  - 稀疏度扫描：nnz/n² ∈ {0.01, 0.1, 0.5}，观察稀疏度对延迟影响

计时方法（与 bench_fft.py 一致）：
    warmup(5) → sync → timed loop → sync → wall-clock / N

用法：
    python benchmark/bench_sparse.py [--iters 20] [--device 0] [--n 10000]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np  # noqa: E402
import musapy as ms  # noqa: E402


def _safe_sync(r):
    try:
        s = r.stream
        s.synchronize()
    except Exception:
        pass


def bench_latency_ms(fn, iters: int) -> float:
    for _ in range(5):
        _ = fn()
    r = fn()
    _safe_sync(r)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    r = fn()
    _safe_sync(r)
    t2 = time.perf_counter()
    return (t2 - t0) * 1000.0 / (iters + 1)


def make_csr(n: int, density: float, dtype, device_str: str):
    """随机稀疏矩阵（density 密度），直接生成 CSR 三元组（避免全矩阵 argwhere）。"""
    rng = np.random.default_rng(7)
    nnz = max(1, int(density * n * n))
    # 随机行分布（不均但足够基准用）
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    data = rng.normal(size=nnz).astype(np.float32 if dtype == 'f32' else np.float64)
    # indptr：按行计数（向量化）
    row_count = np.bincount(rows, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(row_count, out=ptr[1:])
    ind = cols.astype(np.int32)
    csr = ms.sparse.csr_matrix(
        (ms.array(data.tolist(), dtype=dtype, device=device_str),
         ms.array(ind.tolist(), dtype='i32', device=device_str),
         ms.array(ptr.tolist(), dtype='i32', device=device_str)),
        shape=(n, n),
    )
    return csr, nnz


def run_spmv_spmm(device_str: str, iters: int, n: int) -> None:
    print("-" * 72)
    print(f"  [spmv/spmm — {n}×{n} 稀疏矩阵，稀疏度扫描]")
    print("-" * 72)
    print(f"    {'density':>8} {'nnz':>9} {'spmv(ms)':>10} {'spmm k=4(ms)':>14} {'spmv GB/s':>10}")
    print(f"    {'─'*8} {'─'*9} {'─'*10} {'─'*14} {'─'*10}")
    for density in (0.01, 0.1, 0.5):
        for dtype in ('f32', 'f64'):
            csr, nnz = make_csr(n, density, dtype, device_str)
            v = ms.array(np.random.rand(n).astype(
                np.float32 if dtype == 'f32' else np.float64
            ).tolist(), dtype=dtype, device=device_str)
            B = ms.array(np.random.rand(n, 4).astype(
                np.float32 if dtype == 'f32' else np.float64
            ).tolist(), dtype=dtype, device=device_str)

            lat_v = bench_latency_ms(lambda: csr @ v, iters)
            lat_m = bench_latency_ms(lambda: csr @ B, iters)
            # spmv 有效带宽：读 data+indices（4+4 或 8+4 字节/nnz）+ 写 vec
            elem = 4 if dtype == 'f32' else 8
            gbps = (nnz * (elem + 4)) / (lat_v / 1000.0) / 1e9
            print(f"    {density:>8.2f} {nnz:>9} {lat_v:>10.3f} {lat_m:>14.3f} {gbps:>10.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="sparse (csr_matrix + spmv/spmm) benchmark")
    ap.add_argument("--iters", type=int, default=20, help="每配置迭代次数（默认 20）")
    ap.add_argument("--device", type=int, default=0, help="MUSA 设备 id（默认 0）")
    ap.add_argument("--n", type=int, default=10000, help="矩阵规模 n×n（默认 10000）")
    args = ap.parse_args()

    device_str = f"musa:{args.device}"
    try:
        ms.ones([1], device=device_str)
    except BaseException as e:  # noqa: BLE001
        print(f"SKIP: {device_str} not available ({e})")
        sys.exit(0)

    ms.set_default_device(device_str)

    print("=" * 72)
    print("  sparse 性能基准（v0.3 Phase 6: csr_matrix + spmv/spmm, muSPARSE）")
    print("=" * 72)
    print()
    for line in ms.device_summary().splitlines():
        print(f"    {line}")
    print()
    print(f"  iters = {args.iters}, n = {args.n}")
    print(f"  计时方法: warmup(5) → sync → timed({args.iters}+1) → sync → avg")
    print("-" * 72)

    run_spmv_spmm(device_str, args.iters, args.n)

    print("\n" + "=" * 72)
    print("  ✓ 测试结论")
    print("=" * 72)
    print("  ✓ spmv/spmm（f32/f64 × 稀疏度扫描）延迟表全部执行正常")
    print("  ⚠ 有效带宽口径：spmv 读 data+indices（4/8+4 字节/nnz）+ 写输出")
    print("  ⚠ toarray 走 D2H→host→H2D（正确性优先，未纳入吞吐基准）")
    print("=" * 72)


if __name__ == "__main__":
    main()
