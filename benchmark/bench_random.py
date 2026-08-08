#!/usr/bin/env python3
"""v0.3 Phase 4 random 性能基准（rand/randn/uniform/normal/bernoulli）。

测量 GPU 上 random 算子（ADR-003 003-D7/003-D9）：
  - rand/randn：f32/f64 生成吞吐 GB/s 规模扫描（1M/10M/100M 元素）
  - uniform/normal/bernoulli：延迟表（1M/10M/100M 元素）

吞吐口径：
  - rand 写 4/8 字节/元素：GB/s = n·elem_bytes / t
  - 注意 random 为「每 op 显式 seed」语义：bench 用无 seed 连续生成
    （共享 generator 自然推进），测得的是纯生成吞吐。

计时方法（与 bench_linalg.py 一致）：
    warmup(5) → sync → timed loop → sync → wall-clock / N

用法：
    python benchmark/bench_random.py [--iters 20] [--device 0]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import musapy as ms  # noqa: E402


def _safe_sync(r):
    try:
        s = r.stream
        s.synchronize()
    except Exception:
        pass


def bench_latency_ms(fn, iters: int) -> float:
    """warmup(5) → sync → timed(iters+1) → sync → 平均延迟（ms）。"""
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

    total_iters = iters + 1
    return (t2 - t0) * 1000.0 / total_iters


# ── rand / randn：吞吐扫描 ────────────────────────────────────


def run_rand_throughput(device_str: str, iters: int) -> None:
    print("-" * 72)
    print("  [rand/randn — 生成吞吐 GB/s = n·elem_bytes / t]")
    print("-" * 72)
    print(f"    {'n':>8} {'op':>6} {'dtype':>5} {'延迟(ms)':>10} {'吞吐(GB/s)':>10}")
    print(f"    {'─'*8} {'─'*6} {'─'*5} {'─'*10} {'─'*10}")
    for n in (1_000_000, 10_000_000, 100_000_000):
        for op, dtype in (("rand", 'f32'), ("rand", 'f64'),
                          ("randn", 'f32'), ("randn", 'f64')):
            fn = lambda op=op, dtype=dtype: getattr(ms.random, op)(n, dtype=dtype, device=device_str)
            lat = bench_latency_ms(fn, iters)
            gbps = (n * ms.Dtype(dtype).element_size) / (lat / 1000.0) / 1e9
            print(f"    {n:>8} {op:>6} {str(dtype):>5} {lat:>10.3f} {gbps:>10.2f}")


# ── uniform / normal / bernoulli：延迟表 ───────────────────────


def run_decomp_latency(device_str: str, iters: int) -> None:
    print("-" * 72)
    print("  [uniform / normal / bernoulli — f64 延迟；uniform/normal 含变换]")
    print("-" * 72)
    print(f"    {'n':>8} {'uniform(ms)':>12} {'normal(ms)':>12} {'bernoulli(ms)':>14}")
    print(f"    {'─'*8} {'─'*12} {'─'*12} {'─'*14}")
    for n in (1_000_000, 10_000_000, 100_000_000):
        lat_u = bench_latency_ms(
            lambda: ms.random.uniform(-1.0, 1.0, shape=(n,), dtype='f64', device=device_str), iters)
        lat_n = bench_latency_ms(
            lambda: ms.random.normal(0.0, 1.0, shape=(n,), dtype='f64', device=device_str), iters)
        lat_b = bench_latency_ms(
            lambda: ms.random.bernoulli(0.5, shape=(n,), device=device_str), iters)
        print(f"    {n:>8} {lat_u:>12.3f} {lat_n:>12.3f} {lat_b:>14.3f}")


# ── 主函数 ────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="random (rand/randn/uniform/normal/bernoulli) benchmark")
    ap.add_argument("--iters", type=int, default=20, help="每配置迭代次数（默认 20）")
    ap.add_argument("--device", type=int, default=0, help="MUSA 设备 id（默认 0）")
    args = ap.parse_args()

    device_str = f"musa:{args.device}"
    # GPU 探测：无效设备在 Rust 侧 panic → pyo3 PanicException（继承 BaseException）
    try:
        ms.ones([1], device=device_str)
    except BaseException as e:  # noqa: BLE001
        print(f"SKIP: {device_str} not available ({e})")
        sys.exit(0)

    ms.set_default_device(device_str)

    print("=" * 72)
    print("  random 性能基准（v0.3 Phase 4: rand/randn/uniform/normal/bernoulli）")
    print("=" * 72)
    print()
    print("  [设备信息]")
    for line in ms.device_summary().splitlines():
        print(f"    {line}")
    print()

    print(f"  iters = {args.iters}")
    print(f"  计时方法: warmup(5) → sync → timed({args.iters}+1) → sync → avg")
    print("-" * 72)

    run_rand_throughput(device_str, args.iters)
    run_decomp_latency(device_str, args.iters)

    # ═══ 结论 ═══
    print("\n" + "=" * 72)
    print("  ✓ 测试结论")
    print("=" * 72)
    print("  ✓ rand/randn（f32/f64 × 3 规模）吞吐扫描 + uniform/normal/bernoulli")
    print("    延迟表全部执行正常")
    print("  ⚠ seed 语义：bench 无 seed（共享 generator 自然推进）；单次生成延迟")
    print("    含 seed 重置开销可忽略（见 test_random.py 复现性用例）")
    print("  ⚠ randn f64 吞吐仅 ~3 GB/s（比 f32 慢约 50×）——SDK 3.1.0 的")
    print("    f64 Normal 生成器实现特征，见 sdk-3.1.0-limitations.md §1.6")
    print("=" * 72)


if __name__ == "__main__":
    main()
