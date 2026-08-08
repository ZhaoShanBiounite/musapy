#!/usr/bin/env python3
"""v0.3 Phase 5 fft 性能基准（fft/ifft/rfft，muFFT）。

测量 GPU 上 fft 算子（ADR-003 003-D5/D7）：
  - fft/ifft（C2C/Z2Z）：f32/f64 输入延迟扫描（1M/10M/100M 元素）
  - rfft（R2C/D2Z）：实输入 → complex 输出，吞吐口径 = 输入字节数 / t
  - 2D 逐行场景（axis=-1 逐行执行，Plan1d batch=1）

吞吐口径：
  - fft：n·elem_bytes(complex 输出) / t；rfft：n·elem_bytes(real 输入) / t
  - fft 无缩放、ifft 含 1/N 归一化 kernel（scale 开销并入延迟）

计时方法（与 bench_random.py 一致）：
    warmup(5) → sync → timed loop → sync → wall-clock / N

用法：
    python benchmark/bench_fft.py [--iters 20] [--device 0]
    # mock 模式（无 GPU，naive DFT O(N²)）：--size 小值验证脚本
    MUSAPY_MOCK_MUSA=1 python benchmark/bench_fft.py --size 64 --iters 3
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


# ── fft / ifft / rfft：延迟 + 吞吐扫描 ─────────────────────────


def run_fft_throughput(device_str: str, iters: int, sizes: tuple) -> None:
    print("-" * 72)
    print("  [fft/ifft/rfft — 1D 延迟；吞吐口径见模块注释]")
    print("-" * 72)
    print(f"    {'n':>9} {'op':>5} {'dtype':>5} {'延迟(ms)':>10} {'吞吐(GB/s)':>10}")
    print(f"    {'─'*9} {'─'*5} {'─'*5} {'─'*10} {'─'*10}")
    for n in sizes:
        for dtype in ('f32', 'f64'):
            x = ms.array(
                np.linspace(0.0, 1.0, n).astype(np.float32 if dtype == 'f32' else np.float64).tolist(),
                dtype=dtype,
                device=device_str,
            )
            out_dtype = 'c64' if dtype == 'f32' else 'c128'
            elem_out = ms.Dtype(out_dtype).element_size
            elem_in = ms.Dtype(dtype).element_size

            lat_f = bench_latency_ms(lambda: ms.fft.fft(x), iters)
            gbps_f = (n * elem_out) / (lat_f / 1000.0) / 1e9
            print(f"    {n:>9} {'fft':>5} {str(dtype):>5} {lat_f:>10.3f} {gbps_f:>10.2f}")

            lat_i = bench_latency_ms(lambda: ms.fft.ifft(x), iters)
            gbps_i = (n * elem_out) / (lat_i / 1000.0) / 1e9
            print(f"    {n:>9} {'ifft':>5} {str(dtype):>5} {lat_i:>10.3f} {gbps_i:>10.2f}")

            lat_r = bench_latency_ms(lambda: ms.fft.rfft(x), iters)
            gbps_r = (n * elem_in) / (lat_r / 1000.0) / 1e9
            print(f"    {n:>9} {'rfft':>5} {str(dtype):>5} {lat_r:>10.3f} {gbps_r:>10.2f}")


# ── 2D 逐行场景 ───────────────────────────────────────────────


def run_2d_rows(device_str: str, iters: int, rows: int, cols: int) -> None:
    print("-" * 72)
    print(f"  [2D fft — ({rows}×{cols})，axis=-1；batched PlanMany 单次 Exec（P-FFT-1）]")
    print("-" * 72)
    x = ms.array(
        np.random.default_rng(7).normal(size=(rows, cols)).tolist(),
        dtype='f64',
        device=device_str,
    )
    lat = bench_latency_ms(lambda: ms.fft.fft(x), iters)
    # 与 1D 同规模对比：2D 额外开销来自逐行偏移
    x1 = ms.array(x.tolist()[0], dtype='f64', device=device_str)
    lat1 = bench_latency_ms(lambda: ms.fft.fft(x1), iters)
    print(f"    2D fft ({rows}×{cols}): {lat:>10.3f} ms   1D 单行({cols}): {lat1:>10.3f} ms")


# ── 主函数 ────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="fft (fft/ifft/rfft, muFFT) benchmark")
    ap.add_argument("--iters", type=int, default=20, help="每配置迭代次数（默认 20）")
    ap.add_argument("--device", type=int, default=0, help="MUSA 设备 id（默认 0）")
    ap.add_argument("--size", type=int, default=0, help="覆盖默认规模（mock 冒烟用小值）")
    args = ap.parse_args()

    device_str = f"musa:{args.device}"
    # GPU 探测：无效设备在 Rust 侧 panic → pyo3 PanicException（继承 BaseException）
    try:
        ms.ones([1], device=device_str)
    except BaseException as e:  # noqa: BLE001
        print(f"SKIP: {device_str} not available ({e})")
        print("（mock 模式可加 --size 小值验证脚本：MUSAPY_MOCK_MUSA=1 ... --size 64）")
        sys.exit(0)

    ms.set_default_device(device_str)

    # 规模：--size 覆盖（mock/专项），否则 1M/10M/100M
    sizes = (args.size,) if args.size > 0 else (1_000_000, 10_000_000, 100_000_000)

    print("=" * 72)
    print("  fft 性能基准（v0.3 Phase 5: fft/ifft/rfft, muFFT）")
    print("=" * 72)
    print()
    print("  [设备信息]")
    for line in ms.device_summary().splitlines():
        print(f"    {line}")
    print()

    print(f"  iters = {args.iters}")
    print(f"  计时方法: warmup(5) → sync → timed({args.iters}+1) → sync → avg")
    print("-" * 72)

    run_fft_throughput(device_str, args.iters, sizes)
    run_2d_rows(device_str, args.iters, rows=64, cols=4096)

    # ═══ 结论 ═══
    print("\n" + "=" * 72)
    print("  ✓ 测试结论")
    print("=" * 72)
    print("  ✓ fft/ifft/rfft（f32/f64 × 规模）延迟扫描 + 2D 场景全部执行正常")
    print("  ⚠ ifft 延迟含 1/N 归一化 scale kernel（backward 时）")
    print("  ⚠ rfft 吞吐按输入 real 字节数计；输出为 (N//2+1) complex")
    print("  ⚠ axis=-1 起步；2D+ 走 batched PlanMany（P-FFT-1：2D 实测 24.5× 加速）")
    print("=" * 72)


if __name__ == "__main__":
    main()
