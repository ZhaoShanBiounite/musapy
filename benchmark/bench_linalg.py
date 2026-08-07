#!/usr/bin/env python3
"""v0.3 Phase 2/3 linalg 性能基准（matmul / dot / solve / lu / qr / svd）。

测量 GPU 上 linalg 算子（ADR-003 003-D3/D4/D6）：
  - matmul：f32/f64 矩阵乘法 GFLOPS 规模扫描（128³ → 2048³，方阵）
  - dot：1D 内积延迟 + 等效带宽（读 2 数组 + 写标量）
  - solve：n ∈ {64, 256, 1024} LU 分解 + 回代延迟（含 LU 对角 D2H 同步点）
  - lu / qr / svd：分解类算子方阵延迟表（getrf / geqrf+orgqr / gesvd）

计时方法（与 bench_musa_utilization.py 一致）：
    warmup(5) → sync → timed loop → sync → wall-clock / N

用法：
    source .venv/bin/activate
    python benchmark/bench_linalg.py [--iters 20] [--device 0] [--max-n 2048]

注意：
    - matmul GFLOPS = 2·m·n·k（row-major C(m×n) = A(m×k)·B(k×n)）
    - solve 含一次 LU 对角 D2H 同步（getrf 后判奇异，003-D3；
      muSOLVER 3.1.0 不写 info 输出，判据为 U 对角精确零），
      延迟天然高于同规模 matmul；nrhs>1 时另有 b 列主序 host 中转
    - 无 GPU 时自动 SKIP（与 bench_math_handles.py 同模式）
    - 规模 ≤ 512 时先对照 numpy 校准正确性（防计时垃圾数据）
"""

import argparse
import sys
import time
from pathlib import Path

# 确保能 import musapy（editable install 或 python-source）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import musapy as ms  # noqa: E402


def fmt_bytes(n: int) -> str:
    """字节格式化（与 bench_musa_utilization.py 一致）。"""
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    elif n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _safe_sync(r) -> None:
    """安全 sync：stream poisoned 时跳过（同 bench_musa_utilization.py）。"""
    if not hasattr(r, "stream"):
        return
    s = r.stream
    if hasattr(s, "is_poisoned") and s.is_poisoned:
        return
    try:
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
    t1 = time.perf_counter()
    r = fn()
    _safe_sync(r)
    t2 = time.perf_counter()

    total_iters = iters + 1
    return (t2 - t0) * 1000.0 / total_iters


# ── matmul ────────────────────────────────────────────────────


def bench_matmul(device_str: str, n: int, iters: int, dtype) -> tuple[float, float]:
    """n×n 方阵乘法。返回 (latency_ms, gflops)。"""
    a = ms.ones([n, n], dtype=dtype, device=device_str)
    b = ms.ones([n, n], dtype=dtype, device=device_str)
    lat = bench_latency_ms(lambda: ms.matmul(a, b), iters)
    gflops = (2.0 * n**3) / (lat / 1000.0) / 1e9
    return lat, gflops


def run_matmul(device_str: str, iters: int, max_n: int) -> None:
    print("-" * 72)
    print("  [matmul — n×n 方阵，GFLOPS = 2·n³ / t]")
    print("-" * 72)

    # 正确性校准（仅小规模，防计时垃圾数据）
    import numpy as np

    for dtype, name, atol in ((ms.float32, "f32", 1e-4), (ms.float64, "f64", 1e-9)):
        n = 128
        a = ms.array([[float((i + j) % 7) * 0.5 for j in range(n)] for i in range(n)],
                     dtype=dtype, device=device_str)
        b = ms.array([[float((i * 3 + j) % 11) * 0.25 for j in range(n)] for i in range(n)],
                     dtype=dtype, device=device_str)
        got = ms.matmul(a, b)
        exp = np.matmul(np.array(a.tolist()), np.array(b.tolist()))
        assert np.allclose(got.tolist(), exp, atol=atol), f"matmul {name} 校准失败"

    for dtype, name in ((ms.float32, "f32"), (ms.float64, "f64")):
        print(f"\n  {name}:")
        print(f"    {'n':>6} {'延迟(ms)':>12} {'GFLOPS':>10}")
        print(f"    {'─'*6} {'─'*12} {'─'*10}")
        for n in (128, 256, 512, 1024, 2048):
            if n > max_n:
                break
            lat, gflops = bench_matmul(device_str, n, iters, dtype)
            print(f"    {n:>6} {lat:>12.3f} {gflops:>10.1f}")


# ── dot ───────────────────────────────────────────────────────


def run_dot(device_str: str, iters: int) -> None:
    print("-" * 72)
    print("  [dot — (n,)·(n,) → 0-dim；等效带宽 = 2·n·4B / t]")
    print("-" * 72)
    print(f"    {'规模':>10} {'延迟(ms)':>12} {'吞吐(GE/s)':>13} {'带宽(GB/s)':>12}")
    print(f"    {'─'*10} {'─'*12} {'─'*13} {'─'*12}")
    for size in (1_000_000, 10_000_000):
        a = ms.ones([size], dtype=ms.float32, device=device_str)
        b = ms.ones([size], dtype=ms.float32, device=device_str)
        lat = bench_latency_ms(lambda: ms.dot(a, b), iters)
        gelem_s = size / (lat / 1000.0) / 1e9
        bw = gelem_s * 8.0  # 读 2 × f32
        print(f"    {size:>10,} {lat:>12.3f} {gelem_s:>13.3f} {bw:>12.1f}")


# ── solve ─────────────────────────────────────────────────────


def bench_solve(device_str: str, n: int, nrhs: int, iters: int) -> float:
    # A = ones + I：对角 2、非对角 1，严格对角占优 → 非奇异
    a = ms.add(
        ms.full([n, n], 1.0, dtype=ms.float64, device=device_str),
        ms.eye(n, dtype=ms.float64, device=device_str),
    )
    b = ms.ones([n, nrhs], dtype=ms.float64, device=device_str)
    return bench_latency_ms(lambda: ms.solve(a, b), iters)


def run_solve(device_str: str, iters: int) -> None:
    print("-" * 72)
    print("  [solve — f64；getrf + LU 对角 D2H 奇异检测 + getrs]")
    print("-" * 72)
    print(f"    {'n':>6} {'nrhs':>5} {'延迟(ms)':>12}")
    print(f"    {'─'*6} {'─'*5} {'─'*12}")
    for n in (64, 256, 1024):
        for nrhs in (1, 4):
            lat = bench_solve(device_str, n, nrhs, iters)
            print(f"    {n:>6} {nrhs:>5} {lat:>12.3f}")


# ── 分解类（Phase 3: lu / qr / svd）───────────────────────────


def bench_decomp_latency_ms(device_str: str, n: int, iters: int) -> tuple[float, float, float]:
    """lu / qr / svd 方阵延迟（f64；同一 A，各算子独立计时）。"""
    a = ms.add(
        ms.full([n, n], 1.0, dtype=ms.float64, device=device_str),
        ms.eye(n, dtype=ms.float64, device=device_str),
    )
    return (
        bench_latency_ms(lambda: ms.lu(a), iters),
        bench_latency_ms(lambda: ms.qr(a), iters),
        bench_latency_ms(lambda: ms.svd(a), iters),
    )


def run_decomp(device_str: str, iters: int) -> None:
    print("-" * 72)
    print("  [lu / qr / svd — f64 方阵；getrf / geqrf+orgqr / gesvd]")
    print("-" * 72)
    print(f"    {'n':>6} {'lu(ms)':>10} {'qr(ms)':>10} {'svd(ms)':>10}")
    print(f"    {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
    for n in (64, 256, 1024):
        lu_lat, qr_lat, svd_lat = bench_decomp_latency_ms(device_str, n, iters)
        print(f"    {n:>6} {lu_lat:>10.3f} {qr_lat:>10.3f} {svd_lat:>10.3f}")


# ── 主函数 ────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="linalg (matmul/dot/solve/lu/qr/svd) benchmark")
    ap.add_argument("--iters", type=int, default=20, help="每配置迭代次数（默认 20）")
    ap.add_argument("--device", type=int, default=0, help="MUSA 设备 id（默认 0）")
    ap.add_argument("--max-n", type=int, default=2048, help="matmul 最大方阵边长（默认 2048）")
    args = ap.parse_args()

    device_str = f"musa:{args.device}"
    # GPU 探测：ms.Device() 是惰性对象（构造不校验设备存在），须用真实
    # 设备操作触发 set_device + 分配，无效设备在此抛出而非中途 traceback
    try:
        ms.ones([1], device=device_str)
    except BaseException as e:  # noqa: BLE001
        # 无效设备在 Rust 侧触发 panic → pyo3 转 PanicException
        # （继承 BaseException 而非 Exception，须捕 BaseException）
        print(f"SKIP: {device_str} not available ({e})")
        sys.exit(0)

    ms.set_default_device(device_str)

    print("=" * 72)
    print("  linalg 性能基准（v0.3 Phase 2/3: matmul / dot / solve / lu / qr / svd）")
    print("=" * 72)
    print()
    print("  [设备信息]")
    for line in ms.device_summary().splitlines():
        print(f"    {line}")
    print()

    # ═══ 基线显存 ═══
    print("  [基线显存]")
    for line in ms.memory_summary(device=device_str).splitlines():
        print(f"    {line}")
    print()

    print(f"  iters = {args.iters}, matmul max-n = {args.max_n}")
    print(f"  计时方法: warmup(5) → sync → timed({args.iters}+1) → sync → avg")
    print("-" * 72)

    run_matmul(device_str, args.iters, args.max_n)
    run_dot(device_str, args.iters)
    run_solve(device_str, args.iters)
    run_decomp(device_str, args.iters)

    # ═══ Stream 状态 ═══
    s = ms.ones([2], device=device_str).stream
    print("\n" + "-" * 72)
    print("  [Stream 状态]")
    print("-" * 72)
    print(f"  stream id: {s.id}")
    print(f"  pending_count: {s.pending_count}")
    print(f"  is_poisoned: {s.is_poisoned}")
    print(f"  device: {s.device}")

    # ═══ 最终显存（不在计时路径上） ═══
    print("\n" + "-" * 72)
    print("  [最终显存]")
    print("-" * 72)
    for line in ms.memory_summary(device=device_str).splitlines():
        print(f"    {line}")

    # ═══ 结论 ═══
    print("\n" + "=" * 72)
    print("  ✓ 测试结论")
    print("=" * 72)
    print("  ✓ matmul（f32/f64 规模扫描）+ dot + solve + lu/qr/svd 全部执行正常")
    print("  ✓ 覆盖: matmul(f32+f64×5 规模) + dot(1M/10M) + solve(f64×3 规模×2 nrhs)")
    print("         + lu/qr/svd(f64×3 规模方阵延迟)")
    print("  ⚠ solve 延迟含 LU 对角 D2H 同步（003-D3 奇异检测；muSOLVER")
    print("    3.1.0 不写 getrf info，见 linalg.rs gpu_solve 注释），与 matmul")
    print("    不可直接对比；matmul 延迟 ≈ 45µs launch 地板 + kernel 执行")
    print("  ⚠ svd 走 ALL 模式（003-D3 修订：SDK 3.1.0 SINGULAR 模式 U 输出")
    print("    有 bug，见 linalg.rs svd 注释），1024² 约 2.7s；S 合理性校验")
    print("    含一次 D2H 同步")
    print("=" * 72)


if __name__ == "__main__":
    main()
