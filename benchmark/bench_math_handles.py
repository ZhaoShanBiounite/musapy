#!/usr/bin/env python3
"""MUSA-X 句柄生命周期泄漏回归基准（v0.3 P1.7 验收）。

反复触发「懒创建 → SetStream → evict → 延迟销毁」闭环，验证：
  1. 延迟销毁队列最终归零（pending_destroys_after == 0）；
  2. musapy 记账的设备内存无净增长（mem_stats 持平）；
  3. 驱动级 VRAM 无净增长（vram_free 前后差在容差内）。

默认 1e6 轮（--iters 可调），对应计划验收 #2 的泄漏回归。

用法：
    source .venv/bin/activate
    python benchmark/bench_math_handles.py [--iters 1000000] [--device 0]

注意：
    - 冒烟入口 `_core._math_handle_smoke` 仅测试用，非公开 API。
    - 1e6 轮在真实 GPU 上约数十秒～分钟级（取决于句柄创建开销）。
"""

import argparse
import sys
import time
from pathlib import Path

# 确保能 import musapy（editable install 或 python-source）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import musapy as ms  # noqa: E402
from musapy import _core  # noqa: E402


def fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def main() -> None:
    ap = argparse.ArgumentParser(description="MUSA-X handle lifecycle leak regression")
    ap.add_argument("--iters", type=int, default=1_000_000, help="create/destroy 循环次数")
    ap.add_argument("--device", type=int, default=0, help="MUSA 设备 id")
    args = ap.parse_args()

    device = f"musa:{args.device}"

    print(f"== MUSA-X handle leak regression ==")
    print(f"device = {device}, iters = {args.iters:,}")
    print()

    # GPU 可用性探测
    try:
        ms.Device(device)
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: {device} not available ({e})")
        sys.exit(0)

    # 预热：首次懒创建句柄（VRAM 基线取在这之后，见 _math_handle_smoke 实现）
    _core._math_handle_smoke(device=device, iters=0)

    t0 = time.perf_counter()
    r = _core._math_handle_smoke(device=device, iters=args.iters)
    dt = time.perf_counter() - t0

    versions = r["versions"]
    print("library versions:")
    for lib in ("mublas", "murand", "mufft", "musparse"):
        print(f"  {lib:9s} = {versions[lib]}")
    print()

    iters = r["iters"]
    print(f"completed {iters:,} create/destroy cycles in {dt:.2f}s "
          f"({iters / dt:,.0f} cycles/s)" if dt > 0 else "completed (dt=0)")
    print()

    # ── 泄漏判定 ─────────────────────────────────────────────
    ok = True

    pending = r["pending_destroys_after"]
    print(f"pending_destroys_after = {pending}")
    if pending != 0:
        print(f"  FAIL: deferred destroy queue not drained ({pending} entries)")
        ok = False
    else:
        print("  PASS: destroy queue drained")

    ab, aa = r["mem_allocated_bytes_before"], r["mem_allocated_bytes_after"]
    nb, na = r["mem_allocated_buffers_before"], r["mem_allocated_buffers_after"]
    print(f"allocated bytes   : {ab} -> {aa} (delta {aa - ab})")
    print(f"allocated buffers : {nb} -> {na} (delta {na - nb})")
    if aa != ab or na != nb:
        print("  FAIL: musapy-tracked device memory not flat")
        ok = False
    else:
        print("  PASS: musapy-tracked memory flat")

    cached = r["mem_cached_bytes_after"]
    print(f"cached (deferred-free) after = {fmt_bytes(cached)}")
    if cached != 0:
        print("  FAIL: deferred-free cache not empty")
        ok = False
    else:
        print("  PASS: deferred-free cache empty")

    vb, va = r["vram_free_bytes_before"], r["vram_free_bytes_after"]
    if vb is not None and va is not None:
        delta = vb - va  # 正数 = VRAM 净占用增加（泄漏迹象）
        tol = 64 * 1024 * 1024  # 64 MiB 容差（驱动缓存/碎片波动）
        print(f"VRAM free before = {fmt_bytes(vb)}, after = {fmt_bytes(va)}, "
              f"net used delta = {fmt_bytes(delta)}")
        if delta > tol:
            print(f"  FAIL: VRAM grew by {fmt_bytes(delta)} beyond tolerance")
            ok = False
        else:
            print("  PASS: VRAM flat within tolerance")
    else:
        print("VRAM info unavailable; skipping driver-level check")

    print()
    if ok:
        print("RESULT: PASS — no leak detected")
        sys.exit(0)
    else:
        print("RESULT: FAIL — leak detected (see above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
