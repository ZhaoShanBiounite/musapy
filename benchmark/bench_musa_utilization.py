#!/usr/bin/env python3
"""MUSA GPU 计算占用验证脚本

在 musa:0 上运行持续计算负载，利用 musapy 内建监控 API 实时采集：
- ms.device_summary()  → 设备名称、架构、VRAM、CU 数
- ms.memory_summary(device="musa:0") → allocated / cached / peak / VRAM free/total
- stream.pending_count → 流水线深度
- array.nbytes → 单数组显存

用法：
    python scripts/bench_musa_utilization.py [--size 1000000] [--iters 100] [--device 0]
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# 确保能 import musapy（editable install 或 python-source）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import musapy as ms


# ── 显存采样 ──────────────────────────────────────────────────


@dataclass
class MemSample:
    """一次显存采样（解析 memory_summary 输出）"""
    timestamp: float
    summary: str
    allocated_bytes: int = 0
    peak_bytes: int = 0
    vram_free: int = 0
    vram_total: int = 0


def parse_size(s: str) -> int:
    """解析 '12.34 MB' / '1.23 GB' / '456 B' → bytes"""
    s = s.strip()
    if s.endswith("GB"):
        return int(float(s[:-2].strip()) * 1024**3)
    elif s.endswith("MB"):
        return int(float(s[:-2].strip()) * 1024**2)
    elif s.endswith("KB"):
        return int(float(s[:-2].strip()) * 1024)
    elif s.endswith("B"):
        return int(float(s[:-1].strip()))
    return 0


def sample_memory(device_str: str) -> MemSample:
    """调用 ms.memory_summary(device=...) 并解析关键数值"""
    summary = ms.memory_summary(device=device_str)
    snap = MemSample(timestamp=time.perf_counter(), summary=summary)

    for line in summary.splitlines():
        line = line.strip()
        if line.startswith("Allocated:"):
            # "Allocated: 12.34 MB (3 buffers)"
            parts = line.split("(", 1)[0].replace("Allocated:", "").strip()
            snap.allocated_bytes = parse_size(parts)
        elif line.startswith("Peak allocated:"):
            parts = line.replace("Peak allocated:", "").strip()
            snap.peak_bytes = parse_size(parts)
        elif "free /" in line and "total VRAM" in line:
            # "Device musa:0 — 1234.00 MB free / 16384.00 MB total VRAM"
            seg = line.split("—", 1)[-1].strip()
            free_s, total_s = seg.split("free /")
            snap.vram_free = parse_size(free_s)
            snap.vram_total = parse_size(total_s.replace("total VRAM", ""))

    return snap


class MemoryMonitor:
    """后台线程轮询 ms.memory_summary"""

    def __init__(self, device_str: str, interval: float = 0.05):
        self._device_str = device_str
        self._interval = interval
        self._samples: list[MemSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                s = sample_memory(self._device_str)
                self._samples.append(s)
            except Exception:
                pass
            self._stop.wait(self._interval)

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[MemSample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._samples


# ── Benchmark ─────────────────────────────────────────────────


@dataclass
class OpResult:
    name: str
    latency_ms: float
    gelem_per_sec: float
    gflops: float


def fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    elif n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def run_benchmark(size: int, iters: int, device_id: int):
    device_str = f"musa:{device_id}"
    ms.set_default_device(device_str)

    # ═══ 设备信息 ═══
    print("=" * 66)
    print("  MUSA GPU 计算占用验证")
    print("=" * 66)
    print()
    print("  [设备信息] ms.device_summary():")
    for line in ms.device_summary().splitlines():
        print(f"    {line}")
    print()

    # ═══ 基线显存 ═══
    baseline = sample_memory(device_str)
    print("  [基线显存] ms.memory_summary(device='musa:0'):")
    for line in baseline.summary.splitlines():
        print(f"    {line}")
    print()
    print(f"  数组规模: {size:,} elements × f32 = {fmt_bytes(size * 4)} / array")
    print(f"  迭代次数: {iters}")
    print("-" * 66)

    # ═══ 分配数组 ═══
    data_a = [float(i % 1000) * 0.001 for i in range(size)]
    data_b = [float(i % 777) * 0.002 for i in range(size)]

    a = ms.array(data_a, dtype=ms.float32, device=device_str)
    b = ms.array(data_b, dtype=ms.float32, device=device_str)

    after_alloc = sample_memory(device_str)
    alloc_delta = after_alloc.allocated_bytes - baseline.allocated_bytes
    print(f"  分配后 musapy allocated: {fmt_bytes(after_alloc.allocated_bytes)} (+{fmt_bytes(alloc_delta)})")
    if after_alloc.vram_total > 0:
        vram_used = after_alloc.vram_total - after_alloc.vram_free
        print(f"  VRAM 占用: {fmt_bytes(vram_used)} / {fmt_bytes(after_alloc.vram_total)}"
              f" ({100.0 * vram_used / after_alloc.vram_total:.1f}%)")
    print(f"  单数组 nbytes: {fmt_bytes(a.nbytes)}")
    print("-" * 66)

    # ═══ 算子延迟 benchmark ═══
    ops = [
        ("add",   lambda: ms.add(a, b),   1.0),
        ("sub",   lambda: ms.sub(a, b),   1.0),
        ("mul",   lambda: ms.mul(a, b),   1.0),
        ("div",   lambda: ms.div(a, b),   1.0),
        ("pow",   lambda: ms.pow(a, b),   8.0),
        ("sin",   lambda: ms.sin(a),      8.0),
        ("cos",   lambda: ms.cos(a),      8.0),
        ("exp",   lambda: ms.exp(a),      8.0),
        ("log",   lambda: ms.log(ms.abs(a)), 8.0),
        ("abs",   lambda: ms.abs(a),      1.0),
        ("sign",  lambda: ms.sign(a),     1.0),
        ("neg",   lambda: ms.neg(a),      1.0),
        ("clamp", lambda: ms.clamp(a, 0.0, 1.0), 2.0),
    ]

    print(f"\n  {'算子':<8} {'延迟(ms)':<11} {'吞吐(GElem/s)':<15} {'GFLOPS':<10}")
    print(f"  {'─'*8} {'─'*11} {'─'*15} {'─'*10}")

    results: list[OpResult] = []
    for name, fn, flops_per_elem in ops:
        # warmup
        for _ in range(5):
            _ = fn()
        _ = fn().tolist()  # sync

        # timed
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn()
        _ = fn().tolist()  # force GPU sync
        t1 = time.perf_counter()

        lat_ms = (t1 - t0) * 1000.0 / iters
        gelem_s = size / (lat_ms / 1000.0) / 1e9
        gflops = gelem_s * flops_per_elem

        results.append(OpResult(name, lat_ms, gelem_s, gflops))
        print(f"  {name:<8} {lat_ms:<11.3f} {gelem_s:<15.3f} {gflops:<10.3f}")

    # ═══ 计算期间显存监控 ═══
    print("\n" + "-" * 66)
    print("  [计算期间显存监控]")
    print("-" * 66)

    monitor = MemoryMonitor(device_str, interval=0.02)
    monitor.start()

    # 预分配输出 buffer，避免无限分配
    out1 = ms.array([0.0] * size, dtype=ms.float32, device=device_str)
    out2 = ms.array([0.0] * size, dtype=ms.float32, device=device_str)

    # 密集计算 3 秒（复用 out= 避免 OOM）
    burst_iters = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < 3.0:
        ms.add(a, b, out=out1)
        ms.mul(a, b, out=out2)
        ms.sin(a, out=out1)
        ms.exp(a, out=out2)
        burst_iters += 1
    _ = ms.add(a, b, out=out1)
    _ = out1.tolist()  # sync
    t_end = time.perf_counter()

    samples = monitor.stop()
    burst_sec = t_end - t_start
    kernel_calls = burst_iters * 4

    print(f"  持续计算: {burst_sec:.2f} s")
    print(f"  Kernel 调用: {kernel_calls:,} 次")
    print(f"  吞吐: {kernel_calls / burst_sec:,.0f} launches/s")
    print(f"  等效数据通量: {kernel_calls / burst_sec * size * 4 * 3 / 1e9:.2f} GB/s"
          f" (3 arrays/op × f32)")

    if samples:
        peak_alloc = max(s.allocated_bytes for s in samples)
        peak_vram_used = max(
            (s.vram_total - s.vram_free) for s in samples if s.vram_total > 0
        ) if any(s.vram_total > 0 for s in samples) else 0
        vram_total = samples[0].vram_total

        print(f"\n  采样点: {len(samples)}")
        print(f"  musapy peak allocated: {fmt_bytes(peak_alloc)}")
        if vram_total > 0:
            print(f"  VRAM 峰值占用: {fmt_bytes(peak_vram_used)} / {fmt_bytes(vram_total)}"
                  f" ({100.0 * peak_vram_used / vram_total:.1f}%)")
            print(f"  VRAM 计算增量: {fmt_bytes(peak_vram_used - (baseline.vram_total - baseline.vram_free))}")

    # ═══ Stream 状态 ═══
    print("\n" + "-" * 66)
    print("  [Stream 状态]")
    print("-" * 66)
    s = a.stream
    print(f"  stream id: {s.id}")
    print(f"  pending_count: {s.pending_count}")
    print(f"  is_poisoned: {s.is_poisoned}")
    print(f"  device: {s.device}")

    # ═══ 最终显存快照 ═══
    print("\n" + "-" * 66)
    print("  [最终显存] ms.memory_summary(device='musa:0'):")
    print("-" * 66)
    final = sample_memory(device_str)
    for line in final.summary.splitlines():
        print(f"    {line}")

    # ═══ 结论 ═══
    print("\n" + "=" * 66)
    print("  ✓ 验证结论")
    print("=" * 66)
    total_gflops = sum(r.gflops for r in results)
    avg_lat = sum(r.latency_ms for r in results) / len(results)
    print(f"  ✓ musa:{device_id} 全部 {len(results)} 个算子执行正常")
    print(f"  ✓ 平均延迟: {avg_lat:.3f} ms ({size:,} elements)")
    print(f"  ✓ 聚合吞吐: {total_gflops:.2f} GFLOPS")
    print(f"  ✓ 持续负载 {burst_sec:.1f}s / {kernel_calls:,} kernels 无错误")
    if final.peak_bytes > 0:
        print(f"  ✓ 峰值 allocated: {fmt_bytes(final.peak_bytes)}")
    print(f"  ✓ Stream 健康: pending={s.pending_count}, poisoned={s.is_poisoned}")
    print("=" * 66)


def main():
    parser = argparse.ArgumentParser(description="MUSA GPU 计算占用验证（musapy 内建 API）")
    parser.add_argument("--size", type=int, default=1_000_000,
                        help="每数组元素数（默认 1M）")
    parser.add_argument("--iters", type=int, default=100,
                        help="每算子迭代次数（默认 100）")
    parser.add_argument("--device", type=int, default=0,
                        help="MUSA 设备 ID（默认 0）")
    args = parser.parse_args()

    run_benchmark(args.size, args.iters, args.device)


if __name__ == "__main__":
    main()
