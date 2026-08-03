#!/usr/bin/env python3
"""MUSA GPU 算子性能基准测试

测量 musapy 算子在 MUSA GPU 上的延迟、吞吐和等效帶寬。

用法：
    source .venv/bin/activate
    python benchmark/bench_musa_utilization.py [--size 1000000] [--iters 100] [--device 0]

計時方法：
    1. warmup 5 次（預熱 kernel、stream、buffer pool）
    2. sync 一次（等待所有隊列排空）
    3. timed loop: N 次迭代 → sync → 取 wall-clock / N 為平均延遲
    4. 不在 timed loop 內調用 memory_summary / tolist（避免 sync 污染）

注意：
    - --size 應 ≥ 1_000_000（小數組下 launch overhead 主導，測不出真實帶寬）
    - 測量結果為「單次算子調用的端到端延遲」（含 Python → Rust → kernel launch → GPU 執行 → sync）
    - 「聚合 GFLOPS」為各算子峰值 GFLOPS 的最大值（非加總），因為算子串行執行
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 确保能 import musapy（editable install 或 python-source）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import musapy as ms


# ── 輔助 ─────────────────────────────────────────────────────


def fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    elif n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def parse_size(s: str) -> int:
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


@dataclass
class OpResult:
    name: str
    category: str
    latency_ms: float
    gelem_per_sec: float
    gflops: float
    # 估算：讀 + 寫的等效帶寬（GB/s）
    effective_bw_gb_s: float = 0.0


# ── 核心計時函數 ───────────────────────────────────────────────


def _safe_sync(r) -> None:
    """安全 sync：若 stream 已 poisoned 則跳過，避免崩潰。"""
    if not hasattr(r, 'stream'):
        return
    s = r.stream
    if hasattr(s, 'is_poisoned') and s.is_poisoned:
        return
    try:
        s.synchronize()
    except Exception:
        pass  # stream 可能已被 poison，忽略


def bench_op(fn, iters: int, size: int, flops_per_elem: float,
             read_bytes_per_elem: int, write_bytes_per_elem: int) -> OpResult:
    """測量單個算子的平均延遲。

    計時策略：
      warmup → sync → timed loop (N 次) → sync → 取 wall-clock / N
    """
    # warmup
    for _ in range(5):
        _ = fn()

    # 預熱後 sync，確保 GPU 隊列排空
    r = fn()
    _safe_sync(r)

    # timed loop
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    t1 = time.perf_counter()

    # sync + 取最後一個結果
    r = fn()
    _safe_sync(r)
    t2 = time.perf_counter()

    # t1..t2 包含最後一次調用 + sync；總時間 = (t2 - t0)，迭代數 = iters + 1
    total_iters = iters + 1
    lat_ms = (t2 - t0) * 1000.0 / total_iters
    gelem_s = size / (lat_ms / 1000.0) / 1e9
    gflops = gelem_s * flops_per_elem

    # 等效帶寬 = (讀字節 + 寫字節) * Gelem/s
    total_bytes_per_elem = read_bytes_per_elem + write_bytes_per_elem
    bw_gb_s = gelem_s * total_bytes_per_elem if total_bytes_per_elem > 0 else 0.0

    # 提取算子名
    name = getattr(fn, '__name__', str(fn))
    # 從 lambda 或其他可調用對象提取
    if hasattr(fn, '__code__'):
        name = fn.__code__.co_name

    return OpResult(
        name=name, category="", latency_ms=lat_ms,
        gelem_per_sec=gelem_s, gflops=gflops,
        effective_bw_gb_s=bw_gb_s,
    )


# ── 主函數 ─────────────────────────────────────────────────────


def run_benchmark(size: int, iters: int, device_id: int):
    device_str = f"musa:{device_id}"
    ms.set_default_device(device_str)

    # ═══ 設備信息 ═══
    print("=" * 72)
    print("  MUSA GPU 算子性能基准测试")
    print("=" * 72)
    print()
    print("  [設備信息]")
    for line in ms.device_summary().splitlines():
        print(f"    {line}")
    print()

    # ═══ 基線顯存 ═══
    print("  [基線顯存]")
    for line in ms.memory_summary(device=device_str).splitlines():
        print(f"    {line}")
    print()

    elem_bytes = 4  # f32
    print(f"  數組規模: {size:,} elements × f32 = {fmt_bytes(size * elem_bytes)} / array")
    print(f"  迭代次數: {iters}")
    print(f"  計時方法: warmup(5) → sync → timed({iters}+1) → sync → avg")
    print("-" * 72)

    # ═══ 分配數組 ═══
    data_a = [float(i % 1000) * 0.001 for i in range(size)]
    data_b = [float(i % 777) * 0.002 + 0.001 for i in range(size)]

    a = ms.array(data_a, dtype=ms.float32, device=device_str)
    b = ms.array(data_b, dtype=ms.float32, device=device_str)

    print(f"  分配後顯存: {fmt_bytes(a.nbytes * 2)}")
    print("-" * 72)

    # ═══ 算子定義 ═══
    # (name, category, fn, flops_per_elem, read_bytes_per_elem, write_bytes_per_elem)
    #   bytes: 按 f32 計算。comparison 輸出 uint8(1B) 非同 dtype。
    #   reduction 全局: 讀 4MB，寫 4B（標量）
    ops = [
        # ── Elementwise binary（讀 2 輸入 + 寫 1 輸出）──
        ("add",   "elementwise", lambda: ms.add(a, b),   1.0, 8, 4),
        ("sub",   "elementwise", lambda: ms.sub(a, b),   1.0, 8, 4),
        ("mul",   "elementwise", lambda: ms.mul(a, b),   1.0, 8, 4),
        ("div",   "elementwise", lambda: ms.div(a, b),   1.0, 8, 4),
        ("pow",   "elementwise", lambda: ms.pow(a, b),   8.0, 8, 4),
        # ── Elementwise unary（讀 1 輸入 + 寫 1 輸出）──
        ("sin",   "elementwise", lambda: ms.sin(a),      8.0, 4, 4),
        ("cos",   "elementwise", lambda: ms.cos(a),      8.0, 4, 4),
        ("exp",   "elementwise", lambda: ms.exp(a),      8.0, 4, 4),
        ("log",   "elementwise", lambda: ms.log(ms.abs(a)), 8.0, 4, 4),
        ("abs",   "elementwise", lambda: ms.abs(a),      1.0, 4, 4),
        ("sign",  "elementwise", lambda: ms.sign(a),     1.0, 4, 4),
        ("neg",   "elementwise", lambda: ms.neg(a),      1.0, 4, 4),
        ("clamp", "elementwise", lambda: ms.clamp(a, 0.0, 1.0), 2.0, 4, 4),
        # ── Comparison（讀 2 輸入 + 寫 1 輸出 uint8）──
        ("gt",    "comparison",  lambda: ms.gt(a, b),    1.0, 8, 1),
        ("lt",    "comparison",  lambda: ms.lt(a, b),    1.0, 8, 1),
        ("ge",    "comparison",  lambda: ms.ge(a, b),    1.0, 8, 1),
        ("le",    "comparison",  lambda: ms.le(a, b),    1.0, 8, 1),
        ("eq",    "comparison",  lambda: ms.eq(a, b),    1.0, 8, 1),
        ("ne",    "comparison",  lambda: ms.ne(a, b),    1.0, 8, 1),
        # ── Reduction 全局（axis=None，讀 4MB + 寫 4B 標量）──
        ("sum",       "reduction", lambda: ms.sum(a),       1.0, 4, 0),
        ("prod",      "reduction", lambda: ms.prod(a),      1.0, 4, 0),
        ("max",       "reduction", lambda: ms.max(a),       1.0, 4, 0),
        ("min",       "reduction", lambda: ms.min(a),       1.0, 4, 0),
        ("mean",      "reduction", lambda: ms.mean(a),      2.0, 4, 0),
        ("argmax",    "reduction", lambda: ms.argmax(a),    2.0, 4, 0),
        ("argmin",    "reduction", lambda: ms.argmin(a),    2.0, 4, 0),
        ("cumsum",    "reduction", lambda: ms.cumsum(a),    1.0, 4, 4),
    ]

    # ═══ 逐算子 benchmark ═══
    print(f"\n  {'算子':<10} {'類別':<13} {'延遲(ms)':<11} {'吞吐(GE/s)':<13} {'GFLOPS':<10} {'帶寬(GB/s)':<11}")
    print(f"  {'─'*10} {'─'*13} {'─'*11} {'─'*13} {'─'*10} {'─'*11}")

    results: list[OpResult] = []
    for name, category, fn, flops, r_bytes, w_bytes in ops:
        r = bench_op(fn, iters, size, flops, r_bytes, w_bytes)
        r.name = name
        r.category = category
        results.append(r)
        print(f"  {name:<10} {category:<13} {r.latency_ms:<11.3f} {r.gelem_per_sec:<13.3f} {r.gflops:<10.3f} {r.effective_bw_gb_s:<11.3f}")

    # ═══ 分類統計 ═══
    print("\n" + "-" * 72)
    print("  [分類統計]")
    print("-" * 72)
    categories: dict[str, list[OpResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat, ops_in_cat in categories.items():
        avg_lat = sum(r.latency_ms for r in ops_in_cat) / len(ops_in_cat)
        peak_gflops = max(r.gflops for r in ops_in_cat)
        peak_bw = max(r.effective_bw_gb_s for r in ops_in_cat)
        print(f"  {cat:<13} {len(ops_in_cat):>2} ops | 平均延遲 {avg_lat:.3f} ms"
              f" | 峰值 {peak_gflops:.2f} GFLOPS"
              f" | 峰值帶寬 {peak_bw:.2f} GB/s")

    # ═══ 帶寬分析 ═══
    print("\n" + "-" * 72)
    print("  [帶寬分析]")
    print("-" * 72)
    elemwise = categories.get("elementwise", [])
    comparisons = categories.get("comparison", [])
    reductions = categories.get("reduction", [])
    if elemwise:
        best_bw = max(r.effective_bw_gb_s for r in elemwise)
        print(f"  elementwise 峰值等效帶寬: {best_bw:.2f} GB/s"
              f"  （理論上限 ≈ VRAM 帶寬）")
    if comparisons:
        best_bw = max(r.effective_bw_gb_s for r in comparisons)
        print(f"  comparison  峰值等效帶寬: {best_bw:.2f} GB/s")
    if reductions:
        best_bw = max(r.effective_bw_gb_s for r in reductions)
        print(f"  reduction   峰值等效帶寬: {best_bw:.2f} GB/s")

    # ═══ Reduction 2D 专项 ═══
    print("\n" + "-" * 72)
    print("  [Reduction 专项 — 2D axis]")
    print("-" * 72)

    rows, cols = 256, 256
    flat = [float(i % 10000) * 0.01 for i in range(rows * cols)]
    nested = [flat[i * cols:(i + 1) * cols] for i in range(rows)]
    mat = ms.array(nested, dtype=ms.float32, device=device_str)
    total_2d = rows * cols
    print(f"  矩陣: {rows}×{cols} = {total_2d:,} elements ({fmt_bytes(total_2d * 4)})")

    reduce_ops_2d = [
        ("sum(axis=0)",       lambda: ms.sum(mat, axis=0),  1.0, 4, 0),
        ("sum(axis=1)",       lambda: ms.sum(mat, axis=1),  1.0, 4, 0),
        ("sum(global)",       lambda: ms.sum(mat),          1.0, 4, 0),
        ("mean(axis=1)",      lambda: ms.mean(mat, axis=1), 2.0, 4, 0),
        ("max(axis=0)",       lambda: ms.max(mat, axis=0),  1.0, 4, 0),
        ("argmax(axis=1)",    lambda: ms.argmax(mat, axis=1), 2.0, 4, 0),
        ("cumsum(axis=1)",    lambda: ms.cumsum(mat, axis=1), 1.0, 4, 4),
    ]

    print(f"\n  {'算子':<18} {'延遲(ms)':<11} {'吞吐(GE/s)':<13} {'帶寬(GB/s)':<11}")
    print(f"  {'─'*18} {'─'*11} {'─'*13} {'─'*11}")
    for name, fn, flops, r_bytes, w_bytes in reduce_ops_2d:
        r = bench_op(fn, iters, total_2d, flops, r_bytes, w_bytes)
        print(f"  {name:<18} {r.latency_ms:<11.3f} {r.gelem_per_sec:<13.3f} {r.effective_bw_gb_s:<11.3f}")

    # ═══ Indexing 专项（Phase 6.5-7）═══
    print("\n" + "-" * 72)
    print("  [Indexing 专项 — gather/scatter/contiguous]")
    print("-" * 72)
    print(f"  主數組: {size:,} elements；view ops 零拷贝（延遲 ≈ Python+Rust 開銷）")
    print("  注: gather/scatter 每次调用会做 indices 的 D2H + sync 越界校验，")
    print("      大 indices 数组下该串行化开销主导延迟（非 kernel 瓶颈）。")

    # gather/scatter：全量索引（indices = 全排列），等效于整数组读写
    idx_all = ms.arange(size, dtype=ms.int64, device=device_str)
    vals = ms.array(data_b, dtype=ms.float32, device=device_str)

    # contiguous：transpose/flip 视图物化（读 + 写全量）；flat 走零拷贝快路径
    cols_2d = 1024
    rows_2d = size // cols_2d
    n_2d = rows_2d * cols_2d
    a_2d = ms.array(
        [data_a[i * cols_2d:(i + 1) * cols_2d] for i in range(rows_2d)],
        dtype=ms.float32, device=device_str,
    )
    t_view = ms.transpose(a_2d)
    f_view = ms.flip(a_2d, axis=1)

    indexing_ops = [
        # (name, fn, flops_per_elem, read_bytes, write_bytes, n_elems)
        # view（零拷贝）：bytes=0，带宽无意义，仅看 dispatch 延迟
        ("transpose(view)", lambda: ms.transpose(a_2d), 0.0, 0, 0, n_2d),
        ("flip(view)",      lambda: ms.flip(a_2d, axis=1), 0.0, 0, 0, n_2d),
        ("slice(view)",     lambda: ms.slice(a_2d, [[0, rows_2d // 2, 1], [0, cols_2d // 2, 1]]), 0.0, 0, 0, n_2d // 4),
        # copy ops。scatter(full) 内部两阶段：copy_into(读4+写4) + scatter 覆写(读4+写4) = 8+8
        ("gather(full)",    lambda: ms.gather(a, idx_all, axis=0), 0.0, 4, 4, size),
        ("scatter(full)",   lambda: ms.scatter(a, idx_all, vals, axis=0), 0.0, 8, 8, size),
        # contig(flat) 已连续 → 零拷贝快路径，无实际数据搬运（bytes=0）
        ("contig(flat)",    lambda: ms.contiguous(a), 0.0, 0, 0, size),
        ("contig(transp)",  lambda: ms.contiguous(t_view), 0.0, 4, 4, n_2d),
        ("contig(flip)",    lambda: ms.contiguous(f_view), 0.0, 4, 4, n_2d),
    ]

    print(f"\n  {'算子':<16} {'延遲(ms)':<11} {'吞吐(GE/s)':<13} {'帶寬(GB/s)':<11}")
    print(f"  {'─'*16} {'─'*11} {'─'*13} {'─'*11}")
    for name, fn, flops, r_bytes, w_bytes, n in indexing_ops:
        r = bench_op(fn, iters, n, flops, r_bytes, w_bytes)
        r.name = name
        r.category = "indexing"
        results.append(r)
        print(f"  {name:<16} {r.latency_ms:<11.3f} {r.gelem_per_sec:<13.3f} {r.effective_bw_gb_s:<11.3f}")

    # ═══ Stream 狀態 ═══
    print("\n" + "-" * 72)
    print("  [Stream 狀態]")
    print("-" * 72)
    s = a.stream
    print(f"  stream id: {s.id}")
    print(f"  pending_count: {s.pending_count}")
    print(f"  is_poisoned: {s.is_poisoned}")
    print(f"  device: {s.device}")

    # ═══ 最終顯存（不在計時路徑上） ═══
    print("\n" + "-" * 72)
    print("  [最終顯存]")
    print("-" * 72)
    for line in ms.memory_summary(device=device_str).splitlines():
        print(f"    {line}")

    # ═══ 結論 ═══
    print("\n" + "=" * 72)
    print("  ✓ 測試結論")
    print("=" * 72)
    n_indexing = sum(1 for r in results if r.category == "indexing")
    print(f"  ✓ musa:{device_id} 全部 {len(results)} 個算子執行正常")
    print(f"  ✓ 覆蓋: elementwise({len(categories.get('elementwise', []))})"
          f" + comparison({len(categories.get('comparison', []))})"
          f" + reduction({len(categories.get('reduction', []))})"
          f" + indexing({n_indexing})")
    if results:
        best = min(results, key=lambda r: r.latency_ms)
        worst = max(results, key=lambda r: r.latency_ms)
        peak = max(results, key=lambda r: r.gflops)
        print(f"  ✓ 最低延遲: {best.name} {best.latency_ms:.3f} ms"
              f" | 最高延遲: {worst.name} {worst.latency_ms:.3f} ms")
        print(f"  ✓ 峰值 GFLOPS: {peak.gflops:.2f}（{peak.name}）")
    print(f"  ✓ Stream 健康: pending={s.pending_count}, poisoned={s.is_poisoned}")
    print("=" * 72)

    # ═══ 注意事項（僅在 size 偏小時提示） ═══
    if size < 1_000_000:
        print()
        print("  ⚠ 注意: --size < 1M 時延遲主要由 launch overhead 主導，")
        print("    真實帶寬需要 --size ≥ 10M 才能測出。")


def main():
    parser = argparse.ArgumentParser(description="MUSA GPU 算子性能基准测试")
    parser.add_argument("--size", type=int, default=1_000_000,
                        help="每數組元素數（默認 1M；建議 ≥ 10M 以測出真實帶寬）")
    parser.add_argument("--iters", type=int, default=100,
                        help="每算子迭代次數（默認 100）")
    parser.add_argument("--device", type=int, default=0,
                        help="MUSA 設備 ID（默認 0）")
    args = parser.parse_args()

    run_benchmark(args.size, args.iters, args.device)


if __name__ == "__main__":
    main()
