# Benchmark 后续分析（2026-08-04）

依据 `benchmark/analysis-2026-08-03.md` 的 P0→P1→P2→P4→P3→P5 顺序，
在 `perf/p0-p5` 分支（基于 `phase6-indexing`）逐阶段实施并独立提交。
全部数字在 MTT S4000（mp_22, 56 CUs, 47.9 GB）实机测得，pytest 406 +
cargo test 293 全绿。

## 各阶段前后对比

| 阶段 | 算子 | 之前 | 之后 | 加速 | commit |
|------|------|------|------|------|--------|
| P0 | cumsum axis_len > 65536 | 结果错误 + smem 越界 | 分层扫描，容量 256³/轴，结果正确 | 修复 | `14bd55b` |
| P1 | gather(full) 1M | ~10 ms（每次调用同步 + 8MB D2H + host 校验） | **0.178 ms** | ~56× | `f59c56a` |
| P1 | scatter(full) 1M | ~10 ms | **0.268 ms** | ~37× | `f59c56a` |
| P2 | 全局 sum/prod/max/min/mean 1M | 0.129–0.131 ms | **0.084–0.086 ms** | ~1.5× | `8d6c7f1` |
| P2 | 2D 256×256 axis 缩减 | 0.122–0.158 ms | **0.053–0.056 ms** | ~2.4× | `8d6c7f1` |
| P4 | contig(transp) 1M | 0.240 ms（33 GB/s） | **0.063 ms（126 GB/s）** | 3.8× | `015ad7b` |
| P4 | contig(flip) 1M | 0.241 ms | **0.104 ms（77 GB/s）** | 2.3× | `015ad7b` |
| P3 | add 16M | 0.326 ms（589 GB/s） | **0.310 ms（620 GB/s）** | +5.2% | `6553d44` |
| P5 | log 基准（双 kernel） | 0.119 ms | **0.054 ms** | 2.2×（方法学） | 本阶段 |

## 关键发现

### 1. launch 地板 ≈ 45 µs（P3 扫描坐实）

`--size` 1M/4M/16M/64M 扫描（add/abs）：
- 16M→64M 增量斜率 ≈ **683 GB/s ≈ DRAM 峰值（768 GB/s 的 89%）**；
- 外推到 n=0 的截距 ≈ **45 µs 固定开销**（kernel launch + sync 往返，
  driver 提交路径，应用层无法消除）。
- 1M 规模下所有算子延迟 ≈ 45 µs 地板 + kernel 执行时间——1M 的
  "带宽"读数（elementwise ~200 GB/s、transpose 126 GB/s）均为地板
  抑制的伪影，真实带宽需 ≥4M 规模测量。

### 2. mp_22 的 64 位整数除法是软件模拟（P1 根因）

f32 vs f64 gather（+50% 流量）耗时几乎不变 → 计算受限而非内存受限；
2D gather 慢 3× → 每元素两次 64 位 div/mod 是元凶。修复：ndim==1
快路径 + 总数 ≤ 2³² 时用 32 位 div/mod（gather/scatter/copy 均已覆盖）。

### 3. mcc 编译器特性（实测）

- `__device__ static inline` 函数**含 extern __shared__ + __shfl 时不内联**，
  函数调用路径实测 75× 变慢 → partial kernel 的公共框架必须宏内联
  （P2，probe：inline 0.053ms vs fn 4.0ms）。
- float4 显式向量化本身可用且正确（P3 probe），但与 shuffle 组合在
  同一函数时触发病态代码生成（P2 早期误判为 float4 问题，实为
  内联问题）。
- atomicOr / atomicCAS（global int）在 mp_22 可用（P1 Step 0 验证）。

### 4. 各 kernel 当前带宽上限（≥4M 规模，去除地板）

| kernel | 带宽 | 备注 |
|--------|------|------|
| elementwise（标量 + float4） | 620–655 GB/s | 内存受限，float4 +5% |
| 转置 tiled | 289–322 GB/s | 32×32 tile + smem padding |
| flip（通用 copy + u32 路径） | ~127 GB/s | 仍受 div/mod + 标量访问限制 |
| gather/scatter | 45/60 GB/s（1M） | 1M 读数受地板限制；kernel 本体 ~320 GB/s |

## 残留项与远期方向

1. **flip 专用 kernel**：通用 copy 对最后维 stride = ±1 的视图可每线程
   4 元素（div/mod 摊薄 4×），预计 flip 从 ~127 → 200+ GB/s。
2. **MUSA Graphs**（graph capture）：把多次 launch 合并提交，压低
   ~45 µs launch 地板——对 1M 规模的小算子收益最大，属于应用层
   唯一可动的地板。
3. **f64 向量化**：double2（16B）同理，elementwise/partial 均可推广。
4. **argmax/argmin 小 axis 路径**：P2 只覆盖了 sum/prod/max/min/mean，
   arg* 在 axis_len ≤ 1024 仍走 naive（2D 256×256 下 0.138ms）。
5. **reduction final kernel**：1M 全局归约的 final 阶段 1 block 串行
   扫 ~977 个 partials，可改为 warp-shuffle 多 block 两级。

## 语义变更记录（文档已同步）

- gather/scatter GPU 越界：kernel 内检查 + device 错误槽，异常延迟到
  下一次流同步抛出 `ShapeError`，流不毒化（P1，`operators-reference.md`
  已新增 Indexing 算子节）。
- cumsum 容量上限：256³ ≈ 16.7M 元素/轴，超限报错（P0）。
- 小 axis 归约路径：naive 仅剩 axis_len ≤ 16 与 arg*（P2）。

## 复现

```bash
source .venv/bin/activate
maturin develop --release
pytest tests/python/ && cargo test
python benchmark/bench_musa_utilization.py --size 1000000 --iters 100
# 带宽 / 地板扫描
python benchmark/bench_musa_utilization.py --size 4000000 --iters 50
```
