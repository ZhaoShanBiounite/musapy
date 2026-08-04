# musapy 快速上手

> 版本：v0.2-alpha（`phase6-indexing` 开发主干）
> 完整算子参考见 [operators-reference.md](./operators-reference.md)

## 前置条件

| 依赖 | 最低版本 | 说明 |
|---|---|---|
| Python | >= 3.9 | 推荐 3.10+ |
| Rust | stable (edition 2024) | 用于编译核心运行时 |
| MUSA SDK | 3.x+ | 摩尔线程 GPU 计算栈（`MUSA_HOME` 环境变量） |
| maturin | >= 1.5 | Rust → Python 扩展构建工具 |

> **注意**：无 MUSA GPU 时仍可在 CPU 模式下使用 musapy 进行开发和测试
> （`MUSAPY_MOCK_MUSA=1` 构建，全部算子有 CPU fallback）。

## 安装

### 开发模式（推荐）

```bash
# 克隆仓库
git clone https://github.com/ZhaoShanBiounite/musapy.git
cd musapy

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装（自动编译 Rust 扩展）
pip install -e .
```

### 使用 maturin 直接构建

```bash
# 构建并安装到当前环境
maturin develop --release

# 或构建 wheel
maturin build --release
pip install target/wheels/musapy-*.whl
```

### 验证安装

```bash
python -c "import musapy; print(musapy.__version__)"
# v0.2.0-alpha
```

## 快速上手

以下代码在 MUSA GPU（或 CPU 模式）上端到端跑通（v0.2 功能面）：

```python
import musapy as ms

# 设置默认设备（必须显式设置，否则首次创建 Array 抛 DeviceNotConfiguredError）
ms.set_default_device("musa:0")  # 有 GPU 时
# ms.set_default_device("cpu")   # 无 GPU 时用 CPU

# ── 创建（init 套件）──
a = ms.array([[1.0], [2.0], [3.0]])                    # (3,1) float32
b = ms.array([10, 20, 30, 40], dtype=ms.int64)         # (4,) int64
z = ms.zeros((2, 3))                                   # float32（默认 dtype）
r = ms.arange(5)                                       # int64（NumPy 推断）
lin = ms.linspace(0.0, 1.0, 5)                         # float64
eye = ms.eye(3)

# ── 广播 + 类型提升 ──
c = a + b                # 广播 (3,1)+(4,) → (3,4)；int64+float32 提升为 float64
assert c.shape == (3, 4) # 提升规则见 L1-14（int64/uint64 + float → f64）

# ── elementwise ──
# 二元/一元操作数必须是 Array（标量请用 ms.array([x]) 包一层）；
# clamp 的 lo/hi 是 Python 标量
d = ms.sin(c) * ms.exp(ms.array(0.1))
f = ms.clamp(d, 0.0, 1.0)
g = c.astype(ms.float32)   # 显式转换

# ── comparison（输出 bool 数组）──
mask = c > ms.array(15.0)            # (3,4) bool
assert mask.dtype == ms.bool_

# ── reduction（axis/keepdims/argmax/cumsum）──
s = ms.sum(c)                        # 0-dim 标量，全 reduce
row = ms.sum(c, axis=1)              # (3,)
col = ms.sum(c, axis=0, keepdims=True)  # (1,4)
m = ms.mean(c)
am = ms.argmax(c, axis=1)            # (3,) int64
cs = ms.cumsum(c, axis=1)

# ── indexing ──
t = ms.transpose(c)                  # view (4,3)，零拷贝共享 buffer
sl = c[0:2, ::2]                     # view (2,2)
fl = ms.flip(c, axis=1)              # view
# gather/scatter 的 indices 必须是 int64 1D（整数列表请显式指定 dtype）
gg = ms.gather(c, ms.array([0, 2], dtype=ms.int64), axis=1)  # copy (3,2)
ct = ms.contiguous(t)                # 物化为连续布局（copy）

# ── 同步 + 回读 ──
c.stream.synchronize()               # 等所有异步 op 完成
print(row.tolist())                  # [104.0, 108.0, 112.0]
```

## 算子速览

| 分类 | 算子 |
|---|---|
| Binary elementwise | `add` `sub` `mul` `div` `pow`（+ 运算符重载 `a + b` 等） |
| Unary elementwise | `sin` `cos` `exp` `log` `abs` `sign` `neg`（+ `-a` / `abs(a)`） |
| 边界 | `clamp(a, lo, hi)` |
| Comparison | `gt` `lt` `ge` `le` `eq` `ne`（输出 bool） |
| Reduction | `sum` `prod` `max` `min` `mean`（`axis=`/`keepdims=`） |
| Arg-reduction | `argmax` `argmin`（输出 int64 索引） |
| Scan | `cumsum`（单轴容量上限 256³ ≈ 16.7M 元素） |
| Cast | `a.astype(dtype)` |
| Init | `zeros` `ones` `full` `arange` `linspace` `eye` `zeros_like` `ones_like` |
| Indexing | view：`transpose` `permute` `flip` `slice`（`a[i:j:k]`）；copy：`gather` `scatter` `contiguous` |

## 语义注意（与 NumPy 的差异）

1. **GPU 越界错误延迟报错**（P1 方案二）：`gather`/`scatter` 的索引越界
   在 GPU 路径下不立即抛出，而是延迟到下一次流同步（`tolist()`/`item()`
   内部会同步）抛 `ShapeError`；流不毒化，越界条目被跳过，其余结果有效。
2. **标量不自动广播**：二元/比较算子的两个操作数都必须是 Array。
   `ms.add(a, 2.0)` 会抛 `TypeError`——请用 `ms.add(a, ms.array([2.0]))`。
   `clamp` 是例外（lo/hi 为 Python 标量）。
3. **类型提升**：`int64/uint64 + float → float64`（JAX 风格 type-based，
   见 ADR L1-14），不是 NumPy 的"窄优先"。

## 核心概念

### Device

设备标识，支持字符串和对象两种形式：

```python
ms.array([1, 2, 3], device="musa:0")             # 字符串
ms.array([1, 2, 3], device=ms.Device("musa:0"))  # 对象
```

Device 解析遵循 5 级优先级链：

| 优先级 | 来源 | 示例 |
|---|---|---|
| 1 | 函数参数 `device=` | `ms.array(..., device="musa:0")` |
| 2 | context manager | `with ms.device("musa:0"):` |
| 3 | 输入 Array 的 device | `a + b` 跟 a 走 |
| 4 | 全局默认 | `ms.set_default_device("musa:0")` |
| 5 | 启动 auto-probe | 有 MUSA 用 musa:0，否则 cpu |

### Dtype

支持 15 种数据类型：

```
bool, int8, int16, int32, int64,
uint8, uint16, uint32, uint64,
float16, float32, float64, bfloat16,
complex64, complex128
```

未指定 dtype 时默认 `float32`（整数 `arange` 等按 NumPy 规则推断）。

### Stream

每个设备有独立的 stream，用于异步执行：

```python
s = ms.Stream("musa:0", priority=0)
with ms.stream(s):
    a = ms.array([1.0, 2.0])
    b = ms.add(a, a)
s.synchronize()  # 等待所有操作完成
```

### Array

核心数据结构，绑定到特定设备和 stream：

```python
a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="musa:0")
a.shape    # (3,)
a.ndim     # 1
a.dtype    # Dtype(float32)
a.device   # Device(musa:0)
a.stream   # Stream(...)
a.tolist() # 同步 + 回读（GPU 越界错误在此抛出）
```

## 运行测试

```bash
# 运行全部 Python 测试（含 MUSA 硬件 gated 用例）
pytest tests/python/ -v

# 仅运行 CPU 测试（跳过 MUSA 硬件测试）
pytest tests/python/ -v -k "not Musa"

# Rust 单元测试（mock 模式，无 GPU 可跑）
cargo test

# GPU 性能基准（MTT S4000 实测约 45 µs launch 地板，1M 规模数字受地板主导）
python benchmark/bench_musa_utilization.py --size 1000000 --iters 100
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MUSA_HOME` | `/usr/local/musa` | MUSA SDK 安装路径 |
| `MUSAPY_DEBUG` | `0` | 设为 `1` 启用 debug 模式 |

## 下一步

- [operators-reference.md](./operators-reference.md) — 已实现算子参考（kernel 符号、ABI、性能）
- [ADR-zh.md](./ADR-zh.md) — 完整架构决策（L0-L4 分层）
- [v0.2-alpha 实现计划](./v0.2-alpha-plan-zh.md) — 版本范围与阶段
- [benchmark/analysis-followup-2026-08-04.md](../benchmark/analysis-followup-2026-08-04.md) — 性能优化前后对比与 launch 地板分析
