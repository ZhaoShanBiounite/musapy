# musapy 快速上手

> 版本：v0.2-alpha（`phase6-indexing` 开发主干）
> 完整算子参考（kernel 符号 / ABI / 性能）见 [operators-reference.md](./operators-reference.md)

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

以下代码在 MUSA GPU（或 CPU 模式）上端到端跑通（v0.2 全功能面）：

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
c = a + b                # 广播 (3,1)+(4,) → (3,4)
assert c.shape == (3, 4)
assert c.dtype == ms.float32   # i64 + f32 → f32（整数不因位宽升级浮点）

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

## 完整 API 参考（v0.2 全部 Python 接口）

### 1. 创建与转换

| 函数 | 签名 | 说明 |
|---|---|---|
| `array` | `array(data, dtype=None, device=None)` | 从 Python 嵌套列表创建 |
| `astype`（方法） | `a.astype(dtype)` | 显式 dtype 转换（返回新 Array） |

### 2. Elementwise

| 函数 | 签名 | 说明 |
|---|---|---|
| `add / sub / mul / div / pow` | `(a, b, out=None)` | 二元，自动广播 + 提升 |
| `sin / cos / exp / log / abs / sign / neg` | `(a, out=None)` | 一元 |
| `clamp` | `(a, lo, hi, out=None)` | lo/hi 为 Python 标量 |

运算符重载：`a + b` `a - b` `a * b` `a / b` `a ** b` `-a` `abs(a)`。

### 3. Comparison（输出 bool 数组）

| 函数 | 签名 |
|---|---|
| `eq / ne / lt / gt / le / ge` | `(a, b, out=None)` |

运算符重载：`a == b` `a != b` `a < b` `a > b` `a <= b` `a >= b`。

### 4. Reduction

| 函数 | 签名 | 说明 |
|---|---|---|
| `sum / prod / max / min / mean` | `(a, axis=None, keepdims=False, out=None)` | `axis=None` 全缩减 |
| `argmax / argmin` | `(a, axis=None, out=None)` | 输出恒 int64 索引 |
| `cumsum` | `(a, axis=None, out=None)` | 单轴容量上限 256³ ≈ 16.7M 元素 |

### 5. Init / Creation

| 函数 | 签名 | 说明 |
|---|---|---|
| `zeros / ones` | `(shape, *, dtype=None, device=None)` | 默认 float32 |
| `full` | `(shape, fill_value, *, dtype=None, device=None)` | 填充值 |
| `eye` | `(n, m=None, k=0, *, dtype=None, device=None)` | 单位阵 |
| `arange` | `(start, stop=None, step=None, *, dtype=None, device=None)` | 整数→int64，浮点→float64 |
| `linspace` | `(start, stop, num=50, *, dtype=None, device=None)` | 默认 float64 |
| `zeros_like / ones_like` | `(a)` | 继承输入 dtype/device |

### 6. Indexing

| 函数 | 签名 | 说明 |
|---|---|---|
| `transpose` | `(a, axes=None)` | view，零拷贝 |
| `permute` | `(a, dims)` | view，等价 transpose(axes=dims) |
| `flip` | `(a, axis)` | view，stride 取负 |
| `slice` | `(a, specs)` | view；`a[i:j:k]` 切片语法同样可用（含负索引/step/clamp） |
| `index_select` | `(a, axis, index)` | view（按 axis 取 index 处子集） |
| `gather` | `(a, indices, axis=0)` | copy；**indices 必须 int64 1D** |
| `scatter` | `(a, indices, values, axis=0)` | copy，返回新 Array；indices 须 int64 1D |
| `contiguous` | `(a)` | 物化为连续布局；已连续时零拷贝 |

### 7. 运行时与上下文

| 函数 | 说明 |
|---|---|
| `set_default_device(dev)` | 全局默认设备（必须显式设置一次） |
| `set_default_dtype(dt)` | 全局默认 dtype |
| `device(dev)` / `dtype(dt)` / `stream(s)` | context manager（with 块内生效，thread-local） |
| `memory_summary(device=None)` | 内存统计（allocated/cached/peak/VRAM） |
| `device_summary()` | 设备名、arch、VRAM、CU 数 |
| `set_debug(True)` / `debug()` | OpContext 记录（python_frame 归因） |
| `startup_report()` | 运行时启动报告 |
| `__version__` | 版本号 |

### 8. Array 属性与方法

```python
a = ms.array([[1.0, 2.0], [3.0, 4.0]])

# 属性（只读）
a.shape            # (2, 2)
a.ndim             # 2
a.size             # 4
a.dtype            # Dtype(float32)
a.device           # Device(musa:0)
a.stream           # Stream(...)
a.nbytes           # 16
a.is_contiguous    # True
a.name             # None（可选命名）

# 方法
a.set_name("w")    # 命名（OpContext 归因用）
a.clear_name()
a.astype(ms.float64)
a.tolist()         # 同步 + 回读为嵌套列表（GPU 越界错误在此抛出）
a.item()           # 0-dim / 单元素 → Python 标量
```

### 9. Dtype 常量

```python
ms.bool_            # bool（注意不是 ms.bool）
ms.int8  ms.int16  ms.int32  ms.int64
ms.uint8 ms.uint16 ms.uint32 ms.uint64
ms.float16 ms.float32 ms.float64
ms.bfloat16
ms.complex64 ms.complex128
```

未指定 dtype 时默认 `float32`（`arange` 等按 NumPy 规则推断）。

### 10. 异常层级

```
MusapyError
├── DeviceError ────────── DeviceNotConfiguredError
├── DtypeError
├── ShapeError
├── MemoryError ────────── OutOfMemoryError
├── StreamError
├── KernelError
└── InteropError
```

## 语义注意（与 NumPy 的差异）

1. **GPU 越界错误延迟报错**（P1 方案二）：`gather`/`scatter` 的索引越界
   在 GPU 路径下不立即抛出，而是延迟到下一次流同步（`tolist()`/`item()`
   内部会同步）抛 `ShapeError`；流不毒化，越界条目被跳过，其余结果有效。
2. **标量不自动广播**：二元/比较算子的两个操作数都必须是 Array。
   `ms.add(a, 2.0)` 会抛 `TypeError`——请用 `ms.add(a, ms.array([2.0]))`。
   `clamp` 是例外（lo/hi 为 Python 标量）。
3. **类型提升（2026-08 对齐 v0.2 计划 §1.3 与 ADR L1-14）**：
   - `int/uint（任意位宽）+ float → float 本身`——整数不因位宽升级浮点：
     `i64 + f32 → f32`（注意 int64 会被 cast 成 f32，大整数有精度损失）、
     `i64 + f64 → f64`
   - GPU 全运算窄优先：`f32 + f64 → f32`（性能优先）；含 CPU 时走 JAX 表取宽
   - 纯整数：CPU 取宽 / GPU 取窄；`int + uint` 溢出保护升级

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

### Stream

每个设备有独立的 stream，用于异步执行：

```python
s = ms.Stream("musa:0", priority=0)
with ms.stream(s):
    a = ms.array([1.0, 2.0])
    b = ms.add(a, a)
s.synchronize()  # 等待所有操作完成
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
- [ADR-zh.md](./ADR-zh.md) — 完整架构决策（L0-L4 分层；类型提升见 L1-14）
- [v0.2-alpha 实现计划](./v0.2-alpha-plan-zh.md) — 版本范围与阶段
- [benchmark/analysis-followup-2026-08-04.md](../benchmark/analysis-followup-2026-08-04.md) — 性能优化前后对比与 launch 地板分析
