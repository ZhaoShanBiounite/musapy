# musapy 快速上手

## 前置条件

| 依赖 | 最低版本 | 说明 |
|---|---|---|
| Python | >= 3.9 | 推荐 3.10+ |
| Rust | stable (edition 2024) | 用于编译核心运行时 |
| MUSA SDK | 3.x+ | 摩尔线程 GPU 计算栈（`MUSA_HOME` 环境变量） |
| maturin | >= 1.5 | Rust → Python 扩展构建工具 |

> **注意**：无 MUSA GPU 时仍可在 CPU 模式下使用 musapy 进行开发和测试。

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
# v0.1.0-alpha
```

## 快速上手

```python
import musapy as ms

# 设置默认设备（必须显式设置，否则首次创建 Array 会抛 DeviceNotConfiguredError）
ms.set_default_device("musa:0")  # 有 GPU 时
# ms.set_default_device("cpu")   # 无 GPU 时用 CPU

# 创建 Array
a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32)
b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32)

# 逐元素加法
c = ms.add(a, b)       # Array([5.0, 7.0, 9.0], device=musa:0)
c = a + b              # 等价写法

# Stream context
s = ms.Stream("musa:0")
with ms.stream(s):
    d = ms.add(a, b)   # 绑定到 stream s

# 同步 + 回读
s.synchronize()
print(c.tolist())      # [5.0, 7.0, 9.0]

# Resolution feedback（追溯 device 来源）
print(a.device)
# Device(musa:0)  # resolved from: global_default (musa:0)

# 内存汇总
print(ms.memory_summary())
```

## 核心概念

### Device

设备标识，支持字符串和对象两种形式：

```python
ms.array([1, 2, 3], device="musa:0")           # 字符串
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

未指定 dtype 时默认 `float32`。

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
a.name     # None（可选命名）
```

## 运行测试

```bash
# 运行全部 Python 测试
pytest tests/python/ -v

# 仅运行 CPU 测试（跳过 MUSA 硬件测试）
pytest tests/python/ -v -k "not Musa"

# Rust 单元测试
cargo test
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MUSA_HOME` | `/usr/local/musa` | MUSA SDK 安装路径 |
| `MUSAPY_DEBUG` | `0` | 设为 `1` 启用 debug 模式 |

## 下一步

- 查看 [ADR 文档](./ADR-zh.md) 了解完整架构决策
- 查看 [v0.1-alpha 实现计划](./v0.1-alpha-plan-zh.md) 了解版本范围
