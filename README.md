# musapy

**Python + Rust + MUSA 科学计算库** —— 为摩尔线程（Moore Threads）GPU 提供 NumPy 风格的数组计算 API。

```python
import musapy as ms

ms.set_default_device("musa:0")

a = ms.array([[1.0], [2.0], [3.0]])       # (3, 1)
b = ms.array([10.0, 20.0, 30.0, 40.0])    # (4,)
c = a + b                                   # 广播 → (3, 4)
print(c.tolist())
# [[11.0, 21.0, 31.0, 41.0],
#  [12.0, 22.0, 32.0, 42.0],
#  [13.0, 23.0, 33.0, 43.0]]
```

---

## 特性

| 能力 | 说明 |
|------|------|
| NumPy 风格 API | `ms.array()`, `ms.add()`, `a + b`, `a ** b`, `abs(a)` — 零学习成本 |
| NumPy 广播 | 任意 N 维自动广播，stride-aware kernel 零拷贝 |
| 类型提升 | `int64 + float64 → float64`，自动 cast，无需手动转换 |
| 14 个 elementwise 算子 | binary / unary / clamp / astype，全部支持 GPU + CPU |
| 6 个 comparison 算子 | `== / != / < / > / <= / >=`，输出 bool |
| 8 个 reduction 算子 | sum / prod / max / min / mean + argmax / argmin / cumsum（axis=int/tuple，v0.3） |
| 8 个 init 算子 | zeros / ones / full / eye / arange / linspace / `*_like` |
| 7 个 indexing 算子 | transpose / permute / flip / slice（零拷贝 view）+ gather / scatter（copy） |
| 高级索引（v0.3） | `a[mask]` boolean + `a[idx]` fancy（恒 copy，越界抛 IndexError） |
| linalg（v0.3，GPU-only） | matmul / dot / solve / lu / qr / svd |
| random（v0.3，GPU-only） | rand / randn / uniform / normal / bernoulli（seed 可复现） |
| fft（v0.3，GPU-only） | fft / ifft / rfft（含 2D batched） |
| sparse（v0.3，GPU-only） | csr_matrix + `@` spmv / spmm / toarray |
| 复数支持（v0.3） | elementwise / reduction（sum/mean/prod）/ cast（real→c64/c128）；`array([1+2j])` 推断 c128 |
| 字符串 dtype 语法（v0.3） | `dtype='f32'` / `'i64'` / `'c64'` / `'b1'` 等短别名，`a.dtype == 'f32'` |
| 显式设备管理 | 5 级 Device 解析链，数据去向可追溯 |
| Stream 异步 | 多 stream 并行，自动依赖管理，capture-safe 3-phase 骨架 |
| Rust 核心 | 内存安全、零开销抽象、RAII 生命周期 |
| 15 种 dtype | bool / int8-64 / uint8-64 / float16-64 / bfloat16 / complex64-128 |
| Debug 模式 | OpContext 归因、alias 检测、Python 帧捕获 |
| Mock 模式 | `MUSAPY_MOCK_MUSA=1` 无 GPU 开发 / CI |

---

## 架构

```
musapy/
├── python/musapy/              # Python 前端（import musapy as ms）
│   ├── __init__.py             #   公开 API 导出
│   ├── _core.pyi               #   类型 stub（IDE 补全）
│   ├── random.py / fft.py      #   ms.random / ms.fft 命名空间包装
│   └── sparse.py               #   ms.sparse 命名空间包装
├── rust/
│   ├── musapy-core/            # 核心运行时
│   │   ├── device.rs           #   Device 解析链（5 级）
│   │   ├── dtype.rs            #   Dtype + 类型提升
│   │   ├── stream.rs           #   Stream + OpContext
│   │   ├── buffer.rs           #   Buffer（GPU/CPU 内存）
│   │   ├── array.rs            #   Array（Buffer + Layout + metadata）
│   │   ├── layout.rs           #   Layout + broadcast
│   │   ├── math_handle.rs      #   MUSA-X 句柄/plan/workspace 生命周期
│   │   └── musa_ffi.rs         #   MUSA Runtime FFI 绑定
│   ├── musapy-ops/             # 算子层
│   │   ├── op_builder.rs       #   3-phase 骨架（parse → launch → post）
│   │   ├── elementwise.rs      #   elementwise + cast
│   │   ├── reduction.rs        #   reduction / argreduce / cumsum
│   │   ├── linalg.rs           #   matmul / solve / lu / qr / svd
│   │   ├── random.rs           #   muRAND 生成
│   │   ├── fft.rs              #   muFFT 变换
│   │   ├── sparse.rs           #   muSPARSE csr_matrix / spmv / spmm
│   │   ├── indexing.rs         #   view / gather / scatter / 高级索引
│   │   ├── creation.rs         #   zeros / ones / full / eye / arange ...
│   │   ├── broadcast.rs        #   NumPy 广播规则
│   │   └── kernels.rs          #   extern "C" 声明 + mock stub
│   └── musapy-python/          # PyO3 绑定
│       ├── ops.rs              #   #[pyfunction] 模块级函数
│       ├── array.rs            #   Array 方法 + dunders
│       └── lib.rs              #   模块注册
├── kernels/                    # MUSA C kernel（mcc 编译）
│   ├── elementwise.mu          #   binary/unary/clamp/cast kernel
│   ├── reduction.mu            #   归约/arg/scan kernel
│   ├── indexing.mu             #   索引 kernel（adv_gather/nonzero 预留）
│   ├── init.mu                 #   fill/arange/linspace/eye kernel
│   └── include/common.h        #   offset_nd + grid 工具函数
├── benchmark/                  # GPU 计算占用验证脚本
├── tests/python/               # pytest 测试套件
└── docs/                       # ADR + 设计文档
```

**数据流：**

```
Python (ms.add)
  → PyO3 (ops.rs)
    → musapy-ops (elementwise.rs → op_builder.rs)
      → broadcast + type promotion
        → kernels.rs (extern "C")
          → elementwise.mu (MUSA GPU kernel)
```

---

## 安装

### 前置要求

- Python ≥ 3.9
- Rust toolchain（[rustup](https://rustup.rs)）
- MUSA SDK ≥ 3.1（`MUSA_INSTALL_PATH` 或默认 `/usr/local/musa`）
- [maturin](https://github.com/PyO3/maturin) ≥ 1.5

### 构建

```bash
git clone git@github.com:ZhaoShanBiounite/musapy.git
cd musapy

# 构建并安装（editable）
maturin develop --release

# 验证
python -c "import musapy as ms; print(ms.__version__)"
```

### 无 GPU 开发（Mock 模式）

```bash
MUSAPY_MOCK_MUSA=1 maturin develop
MUSAPY_MOCK_MUSA=1 cargo test
```

详细安装指南见 [docs/getting-started.md](docs/getting-started.md)。

---

## API 参考

### 数组创建

```python
ms.array(data, dtype=None, device=None) -> Array
```

### Binary 算子（支持广播 + 类型提升）

```python
ms.add(a, b, out=None)    # a + b
ms.sub(a, b, out=None)    # a - b
ms.mul(a, b, out=None)    # a * b
ms.div(a, b, out=None)    # a / b
ms.pow(a, b, out=None)    # a ** b
```

### Unary 算子（stride-aware）

```python
ms.sin(a, out=None)       # sin(x)
ms.cos(a, out=None)       # cos(x)
ms.exp(a, out=None)       # e^x
ms.log(a, out=None)       # ln(x)
ms.abs(a, out=None)       # |x|
ms.sign(a, out=None)      # sign(x) → {-1, 0, 1}
ms.neg(a, out=None)       # -x
```

### 其他算子

```python
ms.clamp(a, lo, hi, out=None)   # min(max(x, lo), hi)
a.astype(dtype)                  # 显式 dtype 转换（接受 'f32' 字符串或 Dtype）
```

### v0.3 数学库算子（GPU-only，详见 [operators-reference.md](docs/operators-reference.md)）

```python
# linalg（muBLAS + muSOLVER）
ms.matmul(a, b) / ms.dot(a, b) / ms.solve(A, b) / ms.lu(A) / ms.qr(A) / ms.svd(A)

# random（muRAND，seed 可复现）
ms.random.rand(shape, dtype='f32', seed=None)   # uniform [0,1)
ms.random.randn(shape, dtype='f64', seed=1)     # N(0,1)
ms.random.uniform(low, high, shape=...) / normal(loc, scale, shape=...) / bernoulli(p, shape=...)

# fft（muFFT，axis=-1；2D+ batched）
ms.fft.fft(x) / ms.fft.ifft(x) / ms.fft.rfft(x)

# sparse（muSPARSE）
csr = ms.sparse.csr_matrix((data, indices, indptr), shape=(rows, cols))
y = csr @ vec                    # spmv
C = csr @ dense                  # spmm
A = csr.toarray()
```

### v0.3 语义扩展

```python
# 复数（elementwise / reduction / cast；array([1+2j]) 推断 c128）
ms.array([1+2j, 3-4j], dtype='c64')
ms.sum(ms.array([1+2j], dtype='c64'))          # sum/mean/prod 支持复数
ms.max(x)                                     # 复数 → DtypeError（无全序）

# reduction 多轴 + arg* keepdims
ms.sum(a, axis=(0, 1))  /  ms.argmax(a, axis=1, keepdims=True)

# 高级索引（恒 copy；越界抛 IndexError）
a[mask]         # boolean mask（等形或前 md 维广播）
a[[0, 2]]       # fancy（坐标配对/索引形状广播/N-D/负索引）
```

### Python 运算符

```python
a + b       # __add__
a - b       # __sub__
a * b       # __mul__
a / b       # __truediv__
a ** b      # __pow__
-a          # __neg__
abs(a)      # __abs__
```

### 设备与监控

```python
ms.set_default_device("musa:0")
ms.set_default_dtype(ms.float64)
ms.device_summary()                    # 设备名称、arch、VRAM、CU 数
ms.memory_summary(device="musa:0")     # allocated / cached / peak / VRAM
```

### 类型提升规则

| 输入 A | 输入 B | 结果 |
|--------|--------|------|
| float32 | float32 | float32 |
| float64 | float64 | float64 |
| float32 | float64 | float64 |
| int32 | float32 | float32 |
| int64 | float32 | float32（整数不因位宽升级浮点，JAX 语义） |
| int64 | float64 | float64 |
| int32 | int64 | int64（CPU JAX 表）/ int32（GPU 窄优先） |

> 完整规则见 [operators-reference.md](docs/operators-reference.md) 类型提升一节与 ADR L1-14。

---

## 示例

### 广播 + 类型提升

```python
import musapy as ms
ms.set_default_device("musa:0")

# 广播：(3,1) + (4,) → (3,4)
a = ms.array([[1.0], [2.0], [3.0]])
b = ms.array([10.0, 20.0, 30.0, 40.0])
print((a + b).shape)  # (3, 4)

# 类型提升：int64 + float64 → float64
i = ms.array([1, 2, 3], dtype='i64')
f = ms.array([0.1, 0.2, 0.3], dtype='f64')
c = ms.add(i, f)
print(c.dtype)    # float64
print(c.tolist()) # [1.1, 2.2, 3.3]
```

> dtype 支持字符串语法：`'f32'`/`'f64'`/`'i64'` 等短别名或 `'float32'`/`'float64'` 全名（也兼容 `ms.float32` 常量）。

### Unary + Clamp

```python
a = ms.array([-2.0, 0.0, 3.0])

print(ms.abs(a).tolist())              # [2.0, 0.0, 3.0]
print(ms.neg(a).tolist())              # [2.0, 0.0, -3.0]
print(ms.clamp(a, 0.0, 1.0).tolist()) # [0.0, 0.0, 1.0]
print(ms.exp(ms.array([0.0, 1.0])).tolist())  # [1.0, 2.718...]
```

### GPU 监控

```python
ms.set_default_device("musa:0")
a = ms.array([1.0, 2.0, 3.0])
_ = ms.sin(a)

print(ms.device_summary())
# cpu — host memory
# musa:0 — MTT S4000, arch=mp_22, 47.9 GB VRAM, 56 CUs

print(ms.memory_summary(device="musa:0"))
# musapy memory summary
#   Allocated: 7.6 MB (2 buffers)
#   Peak allocated: 11.4 MB
#   Device musa:0 — 15.3 MB used / 49062 MB total VRAM (49046.7 MB free)
```

---

## 开发

```bash
# 构建
maturin develop --release

# Rust 测试（mock 模式，无需 GPU）
MUSAPY_MOCK_MUSA=1 cargo test

# Python 测试（需要 GPU）
pytest tests/python/ -v

# GPU 计算占用验证
python benchmark/bench_musa_utilization.py --size 1000000 --iters 100

# Lint
cargo clippy -- -D warnings
cargo fmt --check
```

---

## Benchmark（MTT S4000）

```
设备: MTT S4000, arch=mp_22, 47.9 GB VRAM, 56 CUs
权威数据: repo.md（2026-08-08 全量报告，含 v0.3 数学库算子）

关键数字（2026-08-08 复测）:
elementwise 峰值带宽（64M）      696 GB/s（≈ DRAM 峰值 90%）
reduction 峰值带宽（64M）        219 GB/s
comparison 峰值带宽（64M）       414 GB/s
f32 matmul 峰值                  13.9 TFLOPS（n=2048）
fft 2D 64×4096                   0.262 ms（batched，P-FFT-1 24.5×）
spmv 2000² d=0.01                0.061 ms（描述符缓存，P-A3 11×）
复数 sum（c64 1M）               0.11 ms（分量并行，~1900×）

复现: python benchmark/bench_linalg.py / bench_musa_utilization.py ...
完整数据: repo.md、benchmark/analysis-*.md
```
> 下方为 2026-08-04 历史快照（P0–P5 优化后基线），权威数据以上方 repo.md 为准。

```
规模: 1,000,000 elements × f32（2026-08-04，P0–P5 优化后）

类别            延迟(ms)      备注
elementwise     ~0.054-0.066  受 ~45 µs launch 地板限制（driver 提交路径）
reduction 全局  0.084-0.086   sum 基准
reduction 2D    0.053-0.056   小 axis 并行路径
gather/scatter  0.178 / 0.240 去同步校验后（~50× 提升）
contig(transp)  0.063         tiled kernel

大数组带宽（≥4M 才反映真实带宽）:
add 16M → 620 GB/s（≈ DRAM 峰值 89%），64M → 655 GB/s
转置 4M/16M → 221 / 289 GB/s

复现: python benchmark/bench_musa_utilization.py --size 1000000 --iters 100
详细分析: benchmark/analysis-2026-08-03.md、benchmark/analysis-followup-2026-08-04.md
```

---

## 路线图

| 版本 | 范围 | 状态 |
|------|------|------|
| v0.1-alpha | 核心运行时（Device / Dtype / Stream / Array / Buffer） | ✅ 完成（v0.1.0-alpha，2026-07-28） |
| v0.2-alpha P1 | Stride-aware ABI + NumPy 广播 | ✅ 完成 |
| v0.2-alpha P2 | Elementwise 全家桶 + 类型提升 + astype | ✅ 完成 |
| v0.2-alpha P3 | Reduction（sum / prod / max / min / mean / argmax / argmin / cumsum） | ✅ 完成 |
| v0.2-alpha P4 | Init（zeros / ones / full / eye / arange / linspace / `*_like`） | ✅ 完成 |
| v0.2-alpha P5 | Indexing（transpose / permute / flip / slice 零拷贝 view + gather / scatter copy） | ✅ 完成（v0.2.0-alpha，2026-08-04） |
| v0.3-alpha | 数学库（linalg / random / fft / sparse）+ axis=tuple / 复数归约 / 高级索引 + 字符串 dtype 语法 | ✅ 完成（v0.3.0-alpha，2026-08-08） |
| v0.4-beta | 互操作（DLPack / NumPy 协议） | 规划中 |
| v1.0 | 正式版 | 规划中 |

---

## 文档

- [快速上手](docs/getting-started.md)
- [已实现算子参考](docs/operators-reference.md)
- [架构决策记录（ADR）](docs/adr/ADR-zh.md)
- [v0.3.0-alpha 发布说明](docs/release/v0.3-alpha-release-note.md)
- [SDK 3.1.0 已知限制](docs/sdk-3.1.0-limitations.md)
- [Benchmark 数据报告](repo.md)
- [v0.2 实现计划](docs/archive/v0.2-alpha-plan-zh.md)
- [v0.1 发布说明](docs/release/v0.1-alpha-release-note.md)

---

## License

[MIT](LICENSE)
