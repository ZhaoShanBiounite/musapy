# musapy

Python + Rust + MUSA 科学计算库 —— 为摩尔线程 GPU 提供 NumPy 风格的数组计算 API。

## 特性

- **NumPy 风格 API**：`ms.array()`、`ms.add()`、`a + b`，零学习成本
- **显式设备管理**：5 级 Device 解析链，数据去向可追溯
- **Rust 核心**：内存安全、零开销抽象、RAII 生命周期
- **Stream 异步**：多 stream 并行执行，自动依赖管理
- **15 种 dtype**：bool/int/uint/float/bfloat16/complex 全覆盖
- **Debug 模式**：运行时 flag 启用 OpContext 归因、alias 检测

## 架构

```
musapy/
├── python/musapy/          # Python 前端（import musapy as ms）
├── rust/
│   ├── musapy-core/        # 核心运行时：Device/Dtype/Stream/Buffer/Array、内存管理
│   ├── musapy-ops/         # 算子层：OpBuilder、kernel 调度
│   └── musapy-python/      # PyO3 绑定：Python ↔ Rust FFI
└── kernels/                # MUSA C kernel（.mu 文件）
```

## 安装

```bash
# 前置：Rust toolchain + MUSA SDK（MUSA_HOME 环境变量）
pip install -e .

# 验证
python -c "import musapy; print(musapy.__version__)"
```

详细安装指南见 [docs/getting-started.md](docs/getting-started.md)。

## 最小示例

```python
import musapy as ms

ms.set_default_device("musa:0")

a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32)
b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32)
c = a + b

print(c.tolist())  # [5.0, 7.0, 9.0]
```

## 开发

```bash
# 构建
maturin develop --release

# 测试
pytest tests/python/ -v
cargo test

# Lint
cargo clippy
cargo fmt
```

## 路线图

| 版本 | 范围 | 状态 |
|---|---|---|
| v0.1-alpha | 核心运行时（Device/Dtype/Stream/Array/Buffer） | 当前 |
| v0.2-alpha | 基础 ops（elementwise/reduction/init/indexing） | 规划中 |
| v0.3-alpha | 数学库（linalg/random/fft/sparse） | 规划中 |
| v0.4-beta | 互操作（DLPack/NumPy 协议）+ 完整错误模型 | 规划中 |
| v1.0 | 正式版 | 规划中 |

## 文档

- [快速上手](docs/getting-started.md)
- [架构决策记录（ADR）](docs/ADR-zh.md)
- [v0.1-alpha 实现计划](docs/v0.1-alpha-plan-zh.md)

## License

MIT License — see [LICENSE](LICENSE) for details.
