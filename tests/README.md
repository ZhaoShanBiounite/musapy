# tests/

## Structure
```
tests/
├── rust/           # Rust 集成测试（独立文件，非 #[cfg(test)] 内联）
│   └── musapy-core/
│       ├── test_device.rs
│       ├── test_dtype.rs
│       ├── test_stream.rs
│       └── test_buffer.rs
└── python/         # Python 集成测试
    ├── conftest.py
    ├── test_array.py        # ms.array() 创建/属性/repr/naming/dtype 常量
    └── test_resolution.py   # 5 级解析/DeviceNotConfigured/context manager/异常层次
```

Rust 单元测试内联在各模块的 `#[cfg(test)] mod tests` 中，通过 `cargo test` 运行。

## Running

```bash
# Rust tests
MUSA_INSTALL_PATH=/usr/local/musa cargo test

# Python tests (after `maturin develop` or `pip install -e .`)
pytest tests/python/
```
