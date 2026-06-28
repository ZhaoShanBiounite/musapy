# tests/

## Structure
tests/
├── rust/ # Rust unit/integration tests
│ └── musapy-core/
│ ├── test_device.rs
│ ├── test_dtype.rs
│ ├── test_stream.rs
│ └── test_buffer.rs
└── python/ # Python integration tests
└── test_add.py
## Running

```bash
# Rust tests
cargo test

# Python tests (after `pip install -e .`)
pytest tests/python/
