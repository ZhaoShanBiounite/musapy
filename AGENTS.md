# Project Conventions

## Python Environment

- Use `uv` for package management (installed at /root/miniconda3/bin/uv)
- Use the project virtual environment at `.venv/` (Python 3.10, created by uv)
- Activate before running Python commands: `source .venv/bin/activate`
- Install/editable: `uv pip install -e .` or `maturin develop` within the venv
- Run tests: `source .venv/bin/activate && pytest tests/python/ -v`
