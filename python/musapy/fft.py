"""ms.fft 命名空间（ADR-003 003-D7，v0.3 Phase 5）。

包装 _core 的 FFT 函数，对齐 NumPy 肌肉记忆：

    ms.fft.fft(x)                                    # complex 输入 FFT（axis=-1）
    ms.fft.ifft(x)                                   # 逆变换（backward 缩放 1/N）
    ms.fft.rfft(x)                                   # 实输入 → 输出形状 (..., N//2+1)
    ms.fft.fft(x, n=8, norm="ortho")                 # n 截断/补零 + 正交归一化

本轮范围（用户确认，2026-08-08）：
  - **axis=-1 起步**：只支持沿最后一维；axis != -1 抛 ShapeError
    （fftn/多轴推迟到 v0.3 后期）。
  - `n`：截断（n < 输入长度）/ 补零（n > 输入长度）。
  - `norm`："backward"（默认）/ "ortho" / "forward"（NumPy 语义）。
  - GPU-only（003-D4）：CPU 设备上调用抛 DeviceError。
"""

from typing import Optional

from . import _core
from ._core import Array

__all__ = ["fft", "ifft", "rfft"]


def fft(a: Array, n: Optional[int] = None, axis: int = -1,
        norm: Optional[str] = None, out: Optional[Array] = None) -> Array:
    """复数一维 FFT（沿 axis=-1；输入可 real 或 complex，输出 complex）。"""
    return _core.fft(a, n=n, axis=axis, norm=norm, out=out)


def ifft(a: Array, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None, out: Optional[Array] = None) -> Array:
    """一维逆 FFT（backward 缩放 1/N；输出 complex）。"""
    return _core.ifft(a, n=n, axis=axis, norm=norm, out=out)


def rfft(a: Array, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None, out: Optional[Array] = None) -> Array:
    """实输入一维 FFT，输出形状 (..., N//2+1) complex。"""
    return _core.rfft(a, n=n, axis=axis, norm=norm, out=out)
