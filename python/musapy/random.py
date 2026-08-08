"""ms.random 命名空间（ADR-003 003-D7，v0.3 Phase 4）。

包装 _core 的 random 生成函数，对齐 NumPy 肌肉记忆：

    ms.random.rand(2, 3)                                   # uniform [0,1) f32
    ms.random.randn(2, 3, dtype='f64', seed=1)             # N(0,1)
    ms.random.uniform(-1.0, 1.0, shape=(4, 4), seed=7)     # [-1, 1)
    ms.random.normal(loc=0.0, scale=2.0, shape=(2, 2))     # N(0, 4)
    ms.random.bernoulli(p=0.3, shape=(2, 2))               # bool

语义约定：
  - seed：给定 seed → 每次调用前重置生成器（同 seed 紧邻两次逐元素可复现）；
    无 seed → 不重置（连续调用产生不同序列）。
  - shape=None（uniform/normal/bernoulli）→ 返回 0-dim 标量数组（NumPy 对齐）。
  - rand/randn 支持 rand(2, 3) 与 rand((2, 3)) 两种形态。
  - GPU-only（003-D4）：CPU 设备上调用抛 DeviceError。
"""

from typing import Sequence

from . import _core
from ._core import Array

__all__ = ["rand", "randn", "uniform", "normal", "bernoulli"]


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """*shape 归一化：rand(2, 3) 与 rand((2, 3)) 两种形态都接受。"""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def rand(*shape, dtype=None, device=None, seed=None) -> Array:
    """uniform [0,1) 随机数（dtype 默认 float32）。

    rand(2, 3) / rand((2, 3)) 两种形态等价；rand() 返回 0-dim 标量数组。
    """
    return _core.rand(_normalize_shape(shape), dtype=dtype, device=device, seed=seed)


def randn(*shape, dtype=None, device=None, seed=None) -> Array:
    """标准正态 N(0,1) 随机数（dtype 默认 float32）。"""
    return _core.randn(_normalize_shape(shape), dtype=dtype, device=device, seed=seed)


def uniform(low=0.0, high=1.0, shape=None, dtype=None, device=None, seed=None) -> Array:
    """[low, high) 均匀分布。shape=None → 0-dim 标量数组。"""
    return _core.uniform(low, high, shape, dtype=dtype, device=device, seed=seed)


def normal(loc=0.0, scale=1.0, shape=None, dtype=None, device=None, seed=None) -> Array:
    """N(loc, scale²) 正态分布（生成器原生 mean/stddev，一步生成）。"""
    return _core.normal(loc, scale, shape, dtype=dtype, device=device, seed=seed)


def bernoulli(p=0.5, shape=None, device=None, seed=None) -> Array:
    """Bernoulli(p) → bool 数组（rand < p）。shape=None → 0-dim 标量数组。"""
    return _core.bernoulli(p, shape, device=device, seed=seed)
