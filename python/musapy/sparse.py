"""ms.sparse 命名空间（ADR-003 003-D7，v0.3 Phase 6）。

包装 _core 的 sparse 类与函数，对齐 scipy.sparse 肌肉记忆：

    csr = ms.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
    y = csr @ vec          # spmv（vec 为 ms.Array 或 numpy.ndarray/list）
    C = csr @ dense        # spmm（dense 为 ms.Array 或 numpy.ndarray/list）
    A = csr.toarray()      # 物化稠密 ms.Array

本轮范围（用户确认，2026-08-08）：
  - 只做 `csr_matrix`（data/indices/indptr 3 个 device buffer）；`coo_matrix`
    与 coo→csr 归并推迟到 v0.3 后期。
  - `@` 右侧：ms.Array（device 直连）或 numpy.ndarray/list
    （Python 层先 ms.array() 转 device，再走 device 路径）。
  - data dtype f32/f64；indices/indptr 须 int32（musparse INDEX_32I）。
  - GPU-only（003-D4）：CPU 设备上调用抛 DeviceError。
"""

from typing import Optional, Tuple


from . import _core
from ._core import Array, CsrMatrix

__all__ = ["csr_matrix", "CsrMatrix", "spmv", "spmm"]


def csr_matrix(
    arg,
    shape: Optional[Tuple[int, int]] = None,
    dtype: Optional[object] = None,
    device: Optional[str] = None,
) -> CsrMatrix:
    """构造 CSR 稀疏矩阵。

    输入两种形态（对齐 scipy.sparse.csr_matrix）：
      - `(data, indices, indptr)`：三个 1D 序列（data f32/f64，indices/indptr int32）。
        非 Array 输入自动 ms.array() 上传到 device。
      - 也可直接传三个 ms.Array。

    `shape=(rows, cols)` 必须提供（nnz>0 时无法从 device 数据推断 cols）。
    """
    if isinstance(arg, (tuple, list)) and len(arg) == 3:
        data, indices, indptr = arg
    else:
        raise TypeError(
            "csr_matrix: expected (data, indices, indptr) tuple (got "
            f"{type(arg).__name__})"
        )

    # 非 Array → ms.array 上传（dtype 默认推断；indices/indptr 显式 int32）
    if not isinstance(data, Array):
        data = _core.array(data, dtype=dtype, device=device)
    if not isinstance(indices, Array):
        indices = _core.array(indices, dtype=_core.int32, device=device)
    if not isinstance(indptr, Array):
        indptr = _core.array(indptr, dtype=_core.int32, device=device)

    return _core.csr_matrix(data, indices, indptr, shape=shape, dtype=dtype)


def spmv(mat: CsrMatrix, vec: Array) -> Array:
    """csr @ vec（1D 向量）→ 1D Array。"""
    return _core.spmv(mat, vec)


def spmm(mat: CsrMatrix, dense: Array) -> Array:
    """csr @ dense（2D 矩阵）→ 2D Array。"""
    return _core.spmm(mat, dense)
