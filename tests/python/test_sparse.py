"""v0.3 Phase 6 (P6.6): ms.sparse 命名空间验收（ADR-003 003-D4/D7）。

覆盖：
  - csr_matrix 构造（Array 输入 / list 输入 / 显式 shape）
  - csr @ vec（spmv）/ csr @ dense（spmm）数值对照 NumPy（f32/f64，多 shape）
  - csr @ ndarray / csr @ list（Python 层转换路径）
  - toarray() 物化稠密
  - 空矩阵（nnz=0）、单元素、非方阵
  - 错误路径：shape 不匹配、indices/indptr dtype、dtype 白名单
  - GPU-only：CPU 设备抛 DeviceError

注：scipy 不在 venv，对照一律用 NumPy 稠密手算。
"""

import numpy as np
import pytest

import musapy as ms

# GPU 探测（mock 模式亦有效）
try:
    _dev = ms.Device("musa:0")
    _gpu_available = True
except Exception:
    _gpu_available = False

musa_required = pytest.mark.skipif(not _gpu_available, reason="MUSA device not available")


def _make_csr(dense, dtype, device="musa:0"):
    """从稠密 NumPy 数组构造 CsrMatrix（取非零项）。"""
    rows, cols = dense.shape
    idx = np.argwhere(dense != 0)
    data = dense[idx[:, 0], idx[:, 1]].astype(np.float32 if dtype == ms.float32 else np.float64)
    ind = idx[:, 1].astype(np.int32)
    ptr = np.zeros(rows + 1, dtype=np.int32)
    for r in range(rows):
        ptr[r + 1] = ptr[r] + int(np.sum(idx[:, 0] == r))
    return ms.sparse.csr_matrix(
        (ms.array(data.tolist(), dtype=dtype, device=device),
         ms.array(ind.tolist(), dtype=ms.int32, device=device),
         ms.array(ptr.tolist(), dtype=ms.int32, device=device)),
        shape=(rows, cols),
    )


@musa_required
class TestSparseGpu:
    """真机数值对照 NumPy。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    @pytest.mark.parametrize("dtype", [ms.float32, ms.float64])
    @pytest.mark.parametrize("shape", [(3, 3), (4, 2), (2, 5), (1, 1)])
    def test_spmv_spmm_toarray(self, dtype, shape):
        rng = np.random.default_rng(42)
        rows, cols = shape
        dense = (rng.normal(size=shape) > 0.5).astype(
            np.float32 if dtype == ms.float32 else np.float64
        )
        dense *= rng.normal(size=shape)
        csr = _make_csr(dense, dtype)
        tol = 1e-4 if dtype == ms.float32 else 1e-10

        # toarray
        assert np.allclose(np.array(csr.toarray().tolist()), dense, atol=tol)

        # spmv
        v = rng.normal(size=cols).astype(dense.dtype)
        y = csr @ ms.array(v.tolist(), dtype=dtype)
        assert np.allclose(np.array(y.tolist()), dense @ v, atol=tol)

        # spmm（k=2）
        B = rng.normal(size=(cols, 2)).astype(dense.dtype)
        C = csr @ ms.array(B.tolist(), dtype=dtype)
        assert np.allclose(np.array(C.tolist()), dense @ B, atol=tol)

    def test_simple_3x3(self):
        """手算 3×3 精确对照。"""
        data = ms.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=ms.float64)
        indices = ms.array([0, 2, 1, 0, 2], dtype=ms.int32)
        indptr = ms.array([0, 2, 3, 5], dtype=ms.int32)
        csr = ms.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        assert csr.nnz == 5
        assert csr.shape == (3, 3)
        v = ms.array([1.0, 2.0, 3.0], dtype=ms.float64)
        assert csr @ v == ms.array([7.0, 6.0, 19.0], dtype=ms.float64)
        B = ms.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=ms.float64)
        assert csr @ B == ms.array([[3.0, 2.0], [0.0, 3.0], [9.0, 5.0]], dtype=ms.float64)

    def test_matmul_ndarray_and_list(self):
        """csr @ ndarray / csr @ list（Python 层转换路径）。"""
        csr = _make_csr(np.array([[1.0, 0, 2], [0, 3, 0], [4, 0, 5]]), ms.float64)
        # ndarray f64
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose((csr @ v).tolist(), [7.0, 6.0, 19.0])
        # ndarray 2D
        B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert np.allclose((csr @ B).tolist(), [[3.0, 2.0], [0.0, 3.0], [9.0, 5.0]])
        # list
        assert np.allclose((csr @ [1.0, 2.0, 3.0]).tolist(), [7.0, 6.0, 19.0])

    def test_empty_matrix(self):
        """nnz=0 空矩阵：spmv/toarray 输出全零。"""
        d0 = ms.array([], dtype=ms.float64)
        i0 = ms.array([], dtype=ms.int32)
        p0 = ms.array([0, 0, 0], dtype=ms.int32)
        csr = ms.sparse.csr_matrix((d0, i0, p0), shape=(2, 2))
        assert csr.nnz == 0
        y = csr @ ms.array([1.0, 2.0], dtype=ms.float64)
        assert np.allclose(y.tolist(), [0.0, 0.0])
        assert np.allclose(csr.toarray().tolist(), [[0.0, 0.0], [0.0, 0.0]])

    def test_list_constructor(self):
        """csr_matrix 收 Python list（sparse.py 自动 ms.array；dtype 默认 f32）。"""
        csr = ms.sparse.csr_matrix(
            ([1.0, 2.0], [0, 1], [0, 1, 2]), shape=(2, 2)
        )
        y = csr @ ms.array([1.0, 1.0], dtype=ms.float32)
        assert np.allclose(y.tolist(), [1.0, 2.0])


class TestSparseErrors:
    """错误路径（不要求 GPU gating，GPU 类独立）。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    def test_rhs_ndim_rejected(self):
        csr = _make_csr(np.array([[1.0, 0], [0, 2.0]]), ms.float64)
        bad = ms.array([[1.0, 2.0, 3.0]], dtype=ms.float64)  # 2D 但 shape 错
        with pytest.raises(ms.ShapeError):
            csr @ bad

    def test_rhs_1d_wrong_len(self):
        csr = _make_csr(np.array([[1.0, 0], [0, 2.0]]), ms.float64)
        with pytest.raises(ms.ShapeError):
            csr @ ms.array([1.0, 2.0, 3.0], dtype=ms.float64)

    def test_indices_wrong_dtype(self):
        d = ms.array([1.0], dtype=ms.float64)
        i = ms.array([0], dtype=ms.int64)  # 必须 int32
        p = ms.array([0, 1], dtype=ms.int32)
        with pytest.raises(ms.DtypeError):
            ms.sparse.csr_matrix((d, i, p), shape=(1, 1))

    def test_data_wrong_dtype(self):
        d = ms.array([1.0], dtype=ms.complex128)  # 白名单 f32/f64
        i = ms.array([0], dtype=ms.int32)
        p = ms.array([0, 1], dtype=ms.int32)
        with pytest.raises(ms.DtypeError):
            ms.sparse.csr_matrix((d, i, p), shape=(1, 1))

    def test_spmv_dtype_mismatch(self):
        csr = _make_csr(np.array([[1.0, 0], [0, 2.0]]), ms.float64)
        with pytest.raises(ms.DtypeError):
            csr @ ms.array([1.0, 2.0], dtype=ms.float32)


class TestSparseCpuRejected:
    """GPU-only（003-D4）：CPU 设备抛 DeviceError。"""

    def test_cpu_construction_rejected(self):
        d = ms.array([1.0], dtype=ms.float64, device="cpu")
        i = ms.array([0], dtype=ms.int32, device="cpu")
        p = ms.array([0, 1], dtype=ms.int32, device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.sparse.csr_matrix((d, i, p), shape=(1, 1))

    def test_cpu_spmv_rejected(self):
        # musa 构造的 csr 不能与 cpu vec 运算（device mismatch → DeviceError）
        ms.set_default_device("musa:0")
        csr = _make_csr(np.array([[1.0, 0], [0, 2.0]]), ms.float64)
        v = ms.array([1.0, 2.0], dtype=ms.float64, device="cpu")
        with pytest.raises(ms.DeviceError):
            csr @ v
