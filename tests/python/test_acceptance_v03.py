"""v0.3-alpha 端到端验收（P9.2，docs/v0.3-alpha-plan-zh.md §1.3 成功定义）。

覆盖 v0.3 全部数学库 + 补全能力端到端跑通：
  linalg（matmul/dot/solve/qr/svd/lu）+ random 全套 + fft 全套
  + sparse（csr_matrix/@/toarray）+ reduction 补全（axis=tuple/复数 sum）
  + 高级索引（mask/fancy）+ Stream.synchronize。
真机 MUSA 执行；mock 构建下亦应通过（GPU stub）。
"""

import numpy as np
import pytest

import musapy as ms

try:
    _dev = ms.Device("musa:0")
    _gpu_available = True
except Exception:
    _gpu_available = False

musa_required = pytest.mark.skipif(not _gpu_available, reason="MUSA device not available")


@musa_required
class TestAcceptanceV03:
    """§1.3 成功定义端到端。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    def test_linalg_end_to_end(self):
        """matmul/dot/solve/qr/svd/lu 端到端（对照 NumPy）。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float64)
        b = ms.array([[5.0, 6.0], [7.0, 8.0]], dtype=ms.float64)
        c = ms.matmul(a, b)
        assert np.allclose(c.tolist(), np.array([[1.,2.],[3.,4.]]) @ np.array([[5.,6.],[7.,8.]]), atol=1e-8)
        x = ms.solve(a, ms.array([1.0, 2.0], dtype=ms.float64))
        assert np.allclose(x.tolist(), np.linalg.solve([[1.,2.],[3.,4.]], [1.,2.]), atol=1e-8)
        q, r = ms.qr(a)
        assert np.allclose(np.array(q.tolist()) @ np.array(r.tolist()),
                           np.array(a.tolist()), atol=1e-8)
        u, s, vh = ms.svd(a)
        assert np.allclose(np.array(u.tolist()) @ np.diag(s.tolist()) @ np.array(vh.tolist()),
                           np.array(a.tolist()), atol=1e-8)
        lu, piv = ms.lu(a)
        assert lu.shape == (2, 2) and piv.shape == (2,)

    def test_random_end_to_end(self):
        """random 全套：rand/randn/uniform/normal/bernoulli。"""
        r1 = ms.random.rand((2, 3))
        assert r1.shape == (2, 3)
        r1a = np.array(r1.tolist())
        assert 0 <= r1a.min() < 1 and 0 < r1a.max() <= 1
        r2 = ms.random.randn((2, 3), seed=1)
        r2b = ms.random.randn((2, 3), seed=1)
        assert r2.tolist() == r2b.tolist()  # seed 复现
        r3 = ms.random.uniform(-1.0, 1.0, shape=(4,))
        assert r3.shape == (4,) and all(-1 <= v <= 1 for v in r3.tolist())
        r4 = ms.random.bernoulli(0.5, (4,))
        assert r4.dtype == ms.bool_

    def test_fft_end_to_end(self):
        """fft/ifft/rfft 端到端。"""
        f = ms.fft.fft(ms.array([1.0, 2.0, 3.0, 4.0], dtype=ms.float64))
        assert f.dtype == ms.complex128
        g = ms.fft.ifft(f)
        assert np.allclose(g.tolist(), [1, 2, 3, 4], atol=1e-8)
        rf = ms.fft.rfft(ms.array([1.0, 2.0, 3.0, 4.0], dtype=ms.float64))
        assert rf.shape == (3,)  # 4//2+1

    def test_sparse_end_to_end(self):
        """sparse：csr_matrix/@/toarray。"""
        csr = ms.sparse.csr_matrix(
            (ms.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=ms.float64),
             ms.array([0, 2, 1, 0, 2], dtype=ms.int32),
             ms.array([0, 2, 3, 5], dtype=ms.int32)),
            shape=(3, 3),
        )
        y = csr @ ms.array([1.0, 2.0, 3.0], dtype=ms.float64)
        assert np.allclose(y.tolist(), [7.0, 6.0, 19.0])
        A = csr.toarray()
        assert np.allclose(A.tolist(), [[1.0, 0, 2], [0, 3, 0], [4, 0, 5]])

    def test_reduction_completion(self):
        """reduction 补全：axis=tuple + 复数 sum。"""
        a = ms.array(np.arange(24.0).reshape(2, 3, 4).tolist(), dtype=ms.float64)
        s2 = ms.sum(a, axis=(0, 1))
        assert np.allclose(s2.tolist(), np.arange(24.0).reshape(2,3,4).sum(axis=(0,1)))
        s3 = ms.sum(a, axis=(0,), keepdims=True)
        assert s3.shape == (1, 3, 4)
        sc = ms.sum(ms.array([1 + 2j, 3 + 4j], dtype=ms.complex64))
        assert abs(sc.item() - (4 + 6j)) < 1e-5

    def test_advanced_indexing(self):
        """高级索引：mask + fancy。"""
        a = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ms.float64)
        # mask 索引（等形 mask 需显式构造；a > 2.0 标量比较暂不支持下轮）
        m_full = ms.array([[True, False, True], [False, True, False]], dtype=ms.bool_)
        sel = a[m_full]
        assert sel.tolist() == [1.0, 3.0, 5.0]
        m = ms.array([True, False], dtype=ms.bool_)
        assert a[m].tolist() == [[1.0, 2.0, 3.0]]
        fancy = a[ms.array([0, 1], dtype=ms.int64)]
        assert fancy.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        # 越界 IndexError
        with pytest.raises(IndexError):
            a[ms.array([5], dtype=ms.int64)]

    def test_stream_synchronize(self):
        """Stream.synchronize 显式同步。"""
        s = ms.Stream("musa:0")
        s.synchronize()
        assert s.pending_count == 0
