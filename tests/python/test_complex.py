"""v0.3 Phase 5 (P5.1): complex 落地回归（ADR-003 003-D5）。

覆盖：
  - ms.array complex 创建：字面量推断（complex128）、显式 dtype、mixed 列表、0-dim
  - tolist/item complex 读回
  - elementwise：add/sub/mul/div/neg/abs（对照 NumPy；abs 输出 real）
  - comparison：eq/ne 支持 complex；lt/gt/le/ge 拒绝（DtypeError）
  - 类型提升：real + complex → complex（f32+c64→c64、f64+c64→c128 等）
  - 混合列表（complex + real）推断
  - 视图（slice）上的 complex 运算

CPU 设备上 complex 运算走既有 CPU fallback（v0.2 算子，非数学库），无 GPU 亦可跑。
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


class TestComplexArray:
    """ms.array complex 创建与读回。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    def test_literal_infers_c128(self):
        a = ms.array([1 + 2j, 3 - 4j])
        assert a.dtype == 'c128'
        assert a.tolist() == [1 + 2j, 3 - 4j]

    def test_explicit_c64(self):
        a = ms.array([1.0, 2.0, 3.0], dtype='c64')
        assert a.dtype == 'c64'
        assert a.tolist() == [1 + 0j, 2 + 0j, 3 + 0j]

    def test_mixed_list(self):
        a = ms.array([1 + 1j, 2.0, 3j])
        assert a.dtype == 'c128'
        assert a.tolist() == [1 + 1j, 2 + 0j, 3j]

    def test_scalar_0dim(self):
        s = ms.array(2.5 + 0.5j)
        assert s.shape == ()
        assert s.dtype == 'c128'
        assert s.item() == 2.5 + 0.5j

    def test_2d_nested(self):
        m = ms.array([[1 + 1j, 2], [3, 4 - 1j]])
        assert m.shape == (2, 2)
        assert m.tolist() == [[1 + 1j, 2 + 0j], [3 + 0j, 4 - 1j]]

    def test_explicit_dtype_overrides_inference(self):
        a = ms.array([1 + 2j, 3 + 4j], dtype='c64')
        assert a.dtype == 'c64'


class TestComplexElementwise:
    """complex elementwise 数值对照 NumPy（CPU 路径可跑，GPU 亦覆盖）。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    @pytest.mark.parametrize("dtype", ['c64', 'c128'])
    def test_binary_ops(self, dtype):
        x = ms.array([1 + 2j, 3 + 4j, -1 + 0.5j], dtype=dtype)
        y = ms.array([1 - 1j, 0 + 2j, 2 - 2j], dtype=dtype)
        xn, yn = np.array(x.tolist()), np.array(y.tolist())
        for name, got, exp in [
            ("add", ms.add(x, y).tolist(), xn + yn),
            ("sub", ms.sub(x, y).tolist(), xn - yn),
            ("mul", ms.mul(x, y).tolist(), xn * yn),
            ("div", ms.div(x, y).tolist(), xn / yn),
        ]:
            got = np.array(got, dtype=complex)
            tol = 1e-5 if dtype == 'c64' else 1e-10
            assert np.allclose(got, exp, rtol=tol, atol=tol), (name, got, exp)

    @pytest.mark.parametrize("dtype", ['c64', 'c128'])
    def test_unary_neg_abs(self, dtype):
        x = ms.array([1 + 2j, 3 - 4j, -0.5 + 0.25j], dtype=dtype)
        xn = np.array(x.tolist())
        assert np.allclose(
            np.array(ms.neg(x).tolist()), -xn, atol=1e-5 if dtype == 'c64' else 1e-10
        )
        # abs 输出 real（NumPy 语义）
        ab = ms.abs(x)
        real_dtype = 'f32' if dtype == 'c64' else 'f64'
        assert ab.dtype == real_dtype
        assert np.allclose(
            np.array(ab.tolist()), np.abs(xn), atol=1e-5 if dtype == 'c64' else 1e-10
        )

    @pytest.mark.parametrize("dtype", ['c64', 'c128'])
    def test_eq_ne(self, dtype):
        x = ms.array([1 + 2j, 3 + 4j, 5 + 0j], dtype=dtype)
        y = ms.array([1 + 2j, 3 - 4j, 5 + 0j], dtype=dtype)
        eq = ms.eq(x, y)
        assert eq.dtype == 'b1'
        assert eq.tolist() == [True, False, True]
        ne = ms.ne(x, y)
        assert ne.tolist() == [False, True, False]

    @pytest.mark.parametrize(
        "fn", [ms.lt, ms.gt, ms.le, ms.ge]
    )
    def test_ordering_rejected(self, fn):
        x = ms.array([1 + 2j, 3 + 4j], dtype='c128')
        y = ms.array([1 + 1j, 2 + 2j], dtype='c128')
        with pytest.raises(ms.DtypeError):
            fn(x, y)

    def test_pow_rejected(self):
        x = ms.array([1 + 2j, 3 + 4j], dtype='c128')
        with pytest.raises(ms.DtypeError):
            ms.pow(x, x)


class TestComplexPromotion:
    """real + complex 类型提升（跨类别 → 宽 complex）。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    def test_f64_add_c64(self):
        a = ms.array([1.0, 2.0], dtype='f64')
        b = ms.array([1 + 1j, 2 - 2j], dtype='c64')
        r = ms.add(a, b)
        assert r.dtype == 'c128'
        assert np.allclose(np.array(r.tolist()), [2 + 1j, 4 - 2j])

    def test_f32_add_c64(self):
        a = ms.array([1.0, 2.0], dtype='f32')
        b = ms.array([1 + 1j, 2 - 2j], dtype='c64')
        r = ms.add(a, b)
        assert r.dtype == 'c64'

    def test_f32_add_c128(self):
        a = ms.array([1.0, 2.0], dtype='f32')
        b = ms.array([1 + 1j, 2 - 2j], dtype='c128')
        r = ms.add(a, b)
        assert r.dtype == 'c128'

    def test_complex_add_real_broadcast(self):
        r = ms.add(ms.array([1 + 1j, 2 + 2j]), ms.array([1.0, 2.0]))
        assert r.dtype == 'c128'
        assert r.tolist() == [2 + 1j, 4 + 2j]

    def test_c64_add_c64_narrow(self):
        a = ms.array([1 + 1j], dtype='c64')
        b = ms.array([2 + 2j], dtype='c64')
        assert ms.add(a, b).dtype == 'c64'


class TestComplexViews:
    """视图（slice）上的 complex 运算。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    def test_slice_view_ops(self):
        arr = ms.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        view = arr[:, 0]
        assert view.tolist() == [1 + 1j, 3 + 3j]
        assert ms.neg(view).tolist() == [-1 - 1j, -3 - 3j]
        assert np.allclose(
            np.array(ms.abs(view).tolist()),
            np.abs(np.array(view.tolist())),
            atol=1e-10,
        )
