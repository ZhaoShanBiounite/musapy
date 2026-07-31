"""Comparison ops 测试（Phase 3）

验证 eq/ne/lt/gt/le/ge：
- 基本比较
- 广播
- 类型提升
- dunders
- bool 输出 dtype
- MUSA GPU
"""

import pytest
import musapy as ms


class TestComparison:
    """比较算子基本功能（CPU）"""

    def test_eq(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([1.0, 5.0, 3.0], device="cpu")
        c = ms.eq(a, b)
        assert c.dtype == ms.bool_
        assert c.tolist() == [True, False, True]

    def test_ne(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([1.0, 5.0, 3.0], device="cpu")
        assert ms.ne(a, b).tolist() == [False, True, False]

    def test_lt(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert ms.lt(a, b).tolist() == [True, False, False]

    def test_gt(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert ms.gt(a, b).tolist() == [False, False, True]

    def test_le(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert ms.le(a, b).tolist() == [True, True, False]

    def test_ge(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert ms.ge(a, b).tolist() == [False, True, True]

    def test_output_dtype_bool(self):
        a = ms.array([1.0, 2.0], device="cpu")
        b = ms.array([1.0, 3.0], device="cpu")
        assert ms.eq(a, b).dtype == ms.bool_
        assert ms.lt(a, b).dtype == ms.bool_

    def test_broadcast(self):
        """(3,1) vs (4,) → (3,4) bool"""
        a = ms.array([[1.0], [2.0], [3.0]], device="cpu")
        b = ms.array([1.0, 2.0, 3.0, 4.0], device="cpu")
        c = ms.eq(a, b)
        assert c.shape == (3, 4)
        assert c.tolist()[0] == [True, False, False, False]
        assert c.tolist()[1] == [False, True, False, False]
        assert c.tolist()[2] == [False, False, True, False]

    def test_type_promotion(self):
        """int64 vs float64 → promote then compare"""
        a = ms.array([1, 2, 3], dtype=ms.int64, device="cpu")
        b = ms.array([1.0, 2.5, 3.0], dtype=ms.float64, device="cpu")
        c = ms.eq(a, b)
        assert c.dtype == ms.bool_
        assert c.tolist() == [True, False, True]

    def test_2d(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        b = ms.array([[1.0, 0.0], [3.0, 0.0]], device="cpu")
        c = ms.gt(a, b)
        assert c.shape == (2, 2)
        assert c.tolist() == [[False, True], [False, True]]

    def test_out_param(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        out = ms.array([False, False, False], dtype=ms.bool_, device="cpu")
        result = ms.gt(a, b, out=out)
        assert result.tolist() == [False, False, True]


class TestComparisonDunders:
    """Python 比较运算符（CPU）"""

    def test_eq_dunder(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert (a == b).tolist() == [False, True, False]

    def test_ne_dunder(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert (a != b).tolist() == [True, False, True]

    def test_lt_dunder(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert (a < b).tolist() == [True, False, False]

    def test_gt_dunder(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert (a > b).tolist() == [False, False, True]

    def test_le_dunder(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert (a <= b).tolist() == [True, True, False]

    def test_ge_dunder(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0, 1.0], device="cpu")
        assert (a >= b).tolist() == [False, True, True]

    def test_dunder_broadcast(self):
        a = ms.array([[1.0], [2.0]], device="cpu")
        b = ms.array([1.0, 2.0, 3.0], device="cpu")
        c = (a == b)
        assert c.shape == (2, 3)
        assert c.tolist() == [[True, False, False], [False, True, False]]


class TestComparisonErrors:
    """比较错误情况"""

    def test_incompatible_shapes(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([1.0, 2.0], device="cpu")
        with pytest.raises(ms.ShapeError):
            ms.eq(a, b)


# ── MUSA 硬件测试 ──

def has_musa():
    try:
        ms.set_default_device("musa:0")
        ms.set_default_device("cpu")
        return True
    except Exception:
        return False

musa_required = pytest.mark.skipif(not has_musa(), reason="no MUSA device available")


@musa_required
class TestComparisonMusa:
    """比较算子（MUSA GPU）"""

    def test_basic_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0])
        b = ms.array([3.0, 2.0, 1.0])
        assert ms.gt(a, b).tolist() == [False, False, True]
        assert ms.eq(a, b).tolist() == [False, True, False]
        assert ms.lt(a, b).tolist() == [True, False, False]

    def test_dunders_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0])
        b = ms.array([3.0, 2.0, 1.0])
        assert (a > b).tolist() == [False, False, True]
        assert (a == b).tolist() == [False, True, False]
        assert (a >= b).dtype == ms.bool_

    def test_broadcast_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([[1.0], [2.0], [3.0]])
        b = ms.array([1.0, 2.0, 3.0, 4.0])
        c = ms.lt(a, b)
        assert c.shape == (3, 4)
        assert c.tolist()[0] == [False, True, True, True]

    def test_acceptance(self):
        """Phase 3 验收标准"""
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0])
        b = ms.array([3.0, 2.0, 1.0])
        assert (a > b).tolist() == [False, False, True]
        assert (a == b).tolist() == [False, True, False]
        assert (a >= b).dtype == ms.bool_
