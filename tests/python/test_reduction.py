"""Phase 4: Reduction 套件测试（sum/prod/max/min/mean/argmax/argmin/cumsum）。

CPU 测试使用 MUSAPY_MOCK_MUSA=1 构建；GPU 测试需真实 MUSA 设备。
"""

import math
import pytest
import musapy as ms

# GPU 测试标记
musa_required = pytest.mark.skipif(
    not hasattr(ms, "_has_musa") or not ms._has_musa(),
    reason="MUSA device not available",
) if hasattr(ms, "_has_musa") else pytest.mark.skipif(True, reason="no _has_musa")

# 尝试检测 GPU 可用性
try:
    _dev = ms.Device("musa:0")
    _gpu_available = True
except Exception:
    _gpu_available = False

musa_required = pytest.mark.skipif(not _gpu_available, reason="MUSA device not available")


# ═══════════════════════════════════════════════════════════════
# TestReductionBasic — 基本全局/axis/keepdims 测试（CPU）
# ═══════════════════════════════════════════════════════════════

class TestReductionBasic:
    """基本 reduction 功能（全局、axis、keepdims）。"""

    def test_sum_global(self):
        a = ms.array([1.0, 2.0, 3.0, 4.0])
        result = ms.sum(a)
        assert result.item() == pytest.approx(10.0)

    def test_sum_axis0(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.sum(a, axis=0)
        assert result.tolist() == [pytest.approx(4.0), pytest.approx(6.0)]

    def test_sum_axis1(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.sum(a, axis=1)
        assert result.tolist() == [pytest.approx(3.0), pytest.approx(7.0)]

    def test_sum_keepdims(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.sum(a, axis=0, keepdims=True)
        assert result.shape == (1, 2)
        assert result.tolist() == [[pytest.approx(4.0), pytest.approx(6.0)]]

    def test_sum_negative_axis(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.sum(a, axis=-1)
        assert result.tolist() == [pytest.approx(3.0), pytest.approx(7.0)]

    def test_prod_global(self):
        a = ms.array([1.0, 2.0, 3.0, 4.0])
        result = ms.prod(a)
        assert result.item() == pytest.approx(24.0)

    def test_prod_axis(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.prod(a, axis=1)
        assert result.tolist() == [pytest.approx(2.0), pytest.approx(12.0)]

    def test_max_global(self):
        a = ms.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = ms.max(a)
        assert result.item() == pytest.approx(5.0)

    def test_max_axis(self):
        a = ms.array([[1.0, 5.0], [3.0, 2.0]])
        result = ms.max(a, axis=0)
        assert result.tolist() == [pytest.approx(3.0), pytest.approx(5.0)]

    def test_min_global(self):
        a = ms.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = ms.min(a)
        assert result.item() == pytest.approx(1.0)

    def test_min_axis(self):
        a = ms.array([[1.0, 5.0], [3.0, 2.0]])
        result = ms.min(a, axis=1)
        assert result.tolist() == [pytest.approx(1.0), pytest.approx(2.0)]

    def test_mean_global(self):
        a = ms.array([1.0, 2.0, 3.0, 4.0])
        result = ms.mean(a)
        assert result.item() == pytest.approx(2.5)

    def test_mean_axis(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.mean(a, axis=1)
        assert result.tolist() == [pytest.approx(1.5), pytest.approx(3.5)]

    def test_mean_keepdims(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.mean(a, axis=0, keepdims=True)
        assert result.shape == (1, 2)
        assert result.tolist() == [[pytest.approx(2.0), pytest.approx(3.0)]]

    def test_sum_0d_scalar(self):
        """全局缩减产生 0-dim scalar。"""
        a = ms.array([5.0])
        result = ms.sum(a)
        assert result.shape == ()
        assert result.item() == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════
# TestReductionDtype — 类型提升/累加规则
# ═══════════════════════════════════════════════════════════════

class TestReductionDtype:
    """ADR-002-D3 累加 dtype 规则。"""

    def test_sum_int_accumulates_i64(self):
        """整数输入 sum → int64 输出。"""
        a = ms.array([1, 2, 3], dtype=ms.int64)
        result = ms.sum(a)
        assert result.dtype == ms.int64
        assert result.item() == 6

    def test_sum_int8_accumulates_i64(self):
        """int8 输入 sum → int64 输出（cast + 累加）。"""
        a = ms.array([1, 2, 3], dtype=ms.int8)
        result = ms.sum(a)
        assert result.dtype == ms.int64
        assert result.item() == 6

    def test_mean_int_gives_float64(self):
        """整数输入 mean → float64 输出。"""
        a = ms.array([1, 2, 3, 4], dtype=ms.int64)
        result = ms.mean(a)
        assert result.dtype == ms.float64
        assert result.item() == pytest.approx(2.5)

    def test_mean_f32_stays_f32(self):
        """float32 输入 mean → float32 输出。"""
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32)
        result = ms.mean(a)
        assert result.dtype == ms.float32

    def test_max_int_gives_i64(self):
        """整数输入 max → int64 输出（alpha 简化）。"""
        a = ms.array([3, 1, 4], dtype=ms.int64)
        result = ms.max(a)
        assert result.dtype == ms.int64
        assert result.item() == 4

    def test_sum_f32_stays_f32(self):
        """float32 输入 sum → float32 输出。"""
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32)
        result = ms.sum(a)
        assert result.dtype == ms.float32


# ═══════════════════════════════════════════════════════════════
# TestArgReduce — argmax / argmin
# ═══════════════════════════════════════════════════════════════

class TestArgReduce:
    """argmax / argmin 测试。"""

    def test_argmax_global(self):
        a = ms.array([1.0, 5.0, 3.0, 2.0])
        result = ms.argmax(a)
        assert result.item() == 1
        assert result.dtype == ms.int64

    def test_argmin_global(self):
        a = ms.array([3.0, 1.0, 4.0, 0.5])
        result = ms.argmin(a)
        assert result.item() == 3

    def test_argmax_axis(self):
        a = ms.array([[1.0, 5.0], [3.0, 2.0]])
        result = ms.argmax(a, axis=1)
        assert result.tolist() == [1, 0]

    def test_argmin_axis(self):
        a = ms.array([[1.0, 5.0], [3.0, 2.0]])
        result = ms.argmin(a, axis=0)
        assert result.tolist() == [0, 1]

    def test_argmax_output_shape(self):
        a = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ms.argmax(a, axis=1)
        assert result.shape == (2,)

    def test_argmax_int_input(self):
        a = ms.array([10, 30, 20], dtype=ms.int64)
        result = ms.argmax(a)
        assert result.item() == 1


# ═══════════════════════════════════════════════════════════════
# TestCumsum — 累积求和
# ═══════════════════════════════════════════════════════════════

class TestCumsum:
    """cumsum 测试。"""

    def test_cumsum_1d(self):
        a = ms.array([1.0, 2.0, 3.0, 4.0])
        result = ms.cumsum(a, axis=0)
        assert result.tolist() == [
            pytest.approx(1.0),
            pytest.approx(3.0),
            pytest.approx(6.0),
            pytest.approx(10.0),
        ]

    def test_cumsum_axis_none_flattens(self):
        """axis=None → 展平为 1D。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.cumsum(a)
        assert result.shape == (4,)
        assert result.tolist() == [
            pytest.approx(1.0),
            pytest.approx(3.0),
            pytest.approx(6.0),
            pytest.approx(10.0),
        ]

    def test_cumsum_axis0(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.cumsum(a, axis=0)
        assert result.shape == (2, 2)
        assert result.tolist() == [
            [pytest.approx(1.0), pytest.approx(2.0)],
            [pytest.approx(4.0), pytest.approx(6.0)],
        ]

    def test_cumsum_axis1(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        result = ms.cumsum(a, axis=1)
        assert result.tolist() == [
            [pytest.approx(1.0), pytest.approx(3.0)],
            [pytest.approx(3.0), pytest.approx(7.0)],
        ]

    def test_cumsum_int_gives_i64(self):
        a = ms.array([1, 2, 3], dtype=ms.int64)
        result = ms.cumsum(a, axis=0)
        assert result.dtype == ms.int64
        assert result.tolist() == [1, 3, 6]


# ═══════════════════════════════════════════════════════════════
# TestReductionErrors — 错误处理
# ═══════════════════════════════════════════════════════════════

class TestReductionErrors:
    """错误输入处理。"""

    def test_axis_out_of_bounds(self):
        a = ms.array([1.0, 2.0, 3.0])
        with pytest.raises(Exception):
            ms.sum(a, axis=5)

    def test_negative_axis_out_of_bounds(self):
        a = ms.array([1.0, 2.0, 3.0])
        with pytest.raises(Exception):
            ms.sum(a, axis=-4)

    def test_out_shape_mismatch(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        out = ms.array([0.0, 0.0, 0.0])  # wrong shape
        with pytest.raises(Exception):
            ms.sum(a, axis=0, out=out)

    def test_out_dtype_mismatch(self):
        a = ms.array([1.0, 2.0, 3.0])
        out = ms.array([0, 0], dtype=ms.int64)  # wrong dtype for float sum
        with pytest.raises(Exception):
            ms.sum(a, axis=0, out=out)


# ═══════════════════════════════════════════════════════════════
# TestReductionMusa — GPU 验收
# ═══════════════════════════════════════════════════════════════

@musa_required
class TestReductionMusa:
    """GPU 验收测试（plan doc 验收标准）。"""

    def test_acceptance_sum_global(self):
        c = ms.array([[1.0, 2.0], [3.0, 4.0]], device="musa:0")
        assert ms.sum(c).item() == pytest.approx(10.0)

    def test_acceptance_sum_axis1(self):
        c = ms.array([[1.0, 2.0], [3.0, 4.0]], device="musa:0")
        result = ms.sum(c, axis=1)
        assert result.tolist() == [pytest.approx(3.0), pytest.approx(7.0)]

    def test_acceptance_sum_keepdims(self):
        c = ms.array([[1.0, 2.0], [3.0, 4.0]], device="musa:0")
        result = ms.sum(c, axis=0, keepdims=True)
        assert result.shape == (1, 2)

    def test_acceptance_argmax(self):
        c = ms.array([[1.0, 2.0], [3.0, 4.0]], device="musa:0")
        result = ms.argmax(c, axis=1)
        assert result.tolist() == [1, 1]

    def test_acceptance_int8_sum(self):
        a = ms.array([1, 2, 3], dtype=ms.int8, device="musa:0")
        result = ms.sum(a)
        assert result.dtype == ms.int64
        assert result.item() == 6

    def test_gpu_mean(self):
        a = ms.array([2.0, 4.0, 6.0, 8.0], device="musa:0")
        result = ms.mean(a)
        assert result.item() == pytest.approx(5.0)

    def test_gpu_max_axis(self):
        a = ms.array([[1.0, 5.0], [3.0, 2.0]], device="musa:0")
        result = ms.max(a, axis=1)
        assert result.tolist() == [pytest.approx(5.0), pytest.approx(3.0)]

    def test_gpu_cumsum(self):
        a = ms.array([1.0, 2.0, 3.0], device="musa:0")
        result = ms.cumsum(a, axis=0)
        assert result.tolist() == [
            pytest.approx(1.0),
            pytest.approx(3.0),
            pytest.approx(6.0),
        ]

    def test_gpu_prod(self):
        a = ms.array([2.0, 3.0, 4.0], device="musa:0")
        result = ms.prod(a)
        assert result.item() == pytest.approx(24.0)
