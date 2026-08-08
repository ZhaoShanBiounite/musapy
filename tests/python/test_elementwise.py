"""ms.add() 广播测试（Phase 1, P1.10, ADR-002-D2）
+ Phase 2 Elementwise Suite 测试

验证 stride-aware _v2 ABI + NumPy 广播规则：
- (3,1) + (4,) → (3,4)
- 0-dim + 任意 shape → 任意 shape
- 高维广播
- 不兼容 shape → ShapeError

Phase 2:
- Binary ops: sub/mul/div/pow + broadcast
- Unary ops: sin/cos/exp/log/abs/sign/neg
- Clamp
- Type promotion (int+float → float)
- astype
- Dunders: -, *, /, **, unary -, abs()
"""

import numpy as np
import pytest

import musapy as ms


# ── CPU 广播测试 ──────────────────────────────────────────────


class TestBroadcast:
    """ms.add() 广播功能（CPU）"""

    def test_broadcast_3x1_plus_4(self):
        """(3,1) + (4,) → (3,4)"""
        a = ms.array([[1.0], [2.0], [3.0]], device="cpu")  # (3,1)
        b = ms.array([10.0, 20.0, 30.0, 40.0], device="cpu")  # (4,)
        c = ms.add(a, b)
        assert c.shape == (3, 4)
        assert c.tolist() == [
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]

    def test_broadcast_1x4_plus_3x1(self):
        """(1,4) + (3,1) → (3,4)"""
        a = ms.array([[1.0, 2.0, 3.0, 4.0]], device="cpu")  # (1,4)
        b = ms.array([[10.0], [20.0], [30.0]], device="cpu")  # (3,1)
        c = ms.add(a, b)
        assert c.shape == (3, 4)
        assert c.tolist() == [
            [11.0, 12.0, 13.0, 14.0],
            [21.0, 22.0, 23.0, 24.0],
            [31.0, 32.0, 33.0, 34.0],
        ]

    def test_broadcast_same_shape(self):
        """同 shape 无广播，等价原 add"""
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([4.0, 5.0, 6.0], device="cpu")
        c = ms.add(a, b)
        assert c.shape == (3,)
        assert c.tolist() == [5.0, 7.0, 9.0]

    def test_broadcast_2d_same_shape(self):
        """2D 同 shape"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        b = ms.array([[10.0, 20.0], [30.0, 40.0]], device="cpu")
        c = ms.add(a, b)
        assert c.shape == (2, 2)
        assert c.tolist() == [[11.0, 22.0], [33.0, 44.0]]

    def test_broadcast_0d_scalar_plus_1d(self):
        """0-dim + (n,) → (n,)"""
        a = ms.array(100.0, device="cpu")  # 0-dim
        b = ms.array([1.0, 2.0, 3.0], device="cpu")
        c = ms.add(a, b)
        assert c.shape == (3,)
        assert c.tolist() == [101.0, 102.0, 103.0]

    def test_broadcast_0d_plus_2d(self):
        """0-dim + (2,3) → (2,3)"""
        a = ms.array(10.0, device="cpu")  # 0-dim
        b = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu")
        c = ms.add(a, b)
        assert c.shape == (2, 3)
        assert c.tolist() == [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]

    def test_broadcast_0d_plus_0d(self):
        """0-dim + 0-dim → 0-dim"""
        a = ms.array(3.0, device="cpu")
        b = ms.array(4.0, device="cpu")
        c = ms.add(a, b)
        assert c.shape == ()
        assert c.item() == 7.0

    def test_broadcast_high_dim(self):
        """(2,1,3) + (4,1) → (2,4,3)"""
        # a: shape (2,1,3)
        a = ms.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], device="cpu")
        # b: shape (4,1)
        b = ms.array([[10.0], [20.0], [30.0], [40.0]], device="cpu")
        c = ms.add(a, b)
        assert c.shape == (2, 4, 3)
        # c[0][0] = [1,2,3] + 10 = [11,12,13]
        # c[0][3] = [1,2,3] + 40 = [41,42,43]
        # c[1][0] = [4,5,6] + 10 = [14,15,16]
        assert c.tolist()[0][0] == [11.0, 12.0, 13.0]
        assert c.tolist()[0][3] == [41.0, 42.0, 43.0]
        assert c.tolist()[1][0] == [14.0, 15.0, 16.0]

    def test_broadcast_1_plus_2d(self):
        """(1,) + (2,3) → (2,3)"""
        a = ms.array([100.0], device="cpu")  # (1,)
        b = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu")
        c = ms.add(a, b)
        assert c.shape == (2, 3)
        assert c.tolist() == [[101.0, 102.0, 103.0], [104.0, 105.0, 106.0]]

    def test_broadcast_dunder(self):
        """a + b 等价 ms.add(a, b)（含广播）"""
        a = ms.array([[1.0], [2.0], [3.0]], device="cpu")
        b = ms.array([10.0, 20.0, 30.0, 40.0], device="cpu")
        c = a + b
        assert c.shape == (3, 4)
        assert c.tolist() == [
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]

    def test_broadcast_f64(self):
        """float64 广播"""
        a = ms.array([[1.0], [2.0]], dtype='f64', device="cpu")
        b = ms.array([0.1, 0.2, 0.3], dtype='f64', device="cpu")
        c = ms.add(a, b)
        assert c.shape == (2, 3)
        assert c.dtype == 'f64'
        row0 = c.tolist()[0]
        assert abs(row0[0] - 1.1) < 1e-10
        assert abs(row0[1] - 1.2) < 1e-10
        assert abs(row0[2] - 1.3) < 1e-10

    def test_broadcast_out_param(self):
        """out= 参数配合广播"""
        a = ms.array([[1.0], [2.0], [3.0]], device="cpu")
        b = ms.array([10.0, 20.0, 30.0, 40.0], device="cpu")
        out = ms.array([[0.0] * 4] * 3, device="cpu")
        result = ms.add(a, b, out=out)
        assert result.shape == (3, 4)
        assert result.tolist() == [
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]


class TestBroadcastErrors:
    """广播错误情况"""

    def test_broadcast_incompatible_shapes(self):
        """(2,3) + (4,) → ShapeError（3 != 4 且都非 1）"""
        a = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu")
        b = ms.array([1.0, 2.0, 3.0, 4.0], device="cpu")
        with pytest.raises(ms.ShapeError):
            ms.add(a, b)

    def test_broadcast_incompatible_1d(self):
        """(3,) + (4,) → ShapeError"""
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([1.0, 2.0, 3.0, 4.0], device="cpu")
        with pytest.raises(ms.ShapeError):
            ms.add(a, b)

    def test_broadcast_incompatible_2d(self):
        """(2,3) + (2,4) → ShapeError"""
        a = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cpu")
        b = ms.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device="cpu")
        with pytest.raises(ms.ShapeError):
            ms.add(a, b)

    def test_broadcast_out_shape_mismatch(self):
        """out shape 与广播输出不匹配 → ShapeError"""
        a = ms.array([[1.0], [2.0], [3.0]], device="cpu")
        b = ms.array([10.0, 20.0, 30.0, 40.0], device="cpu")
        out = ms.array([[0.0, 0.0, 0.0]], device="cpu")  # (1,3) != (3,4)
        with pytest.raises(ms.ShapeError):
            ms.add(a, b, out=out)


# ── MUSA 硬件测试 ─────────────────────────────────────────────


def has_musa():
    """探测是否有可用 MUSA 设备。"""
    try:
        ms.set_default_device("musa:0")
        ms.set_default_device("cpu")
        return True
    except Exception:
        return False


musa_required = pytest.mark.skipif(not has_musa(), reason="no MUSA device available")


@musa_required
class TestBroadcastMusa:
    """ms.add() 广播功能（MUSA GPU）"""

    def test_broadcast_3x1_plus_4_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([[1.0], [2.0], [3.0]])
        b = ms.array([10.0, 20.0, 30.0, 40.0])
        c = ms.add(a, b)
        assert c.shape == (3, 4)
        assert c.tolist() == [
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]

    def test_broadcast_0d_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array(100.0)
        b = ms.array([1.0, 2.0, 3.0])
        c = a + b
        assert c.shape == (3,)
        assert c.tolist() == [101.0, 102.0, 103.0]

    def test_broadcast_dunder_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([[1.0], [2.0], [3.0]])
        b = ms.array([10.0, 20.0, 30.0, 40.0])
        c = a + b
        assert c.shape == (3, 4)
        assert c.tolist()[0] == [11.0, 21.0, 31.0, 41.0]

    def test_broadcast_acceptance(self):
        """Phase 1 验收标准（ADR-002-D2）"""
        ms.set_default_device("musa:0")
        a = ms.array([[1.0], [2.0], [3.0]])  # (3,1)
        b = ms.array([10.0, 20.0, 30.0, 40.0])  # (4,)
        c = ms.add(a, b)  # (3,4)
        assert c.shape == (3, 4)
        assert c.tolist() == [
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]


# ═══════════════════════════════════════════════════════════════
# Phase 2: Elementwise Suite
# ═══════════════════════════════════════════════════════════════


class TestBinaryOps:
    """Binary elementwise ops（CPU）"""

    def test_sub(self):
        a = ms.array([5.0, 3.0, 1.0], device="cpu")
        b = ms.array([1.0, 2.0, 3.0], device="cpu")
        assert ms.sub(a, b).tolist() == [4.0, 1.0, -2.0]

    def test_mul(self):
        a = ms.array([2.0, 3.0, 4.0], device="cpu")
        b = ms.array([5.0, 6.0, 7.0], device="cpu")
        assert ms.mul(a, b).tolist() == [10.0, 18.0, 28.0]

    def test_div(self):
        a = ms.array([10.0, 9.0, 8.0], device="cpu")
        b = ms.array([2.0, 3.0, 4.0], device="cpu")
        assert ms.div(a, b).tolist() == [5.0, 3.0, 2.0]

    def test_pow(self):
        a = ms.array([2.0, 3.0, 4.0], device="cpu")
        b = ms.array([3.0, 2.0, 0.5], device="cpu")
        result = ms.pow(a, b).tolist()
        assert abs(result[0] - 8.0) < 1e-5
        assert abs(result[1] - 9.0) < 1e-5
        assert abs(result[2] - 2.0) < 1e-5

    def test_sub_broadcast(self):
        """(3,1) - (4,) → (3,4)"""
        a = ms.array([[10.0], [20.0], [30.0]], device="cpu")
        b = ms.array([1.0, 2.0, 3.0, 4.0], device="cpu")
        c = ms.sub(a, b)
        assert c.shape == (3, 4)
        assert c.tolist()[0] == [9.0, 8.0, 7.0, 6.0]
        assert c.tolist()[2] == [29.0, 28.0, 27.0, 26.0]

    def test_mul_broadcast_0d(self):
        """scalar * (n,) → (n,)"""
        a = ms.array(3.0, device="cpu")
        b = ms.array([1.0, 2.0, 3.0], device="cpu")
        c = ms.mul(a, b)
        assert c.shape == (3,)
        assert c.tolist() == [3.0, 6.0, 9.0]

    def test_binary_out_param(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([4.0, 5.0, 6.0], device="cpu")
        out = ms.array([0.0, 0.0, 0.0], device="cpu")
        result = ms.mul(a, b, out=out)
        assert result.tolist() == [4.0, 10.0, 18.0]

    def test_binary_f64(self):
        a = ms.array([1.0, 2.0], dtype='f64', device="cpu")
        b = ms.array([0.1, 0.2], dtype='f64', device="cpu")
        c = ms.add(a, b)
        assert c.dtype == 'f64'
        assert abs(c.tolist()[0] - 1.1) < 1e-10


class TestUnaryOps:
    """Unary elementwise ops（CPU）"""

    def test_sin(self):
        import math
        a = ms.array([0.0, math.pi / 2, math.pi], device="cpu")
        result = ms.sin(a).tolist()
        assert abs(result[0]) < 1e-6
        assert abs(result[1] - 1.0) < 1e-6
        assert abs(result[2]) < 1e-5

    def test_cos(self):
        import math
        a = ms.array([0.0, math.pi / 2, math.pi], device="cpu")
        result = ms.cos(a).tolist()
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1]) < 1e-5
        assert abs(result[2] + 1.0) < 1e-5

    def test_exp(self):
        import math
        a = ms.array([0.0, 1.0, 2.0], device="cpu")
        result = ms.exp(a).tolist()
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1] - math.e) < 1e-5
        assert abs(result[2] - math.e**2) < 1e-4

    def test_log(self):
        import math
        a = ms.array([1.0, math.e, math.e**2], device="cpu")
        result = ms.log(a).tolist()
        assert abs(result[0]) < 1e-6
        assert abs(result[1] - 1.0) < 1e-5
        assert abs(result[2] - 2.0) < 1e-5

    def test_abs(self):
        a = ms.array([-3.0, 0.0, 5.0], device="cpu")
        assert ms.abs(a).tolist() == [3.0, 0.0, 5.0]

    def test_sign(self):
        a = ms.array([-10.0, 0.0, 7.0], device="cpu")
        assert ms.sign(a).tolist() == [-1.0, 0.0, 1.0]

    def test_neg(self):
        a = ms.array([1.0, -2.0, 0.0], device="cpu")
        assert ms.neg(a).tolist() == [-1.0, 2.0, 0.0]

    def test_unary_2d(self):
        a = ms.array([[-1.0, 2.0], [3.0, -4.0]], device="cpu")
        c = ms.abs(a)
        assert c.shape == (2, 2)
        assert c.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_unary_out_param(self):
        a = ms.array([1.0, 4.0, 9.0], device="cpu")
        out = ms.array([0.0, 0.0, 0.0], device="cpu")
        import math
        result = ms.log(a, out=out)
        assert abs(result.tolist()[0]) < 1e-6
        assert abs(result.tolist()[1] - math.log(4.0)) < 1e-5

    def test_unary_f64(self):
        a = ms.array([0.0, 1.0], dtype='f64', device="cpu")
        c = ms.exp(a)
        assert c.dtype == 'f64'
        assert abs(c.tolist()[0] - 1.0) < 1e-12


class TestClamp:
    """ms.clamp() 测试（CPU）"""

    def test_clamp_basic(self):
        a = ms.array([-5.0, 0.5, 3.0, 10.0], device="cpu")
        c = ms.clamp(a, 0.0, 1.0)
        assert c.tolist() == [0.0, 0.5, 1.0, 1.0]

    def test_clamp_no_effect(self):
        a = ms.array([0.2, 0.5, 0.8], device="cpu")
        c = ms.clamp(a, 0.0, 1.0)
        result = c.tolist()
        assert abs(result[0] - 0.2) < 1e-6
        assert abs(result[1] - 0.5) < 1e-6
        assert abs(result[2] - 0.8) < 1e-6

    def test_clamp_all_below(self):
        a = ms.array([-10.0, -5.0, -1.0], device="cpu")
        c = ms.clamp(a, 0.0, 100.0)
        assert c.tolist() == [0.0, 0.0, 0.0]

    def test_clamp_2d(self):
        a = ms.array([[-1.0, 5.0], [0.5, 20.0]], device="cpu")
        c = ms.clamp(a, 0.0, 10.0)
        assert c.tolist() == [[0.0, 5.0], [0.5, 10.0]]

    def test_clamp_f64(self):
        a = ms.array([-1.0, 0.5, 2.0], dtype='f64', device="cpu")
        c = ms.clamp(a, 0.0, 1.0)
        assert c.dtype == 'f64'
        assert c.tolist() == [0.0, 0.5, 1.0]


class TestTypePromotion:
    """类型提升测试（CPU）"""

    def test_i64_plus_f64(self):
        """int64 + float64 → float64"""
        a = ms.array([1, 2, 3], dtype='i64', device="cpu")
        b = ms.array([0.5, 0.5, 0.5], dtype='f64', device="cpu")
        c = ms.add(a, b)
        assert c.dtype == 'f64'
        assert c.tolist() == [1.5, 2.5, 3.5]

    def test_i32_plus_f32(self):
        """int32 + float32 → float32"""
        a = ms.array([1, 2, 3], dtype='i32', device="cpu")
        b = ms.array([0.5, 0.5, 0.5], dtype='f32', device="cpu")
        c = ms.add(a, b)
        assert c.dtype == 'f32'
        result = c.tolist()
        assert abs(result[0] - 1.5) < 1e-6

    def test_i64_plus_f32(self):
        """int64 + float32 → float32（JAX 语义：整数不因位宽升级浮点，
        对齐 v0.2 计划 §1.3 与 ADR L1-14 扩展表；2026-08 修正自 f64）"""
        a = ms.array([1, 2, 3], dtype='i64', device="cpu")
        b = ms.array([0.5, 0.5, 0.5], dtype='f32', device="cpu")
        c = ms.add(a, b)
        assert c.dtype == 'f32'
        result = c.tolist()
        assert abs(result[0] - 1.5) < 1e-6

    def test_f32_plus_f64(self):
        """float32 + float64 → float64"""
        a = ms.array([1.0, 2.0], dtype='f32', device="cpu")
        b = ms.array([0.1, 0.2], dtype='f64', device="cpu")
        c = ms.add(a, b)
        assert c.dtype == 'f64'

    def test_same_dtype_no_promotion(self):
        """同 dtype 不触发 cast"""
        a = ms.array([1.0, 2.0], dtype='f32', device="cpu")
        b = ms.array([3.0, 4.0], dtype='f32', device="cpu")
        c = ms.add(a, b)
        assert c.dtype == 'f32'
        assert c.tolist() == [4.0, 6.0]

    def test_promotion_sub(self):
        """sub 也支持类型提升"""
        a = ms.array([10, 20], dtype='i64', device="cpu")
        b = ms.array([0.5, 1.5], dtype='f64', device="cpu")
        c = ms.sub(a, b)
        assert c.dtype == 'f64'
        assert c.tolist() == [9.5, 18.5]


class TestAstype:
    """astype 测试（CPU）"""

    def test_f32_to_f64(self):
        a = ms.array([1.0, 2.0, 3.0], dtype='f32', device="cpu")
        b = a.astype('f64')
        assert b.dtype == 'f64'
        assert b.tolist() == [1.0, 2.0, 3.0]

    def test_f64_to_f32(self):
        a = ms.array([1.5, 2.5], dtype='f64', device="cpu")
        b = a.astype('f32')
        assert b.dtype == 'f32'
        assert abs(b.tolist()[0] - 1.5) < 1e-6

    def test_i64_to_f32(self):
        a = ms.array([1, 2, 3], dtype='i64', device="cpu")
        b = a.astype('f32')
        assert b.dtype == 'f32'
        assert b.tolist() == [1.0, 2.0, 3.0]

    def test_i32_to_f64(self):
        a = ms.array([10, 20], dtype='i32', device="cpu")
        b = a.astype('f64')
        assert b.dtype == 'f64'
        assert b.tolist() == [10.0, 20.0]

    def test_astype_preserves_shape(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], dtype='f32', device="cpu")
        b = a.astype('f64')
        assert b.shape == (2, 2)
        assert b.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_astype_same_dtype_copy(self):
        """同 dtype astype 返回深拷贝"""
        a = ms.array([1.0, 2.0], dtype='f32', device="cpu")
        b = a.astype('f32')
        assert b.dtype == 'f32'
        assert b.tolist() == [1.0, 2.0]


class TestAstypeMusa:
    """astype 测试（MUSA GPU；回归 cast dispatch unreachable bug）"""

    def test_f32_to_i64_musa(self):
        """f32 → i64（截断取整；dispatch 曾缺 arm 直接 panic）"""
        ms.set_default_device("musa:0")
        a = ms.array([1.7, -2.9, 3.0], dtype='f32')
        b = a.astype('i64')
        assert b.dtype == 'i64'
        assert b.tolist() == [1, -2, 3]

    def test_f64_to_i64_musa(self):
        """f64 → i64"""
        ms.set_default_device("musa:0")
        a = ms.array([1.7, 2.9, -0.9], dtype='f64')
        b = a.astype('i64')
        assert b.dtype == 'i64'
        assert b.tolist() == [1, 2, 0]

    def test_complex_to_real_rejected(self):
        """complex → real 显式 DtypeError（曾因 validate 与 dispatch 不一致触发
        unreachable panic）"""
        ms.set_default_device("musa:0")
        for src, dst in [('c64', 'f64'), ('c64', 'f32'), ('c128', 'f64'), ('c128', 'f32')]:
            x = ms.array([1 + 2j], dtype=src)
            with pytest.raises(ms.DtypeError):
                x.astype(dst)

    def test_c64_to_c128_musa(self):
        """complex 宽度提升 c64 → c128 仍可用"""
        ms.set_default_device("musa:0")
        x = ms.array([1 + 2j, 3 - 4j], dtype='c64')
        y = x.astype('c128')
        assert y.dtype == 'c128'
        assert y.tolist() == [1 + 2j, 3 - 4j]


class TestDunders:
    """Python dunder 运算符测试（CPU）"""

    def test_sub_dunder(self):
        a = ms.array([5.0, 3.0], device="cpu")
        b = ms.array([1.0, 2.0], device="cpu")
        assert (a - b).tolist() == [4.0, 1.0]

    def test_mul_dunder(self):
        a = ms.array([2.0, 3.0], device="cpu")
        b = ms.array([4.0, 5.0], device="cpu")
        assert (a * b).tolist() == [8.0, 15.0]

    def test_truediv_dunder(self):
        a = ms.array([10.0, 6.0], device="cpu")
        b = ms.array([2.0, 3.0], device="cpu")
        assert (a / b).tolist() == [5.0, 2.0]

    def test_pow_dunder(self):
        a = ms.array([2.0, 3.0], device="cpu")
        b = ms.array([3.0, 2.0], device="cpu")
        result = (a ** b).tolist()
        assert abs(result[0] - 8.0) < 1e-5
        assert abs(result[1] - 9.0) < 1e-5

    def test_neg_dunder(self):
        a = ms.array([1.0, -2.0, 0.0], device="cpu")
        assert (-a).tolist() == [-1.0, 2.0, 0.0]

    def test_abs_dunder(self):
        a = ms.array([-3.0, 0.0, 5.0], device="cpu")
        assert abs(a).tolist() == [3.0, 0.0, 5.0]

    def test_dunder_broadcast(self):
        """dunder 也支持广播"""
        a = ms.array([[1.0], [2.0]], device="cpu")  # (2,1)
        b = ms.array([10.0, 20.0, 30.0], device="cpu")  # (3,)
        c = a * b
        assert c.shape == (2, 3)
        assert c.tolist() == [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]]


# ── Phase 2 MUSA 硬件测试 ─────────────────────────────────────


@musa_required
class TestElementwiseMusa:
    """Phase 2 elementwise ops（MUSA GPU）"""

    def test_binary_ops_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([6.0, 8.0, 10.0])
        b = ms.array([2.0, 4.0, 5.0])
        assert ms.sub(a, b).tolist() == [4.0, 4.0, 5.0]
        assert ms.mul(a, b).tolist() == [12.0, 32.0, 50.0]
        assert ms.div(a, b).tolist() == [3.0, 2.0, 2.0]

    def test_pow_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([2.0, 3.0])
        b = ms.array([4.0, 3.0])
        result = ms.pow(a, b).tolist()
        assert abs(result[0] - 16.0) < 1e-4
        assert abs(result[1] - 27.0) < 1e-4

    def test_unary_ops_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([-2.0, 0.0, 3.0])
        assert ms.abs(a).tolist() == [2.0, 0.0, 3.0]
        assert ms.sign(a).tolist() == [-1.0, 0.0, 1.0]
        assert ms.neg(a).tolist() == [2.0, 0.0, -3.0]

    def test_sin_cos_musa(self):
        import math
        ms.set_default_device("musa:0")
        a = ms.array([0.0, math.pi / 2])
        sin_r = ms.sin(a).tolist()
        cos_r = ms.cos(a).tolist()
        assert abs(sin_r[0]) < 1e-5
        assert abs(sin_r[1] - 1.0) < 1e-5
        assert abs(cos_r[0] - 1.0) < 1e-5
        assert abs(cos_r[1]) < 1e-5

    def test_exp_log_musa(self):
        import math
        ms.set_default_device("musa:0")
        a = ms.array([0.0, 1.0])
        exp_r = ms.exp(a).tolist()
        assert abs(exp_r[0] - 1.0) < 1e-5
        assert abs(exp_r[1] - math.e) < 1e-4
        b = ms.array([1.0, math.e])
        log_r = ms.log(b).tolist()
        assert abs(log_r[0]) < 1e-5
        assert abs(log_r[1] - 1.0) < 1e-5

    def test_clamp_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([-5.0, 0.5, 10.0])
        c = ms.clamp(a, 0.0, 1.0)
        assert c.tolist() == [0.0, 0.5, 1.0]

    def test_broadcast_binary_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([[1.0], [2.0], [3.0]])  # (3,1)
        b = ms.array([10.0, 20.0, 30.0, 40.0])  # (4,)
        c = ms.mul(a, b)
        assert c.shape == (3, 4)
        assert c.tolist()[0] == [10.0, 20.0, 30.0, 40.0]
        assert c.tolist()[2] == [30.0, 60.0, 90.0, 120.0]

    def test_dunders_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([4.0, 9.0])
        b = ms.array([2.0, 3.0])
        assert (a - b).tolist() == [2.0, 6.0]
        assert (a * b).tolist() == [8.0, 27.0]
        assert (a / b).tolist() == [2.0, 3.0]
        assert (-a).tolist() == [-4.0, -9.0]
        assert abs(ms.array([-4.0, 9.0])).tolist() == [4.0, 9.0]

    def test_type_promotion_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([1, 2, 3], dtype='i64')
        b = ms.array([0.5, 0.5, 0.5], dtype='f64')
        c = ms.add(a, b)
        assert c.dtype == 'f64'
        assert c.tolist() == [1.5, 2.5, 3.5]

    def test_astype_musa(self):
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0], dtype='f32')
        b = a.astype('f64')
        assert b.dtype == 'f64'
        assert b.tolist() == [1.0, 2.0, 3.0]

    def test_phase2_acceptance(self):
        """Phase 2 验收：GPU 上完整 elementwise 流水线"""
        ms.set_default_device("musa:0")
        # binary + broadcast
        a = ms.array([[1.0], [2.0]])  # (2,1)
        b = ms.array([3.0, 4.0, 5.0])  # (3,)
        c = ms.add(a, b)  # (2,3)
        assert c.shape == (2, 3)
        assert c.tolist() == [[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]]
        # unary
        d = ms.exp(ms.array([0.0, 1.0]))
        assert abs(d.tolist()[0] - 1.0) < 1e-5
        # clamp
        e = ms.clamp(ms.array([-1.0, 0.5, 2.0]), 0.0, 1.0)
        assert e.tolist() == [0.0, 0.5, 1.0]
        # promotion
        f = ms.mul(ms.array([2, 3], dtype='i64'), ms.array([1.5, 2.5], dtype='f64'))
        assert f.dtype == 'f64'
        assert f.tolist() == [3.0, 7.5]

    # ── P3: float4 向量化路径 ─────────────────────────────────

    def test_vec4_path_correctness(self):
        """1M 连续对齐 → vec4 路径：binary 5 op + unary 7 op 对比 numpy。"""
        rng = np.random.default_rng(51)
        n = 1_000_000  # ≥ VEC4_THRESHOLD 且 %4==0
        x = (rng.random(n).astype(np.float32) * 2.0 + 0.5)  # 正数域，避免 log/div 奇异
        y = rng.random(n).astype(np.float32) * 2.0 + 0.5
        a = ms.array(x.tolist(), device="musa:0")
        b = ms.array(y.tolist(), device="musa:0")
        for op, wf in ((ms.add, lambda: x + y), (ms.sub, lambda: x - y),
                       (ms.mul, lambda: x * y), (ms.div, lambda: x / y)):
            np.testing.assert_allclose(np.array(op(a, b).tolist()), wf(),
                                       rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.array(ms.pow(a, b).tolist()),
                                   np.power(x, y), rtol=1e-2, atol=1e-2)
        for op, wf in ((ms.abs, lambda: np.abs(x)), (ms.neg, lambda: -x),
                       (ms.exp, lambda: np.exp(x)), (ms.log, lambda: np.log(x)),
                       (ms.sin, lambda: np.sin(x)), (ms.cos, lambda: np.cos(x))):
            np.testing.assert_allclose(np.array(op(a).tolist()), wf(),
                                       rtol=1e-3, atol=1e-3)
        half = ms.array([0.5], device="musa:0")
        np.testing.assert_allclose(np.array(ms.sign(ms.sub(a, half)).tolist()),
                                   np.sign(x - 0.5), rtol=1e-3, atol=1e-3)

    def test_vec4_threshold_boundaries(self):
        """阈值两侧 + n%4≠0：全部走标量路径且结果正确。"""
        rng = np.random.default_rng(52)
        for n in (999_999, 1_000_001, 1_000_003):
            x = rng.random(n).astype(np.float32)
            a = ms.array(x.tolist(), device="musa:0")
            np.testing.assert_allclose(np.array(ms.exp(a).tolist()),
                                       np.exp(x), rtol=1e-3, atol=1e-3)

    def test_vec4_unaligned_offset_view(self):
        """offset 视图（指针未 16B 对齐）→ 标量路径，结果正确。"""
        rng = np.random.default_rng(53)
        x = rng.random(1_000_002).astype(np.float32)
        a = ms.array(x.tolist(), device="musa:0")
        sl = ms.slice(a, [[1, 1_000_001, 1]])  # offset=1 → 未对齐
        np.testing.assert_allclose(np.array(ms.add(sl, sl).tolist()),
                                   x[1:1_000_001] * 2, rtol=1e-3, atol=1e-3)

    def test_vec4_broadcast_not_triggered(self):
        """广播（stride=0）不触发 vec4（is_contiguous 检查），结果正确。"""
        rng = np.random.default_rng(54)
        mat = rng.random((1000, 1000)).astype(np.float32)
        am = ms.array(mat.tolist(), device="musa:0")
        scl = ms.array([2.0], device="musa:0")
        np.testing.assert_allclose(np.array(ms.add(am, scl).tolist()),
                                   mat + 2.0, rtol=1e-3, atol=1e-3)
