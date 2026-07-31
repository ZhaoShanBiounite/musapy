"""ms.add() 验收测试 — 逐元素加法端到端（Phase 6, P6.12）。

对应 ADR：L1-12（OpBuilder capture-safe）、L2-4（ops layer）、
L2-5（alias detection）、L1-8（out= stream semantics）、
L1-11（tolist 显式 sync + D2H）、L3-1（launch error 检测）、L3-2（OpContext）。
"""

import pytest

import musapy as ms


# ============================================================
# CPU 测试（mock 模式 / CI 可用）
# ============================================================


class TestAddBasic:
    """ms.add() 基本功能（CPU）。"""

    def test_add_f32(self):
        """float32 逐元素加法。"""
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="cpu")
        b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32, device="cpu")
        c = ms.add(a, b)
        assert c.tolist() == [5.0, 7.0, 9.0]

    def test_add_f64(self):
        """float64 逐元素加法。"""
        a = ms.array([1.0, 2.0], dtype=ms.float64, device="cpu")
        b = ms.array([3.0, 4.0], dtype=ms.float64, device="cpu")
        c = ms.add(a, b)
        assert c.tolist() == [4.0, 6.0]

    def test_add_dunder(self):
        """`a + b` 等价于 `ms.add(a, b)`。"""
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="cpu")
        b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32, device="cpu")
        c = a + b
        assert c.tolist() == [5.0, 7.0, 9.0]

    def test_add_returns_new_array(self):
        """无 out= 时返回新 Array。"""
        a = ms.array([1.0, 2.0], dtype=ms.float32, device="cpu")
        b = ms.array([3.0, 4.0], dtype=ms.float32, device="cpu")
        c = ms.add(a, b)
        # a 和 b 不被修改
        assert a.tolist() == [1.0, 2.0]
        assert b.tolist() == [3.0, 4.0]
        # c 是新数组
        assert c.tolist() == [4.0, 6.0]

    def test_add_self(self):
        """`a + a` 合法（两个输入共享同一 buffer 是 read-only）。"""
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="cpu")
        c = a + a
        assert c.tolist() == [2.0, 4.0, 6.0]

    def test_add_single_element(self):
        """单元素数组加法。"""
        a = ms.array([1.0], dtype=ms.float32, device="cpu")
        b = ms.array([2.0], dtype=ms.float32, device="cpu")
        c = ms.add(a, b)
        assert c.tolist() == [3.0]


class TestAddOut:
    """out= 参数测试（ADR L1-8）。"""

    def test_add_out(self):
        """out= 参数写入预分配 buffer。"""
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="cpu")
        b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32, device="cpu")
        out = ms.array([0.0, 0.0, 0.0], dtype=ms.float32, device="cpu")
        result = ms.add(a, b, out=out)
        # out 被写入
        assert out.tolist() == [5.0, 7.0, 9.0]
        # 返回值也指向同一数据
        assert result.tolist() == [5.0, 7.0, 9.0]

    def test_add_alias_error(self):
        """out 与输入相同 → MemoryError(AliasDetected)（ADR L2-5）。"""
        a = ms.array([1.0, 2.0], dtype=ms.float32, device="cpu")
        b = ms.array([3.0, 4.0], dtype=ms.float32, device="cpu")
        with pytest.raises(ms.MemoryError):
            ms.add(a, b, out=a)
        with pytest.raises(ms.MemoryError):
            ms.add(a, b, out=b)


class TestAddErrors:
    """错误场景测试。"""

    def test_shape_mismatch(self):
        """不同 shape → ShapeError。"""
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="cpu")
        b = ms.array([1.0, 2.0], dtype=ms.float32, device="cpu")
        with pytest.raises(ms.ShapeError):
            ms.add(a, b)

    def test_dtype_mismatch(self):
        """不同 float dtype → 类型提升到 float64（Phase 2 type promotion）。"""
        a = ms.array([1.0, 2.0], dtype=ms.float32, device="cpu")
        b = ms.array([3.0, 4.0], dtype=ms.float64, device="cpu")
        c = ms.add(a, b)
        assert c.dtype == ms.float64
        assert c.tolist() == [4.0, 6.0]


class TestToList:
    """tolist() / item() 测试（ADR L1-11）。"""

    def test_tolist_f32(self):
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="cpu")
        assert a.tolist() == [1.0, 2.0, 3.0]

    def test_tolist_f64(self):
        a = ms.array([1.0, 2.0], dtype=ms.float64, device="cpu")
        assert a.tolist() == [1.0, 2.0]

    def test_tolist_int32(self):
        a = ms.array([10, 20, 30], dtype=ms.int32, device="cpu")
        assert a.tolist() == [10, 20, 30]

    def test_tolist_empty(self):
        a = ms.array([], dtype=ms.float32, device="cpu")
        assert a.tolist() == []

    def test_item_scalar(self):
        """size=1 array → Python 标量。"""
        a = ms.array([3.14], dtype=ms.float32, device="cpu")
        v = a.item()
        assert abs(v - 3.14) < 1e-5

    def test_item_f64(self):
        a = ms.array([2.71828], dtype=ms.float64, device="cpu")
        assert abs(a.item() - 2.71828) < 1e-10

    def test_item_size_mismatch(self):
        """size > 1 → ValueError。"""
        a = ms.array([1.0, 2.0], dtype=ms.float32, device="cpu")
        with pytest.raises(ValueError):
            a.item()


# ============================================================
# MUSA 硬件测试（需真实 MUSA GPU，由用户手动运行）
# ============================================================


def has_musa():
    """检测是否有可用的 MUSA 设备。"""
    try:
        ms.set_default_device("musa:0")
        ms.set_default_device("cpu")  # 恢复
        return True
    except Exception:
        return False


musa_required = pytest.mark.skipif(
    not has_musa(), reason="no MUSA device available"
)


@musa_required
class TestAddMusa:
    """MUSA GPU 上的 add 端到端测试。"""

    def test_add_f32_musa(self):
        """float32 加法在 MUSA GPU 上运行。"""
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32)
        b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32)
        c = ms.add(a, b)
        assert c.tolist() == [5.0, 7.0, 9.0]

    def test_add_dunder_musa(self):
        """`a + b` 在 MUSA GPU 上运行。"""
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32)
        b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32)
        c = a + b
        assert c.tolist() == [5.0, 7.0, 9.0]

    def test_add_f64_musa(self):
        """float64 加法在 MUSA GPU 上运行。"""
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0], dtype=ms.float64)
        b = ms.array([3.0, 4.0], dtype=ms.float64)
        c = a + b
        assert c.tolist() == [4.0, 6.0]

    def test_acceptance(self):
        """Phase 6 验收标准（plan 文档）。"""
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32)
        b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32)
        c = a + b
        assert c.tolist() == [5.0, 7.0, 9.0]

    def test_add_out_musa(self):
        """out= 参数在 MUSA GPU 上运行。"""
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32)
        b = ms.array([4.0, 5.0, 6.0], dtype=ms.float32)
        out = ms.array([0.0, 0.0, 0.0], dtype=ms.float32)
        ms.add(a, b, out=out)
        assert out.tolist() == [5.0, 7.0, 9.0]
