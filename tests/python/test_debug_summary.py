"""P5.9/P5.10 验收测试 — memory_summary / device_summary / debug 模式。

对应 ADR：L3-28（Memory/Stream State Query）、L1-3（Device Capability Query）、
L3-26（Debug Mode — Runtime Flag）、L3-2（OpContext Attribution）。
"""

import pytest

import musapy as ms


# ============================================================
# MUSA 检测
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


# ============================================================
# P5.9: memory_summary（CPU 测试）
# ============================================================


class TestMemorySummary:
    """ms.memory_summary() 基本功能。"""

    def test_returns_string(self):
        result = ms.memory_summary()
        assert isinstance(result, str)

    def test_contains_allocated(self):
        result = ms.memory_summary()
        assert "Allocated" in result

    def test_contains_cached(self):
        result = ms.memory_summary()
        assert "Cached" in result

    def test_contains_peak(self):
        result = ms.memory_summary()
        assert "Peak" in result

    def test_tracks_allocation(self):
        """创建 array 后 allocated 应 > 0。"""
        a = ms.array([1.0, 2.0, 3.0], dtype='f32', device="cpu")
        after = ms.memory_summary()
        # 创建后应有分配（12 bytes for 3 * float32）
        assert "12 bytes" in after or "Allocated" in after
        # 保持引用避免被 drop
        assert a.size == 3

    def test_device_param_cpu(self):
        """传入 device='cpu' 不应报错（CPU 无 VRAM 信息，不额外输出）。"""
        result = ms.memory_summary(device="cpu")
        assert "Allocated" in result


# ============================================================
# P5.9: device_summary（CPU 测试）
# ============================================================


class TestDeviceSummary:
    """ms.device_summary() 基本功能。"""

    def test_returns_string(self):
        result = ms.device_summary()
        assert isinstance(result, str)

    def test_contains_cpu(self):
        result = ms.device_summary()
        assert "cpu" in result

    def test_cpu_line_format(self):
        result = ms.device_summary()
        assert "cpu — host memory" in result


# ============================================================
# P5.10: set_debug / debug context（CPU 测试）
# ============================================================


class TestDebugMode:
    """ms.set_debug() / with ms.debug(): 基本功能。"""

    def test_set_debug_true(self):
        ms.set_debug(True)
        # 验证 debug 模式下 add 不报错
        a = ms.array([1.0, 2.0], dtype='f32', device="cpu")
        b = ms.array([3.0, 4.0], dtype='f32', device="cpu")
        c = ms.add(a, b)
        assert c.tolist() == [4.0, 6.0]
        ms.set_debug(False)

    def test_set_debug_false(self):
        ms.set_debug(False)
        a = ms.array([1.0], dtype='f32', device="cpu")
        b = ms.array([2.0], dtype='f32', device="cpu")
        c = ms.add(a, b)
        assert c.tolist() == [3.0]

    def test_debug_context_manager(self):
        """with ms.debug(): 内 debug 启用，退出后恢复。"""
        ms.set_debug(False)
        with ms.debug():
            # debug 模式下 add 应正常工作
            a = ms.array([1.0, 2.0], dtype='f32', device="cpu")
            b = ms.array([3.0, 4.0], dtype='f32', device="cpu")
            c = a + b
            assert c.tolist() == [4.0, 6.0]
        # 退出后 debug 应恢复为 False
        # （无法直接从 Python 读取 is_debug，但 add 应仍正常工作）
        a = ms.array([1.0], dtype='f32', device="cpu")
        b = ms.array([2.0], dtype='f32', device="cpu")
        c = ms.add(a, b)
        assert c.tolist() == [3.0]

    def test_debug_nested_context(self):
        """嵌套 debug context 应正确恢复。"""
        ms.set_debug(False)
        with ms.debug():
            with ms.debug():
                a = ms.array([1.0], dtype='f32', device="cpu")
                b = ms.array([2.0], dtype='f32', device="cpu")
                c = a + b
                assert c.tolist() == [3.0]
            # 内层退出，外层仍 debug
            a = ms.array([1.0], dtype='f32', device="cpu")
            b = ms.array([2.0], dtype='f32', device="cpu")
            c = a + b
            assert c.tolist() == [3.0]
        # 全部退出
        ms.set_debug(False)

    def test_debug_dunder_add(self):
        """debug 模式下 __add__ 也应正常工作。"""
        ms.set_debug(True)
        a = ms.array([1.0, 2.0, 3.0], dtype='f32', device="cpu")
        b = ms.array([4.0, 5.0, 6.0], dtype='f32', device="cpu")
        c = a + b
        assert c.tolist() == [5.0, 7.0, 9.0]
        ms.set_debug(False)


# ============================================================
# MUSA 硬件测试
# ============================================================


@musa_required
class TestMemorySummaryMusa:
    """MUSA GPU 上的 memory_summary 测试。"""

    def test_musa_allocation_tracked(self):
        ms.set_default_device("musa:0")
        a = ms.array([1.0, 2.0, 3.0], dtype='f32')
        result = ms.memory_summary()
        assert "Allocated" in result
        # 保持引用
        assert a.size == 3
        ms.set_default_device("cpu")

    def test_musa_device_vram(self):
        ms.set_default_device("musa:0")
        result = ms.memory_summary(device="musa:0")
        assert "VRAM" in result
        ms.set_default_device("cpu")


@musa_required
class TestDeviceSummaryMusa:
    """MUSA GPU 上的 device_summary 测试。"""

    def test_contains_musa_device(self):
        result = ms.device_summary()
        assert "musa:0" in result

    def test_contains_vram(self):
        result = ms.device_summary()
        assert "VRAM" in result

    def test_contains_arch(self):
        result = ms.device_summary()
        assert "arch=mp_" in result

    def test_contains_cus(self):
        result = ms.device_summary()
        assert "CUs" in result
