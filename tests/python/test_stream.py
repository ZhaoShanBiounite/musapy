"""多 Stream 交叉使用验收测试（Phase 7, P7.2）。

对应 ADR：L1-7（default stream）、L1-8（out= stream 语义，自动 wait）、
L3-10（dealloc stream 选择策略 b）、L3-11（deferred-free）。

验证目标：
- 两个 stream 上分别创建 Array 并执行 add，互不干扰
- 跨 stream 的 out= 操作（自动 wait 输入 stream）
- stream.synchronize() 后数据正确
- 多 stream 交替使用不崩溃
"""

import pytest

import musapy as ms
from musapy import Stream


# ============================================================
# 辅助
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
# CPU 多 Stream 测试
# ============================================================


class TestMultiStreamBasic:
    """多 stream 基本隔离性（CPU）。"""

    def test_two_streams_independent_arrays(self):
        """两个 stream 上分别创建 Array，互不干扰。"""
        s1 = Stream("cpu")
        s2 = Stream("cpu")

        with ms.stream(s1):
            a = ms.array([1.0, 2.0, 3.0], dtype='f32')
        with ms.stream(s2):
            b = ms.array([4.0, 5.0, 6.0], dtype='f32')

        # 各自同步后数据正确
        s1.synchronize()
        s2.synchronize()
        assert a.tolist() == [1.0, 2.0, 3.0]
        assert b.tolist() == [4.0, 5.0, 6.0]

    def test_cross_stream_add(self):
        """跨 stream 输入执行 add（ADR L1-8：自动 wait 输入 stream）。"""
        s1 = Stream("cpu")
        s2 = Stream("cpu")

        with ms.stream(s1):
            a = ms.array([1.0, 2.0], dtype='f32')
        with ms.stream(s2):
            b = ms.array([3.0, 4.0], dtype='f32')

        # 在 s2 上执行 add，输入 a 来自 s1
        with ms.stream(s2):
            c = ms.add(a, b)

        s2.synchronize()
        assert c.tolist() == [4.0, 6.0]

    def test_cross_stream_out_param(self):
        """跨 stream 的 out= 操作（ADR L1-8）。"""
        s1 = Stream("cpu")
        s2 = Stream("cpu")

        with ms.stream(s1):
            a = ms.array([1.0, 2.0, 3.0], dtype='f32')
        with ms.stream(s2):
            b = ms.array([4.0, 5.0, 6.0], dtype='f32')
            out = ms.array([0.0, 0.0, 0.0], dtype='f32')

        # 在 s2 上用 out= 执行，a 来自 s1
        with ms.stream(s2):
            result = ms.add(a, b, out=out)

        s2.synchronize()
        assert out.tolist() == [5.0, 7.0, 9.0]
        assert result.tolist() == [5.0, 7.0, 9.0]

    def test_stream_ops_do_not_corrupt_other_stream(self):
        """一个 stream 上的操作不破坏另一个 stream 的数据。"""
        s1 = Stream("cpu")
        s2 = Stream("cpu")

        with ms.stream(s1):
            a = ms.array([10.0, 20.0], dtype='f32')
        with ms.stream(s2):
            b = ms.array([1.0, 2.0], dtype='f32')
            c = ms.add(b, b)  # s2 上独立计算

        s1.synchronize()
        s2.synchronize()

        # s1 的数据不受 s2 操作影响
        assert a.tolist() == [10.0, 20.0]
        assert c.tolist() == [2.0, 4.0]


class TestMultiStreamInterleaved:
    """多 stream 交替使用不崩溃。"""

    def test_interleaved_ops(self):
        """交替在两个 stream 上执行多个 op。"""
        s1 = Stream("cpu")
        s2 = Stream("cpu")

        with ms.stream(s1):
            a1 = ms.array([1.0, 2.0], dtype='f32')
        with ms.stream(s2):
            b1 = ms.array([3.0, 4.0], dtype='f32')
        with ms.stream(s1):
            c1 = ms.add(a1, a1)
        with ms.stream(s2):
            d1 = ms.add(b1, b1)
        with ms.stream(s1):
            e1 = ms.add(c1, a1)
        with ms.stream(s2):
            f1 = ms.add(d1, b1)

        s1.synchronize()
        s2.synchronize()

        assert c1.tolist() == [2.0, 4.0]
        assert d1.tolist() == [6.0, 8.0]
        assert e1.tolist() == [3.0, 6.0]
        assert f1.tolist() == [9.0, 12.0]

    def test_many_streams(self):
        """创建多个 stream 并行使用不崩溃。"""
        streams = [Stream("cpu") for _ in range(4)]
        arrays = []

        for i, s in enumerate(streams):
            with ms.stream(s):
                arr = ms.array([float(i), float(i + 1)], dtype='f32')
                arrays.append(arr)

        for s in streams:
            s.synchronize()

        for i, arr in enumerate(arrays):
            assert arr.tolist() == [float(i), float(i + 1)]

    def test_stream_reuse_after_sync(self):
        """stream 同步后可继续复用。"""
        s = Stream("cpu")

        with ms.stream(s):
            a = ms.array([1.0, 2.0], dtype='f32')
            b = ms.add(a, a)
        s.synchronize()
        assert b.tolist() == [2.0, 4.0]

        # 复用同一 stream
        with ms.stream(s):
            c = ms.add(b, b)
        s.synchronize()
        assert c.tolist() == [4.0, 8.0]


# ============================================================
# MUSA 硬件多 Stream 测试
# ============================================================


@musa_required
class TestMultiStreamMusa:
    """MUSA GPU 上的多 stream 验证。"""

    def test_two_streams_musa(self):
        """两个 MUSA stream 上分别计算。"""
        ms.set_default_device("musa:0")
        s1 = Stream("musa:0")
        s2 = Stream("musa:0")

        with ms.stream(s1):
            a = ms.array([1.0, 2.0, 3.0], dtype='f32')
        with ms.stream(s2):
            b = ms.array([4.0, 5.0, 6.0], dtype='f32')

        s1.synchronize()
        s2.synchronize()
        assert a.tolist() == [1.0, 2.0, 3.0]
        assert b.tolist() == [4.0, 5.0, 6.0]

    def test_cross_stream_add_musa(self):
        """跨 MUSA stream 的 add 操作。"""
        ms.set_default_device("musa:0")
        s1 = Stream("musa:0")
        s2 = Stream("musa:0")

        with ms.stream(s1):
            a = ms.array([1.0, 2.0], dtype='f32')
        with ms.stream(s2):
            b = ms.array([3.0, 4.0], dtype='f32')
            c = ms.add(a, b)  # 跨 stream 输入

        s2.synchronize()
        assert c.tolist() == [4.0, 6.0]

    def test_interleaved_ops_musa(self):
        """MUSA 上交替 stream 操作不崩溃。"""
        ms.set_default_device("musa:0")
        s1 = Stream("musa:0")
        s2 = Stream("musa:0")

        with ms.stream(s1):
            a = ms.array([1.0, 2.0], dtype='f32')
        with ms.stream(s2):
            b = ms.array([10.0, 20.0], dtype='f32')
        with ms.stream(s1):
            c = ms.add(a, a)
        with ms.stream(s2):
            d = ms.add(b, b)

        s1.synchronize()
        s2.synchronize()
        assert c.tolist() == [2.0, 4.0]
        assert d.tolist() == [20.0, 40.0]
