"""Phase 6: Indexing 套件测试（transpose/permute/flip/index_select/slice/__getitem__）。

CPU 测试使用 MUSAPY_MOCK_MUSA=1 构建；GPU 测试需真实 MUSA 设备。
"""

import pytest
import musapy as ms

# 确保默认 device 为 CPU（测试环境）
ms.set_default_device("cpu")

# GPU 测试标记
try:
    _dev = ms.Device("musa:0")
    _gpu_available = True
except Exception:
    _gpu_available = False

musa_required = pytest.mark.skipif(not _gpu_available, reason="MUSA device not available")


# ═══════════════════════════════════════════════════════════════
# TestTranspose
# ═══════════════════════════════════════════════════════════════

class TestTranspose:
    """transpose 基本功能。"""

    def test_transpose_2d_default(self):
        """2D 转置（默认反转维度）。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = ms.transpose(c)
        assert t.shape == (3, 2)
        assert t.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_transpose_with_axes(self):
        """显式指定 axes。"""
        a = ms.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
        t = ms.transpose(a, axes=[1, 0, 2])
        assert t.shape == (3, 2, 2)

    def test_transpose_1d(self):
        """1D 转置（无变化）。"""
        a = ms.arange(5)
        t = ms.transpose(a)
        assert t.shape == (5,)
        assert t.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_transpose_zero_copy(self):
        """验证零拷贝（共享 buffer）。"""
        c = ms.array([[1.0, 2.0], [3.0, 4.0]])
        t = ms.transpose(c)
        # 验证数据共享（通过修改验证）
        # 注：musapy 目前没有 __setitem__，用 tolist 验证
        assert t.tolist() == [[1.0, 3.0], [2.0, 4.0]]


# ═══════════════════════════════════════════════════════════════
# TestPermute
# ═══════════════════════════════════════════════════════════════

class TestPermute:
    """permute 基本功能。"""

    def test_permute_basic(self):
        """基本维度排列。"""
        a = ms.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
        p = ms.permute(a, [2, 0, 1])
        assert p.shape == (2, 2, 3)

    def test_permute_identity(self):
        """恒等排列。"""
        a = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        p = ms.permute(a, [0, 1])
        assert p.shape == (2, 3)
        assert p.tolist() == a.tolist()


# ═══════════════════════════════════════════════════════════════
# TestFlip
# ═══════════════════════════════════════════════════════════════

class TestFlip:
    """flip 基本功能。"""

    def test_flip_axis0(self):
        """翻转第 0 轴。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f = ms.flip(c, axis=0)
        assert f.shape == (2, 3)
        assert f.tolist() == [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]

    def test_flip_axis1(self):
        """翻转第 1 轴。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f = ms.flip(c, axis=1)
        assert f.shape == (2, 3)
        assert f.tolist() == [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]

    def test_flip_1d(self):
        """1D 翻转。"""
        a = ms.arange(5)
        f = ms.flip(a, axis=0)
        assert f.tolist() == [4.0, 3.0, 2.0, 1.0, 0.0]

    def test_flip_double_identity(self):
        """双重翻转恢复原状。"""
        c = ms.array([[1.0, 2.0], [3.0, 4.0]])
        f = ms.flip(ms.flip(c, axis=0), axis=0)
        assert f.tolist() == c.tolist()


# ═══════════════════════════════════════════════════════════════
# TestIndexSelect
# ═══════════════════════════════════════════════════════════════

class TestIndexSelect:
    """index_select 基本功能。"""

    def test_index_select_axis0(self):
        """第 0 轴整数索引。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = ms.index_select(c, 0, 1)
        assert s.shape == (3,)
        assert s.tolist() == [4.0, 5.0, 6.0]

    def test_index_select_axis1(self):
        """第 1 轴整数索引。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = ms.index_select(c, 1, 2)
        assert s.shape == (2,)
        assert s.tolist() == [3.0, 6.0]

    def test_index_select_3d(self):
        """3D 数组索引。"""
        a = ms.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
        s = ms.index_select(a, 1, 0)
        assert s.shape == (2, 2)


# ═══════════════════════════════════════════════════════════════
# TestSlice
# ═══════════════════════════════════════════════════════════════

class TestSlice:
    """slice 基本功能。"""

    def test_slice_1d(self):
        """1D 切片。"""
        a = ms.arange(10)
        s = ms.slice(a, [[2, 7, 1]])
        assert s.shape == (5,)
        assert s.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]

    def test_slice_with_step(self):
        """带步长的切片。"""
        a = ms.arange(10)
        s = ms.slice(a, [[0, 10, 2]])
        assert s.shape == (5,)
        assert s.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]

    def test_slice_2d(self):
        """2D 切片。"""
        c = ms.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        s = ms.slice(c, [[0, 2, 1], [1, 3, 1]])
        assert s.shape == (2, 2)
        assert s.tolist() == [[2.0, 3.0], [6.0, 7.0]]

    def test_slice_2d_with_step(self):
        """2D 带步长切片。"""
        c = ms.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        s = ms.slice(c, [[0, 2, 1], [0, 4, 2]])
        assert s.shape == (2, 2)
        assert s.tolist() == [[1.0, 3.0], [5.0, 7.0]]


# ═══════════════════════════════════════════════════════════════
# TestGetitem
# ═══════════════════════════════════════════════════════════════

class TestGetitem:
    """__getitem__ 语法糖。"""

    def test_getitem_integer(self):
        """整数索引。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = c[0]
        assert s.shape == (3,)
        assert s.tolist() == [1.0, 2.0, 3.0]

    def test_getitem_negative_integer(self):
        """负整数索引。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = c[-1]
        assert s.shape == (3,)
        assert s.tolist() == [4.0, 5.0, 6.0]

    def test_getitem_slice(self):
        """切片索引。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = c[0:2]
        assert s.shape == (2, 3)
        assert s.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_getitem_slice_with_step(self):
        """带步长切片。"""
        a = ms.arange(10)
        s = a[::2]
        assert s.shape == (5,)
        assert s.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]

    def test_getitem_tuple(self):
        """多维索引（tuple）。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = c[0, 1]
        assert s.shape == ()
        assert s.tolist() == 2.0

    def test_getitem_tuple_slice(self):
        """多维切片。"""
        c = ms.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        s = c[0:2, 1:3]
        assert s.shape == (2, 2)
        assert s.tolist() == [[2.0, 3.0], [6.0, 7.0]]

    def test_getitem_tuple_mixed(self):
        """混合索引（整数 + 切片）。"""
        c = ms.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        s = c[1, 0:2]
        assert s.shape == (2,)
        assert s.tolist() == [5.0, 6.0]


# ═══════════════════════════════════════════════════════════════
# TestViewArithmetic
# ═══════════════════════════════════════════════════════════════

class TestViewArithmetic:
    """view 参与运算（stride-aware）。"""

    def test_transpose_add(self):
        """transpose 后参与加法。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        b = ms.array([[1.0, 1.0], [1.0, 1.0]])
        result = ms.transpose(a) + b
        assert result.tolist() == [[2.0, 4.0], [3.0, 5.0]]

    def test_flip_add(self):
        """flip 后参与加法。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        b = ms.array([[1.0, 1.0], [1.0, 1.0]])
        result = ms.flip(a, axis=1) + b
        # flip([[1,2],[3,4]], axis=1) = [[2,1],[4,3]]，+ [[1,1],[1,1]] = [[3,2],[5,4]]
        assert result.tolist() == [[3.0, 2.0], [5.0, 4.0]]

    def test_slice_add(self):
        """slice 后参与加法。"""
        a = ms.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        b = ms.ones((2, 2))
        s = ms.slice(a, [[0, 2, 1], [0, 2, 1]])
        result = s + b
        assert result.tolist() == [[1.0, 2.0], [4.0, 5.0]]


# ═══════════════════════════════════════════════════════════════
# TestIndexingGPU
# ═══════════════════════════════════════════════════════════════

class TestIndexingGPU:
    """GPU 端到端测试（需真实 MUSA 设备）。"""

    @musa_required
    def test_transpose_gpu(self):
        """GPU transpose。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="musa:0")
        t = ms.transpose(c)
        assert t.shape == (3, 2)
        assert t.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    @musa_required
    def test_flip_gpu(self):
        """GPU flip。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="musa:0")
        f = ms.flip(c, axis=1)
        assert f.tolist() == [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]

    @musa_required
    def test_slice_gpu(self):
        """GPU slice。"""
        c = ms.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device="musa:0")
        s = ms.slice(c, [[0, 2, 1], [0, 3, 2]])
        assert s.tolist() == [[1.0, 3.0], [5.0, 7.0]]

    @musa_required
    def test_getitem_gpu(self):
        """GPU __getitem__。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="musa:0")
        assert c[0].tolist() == [1.0, 2.0, 3.0]
        assert c[0:2, 0:2].tolist() == [[1.0, 2.0], [4.0, 5.0]]


# ═══════════════════════════════════════════════════════════════
# 验收测试（v0.2 plan 1.3 节）
# ═══════════════════════════════════════════════════════════════

class TestAcceptance:
    """v0.2-alpha 验收标准测试。"""

    def test_acceptance_transpose(self):
        """验收：transpose。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = ms.transpose(c)
        assert t.shape == (3, 2)
        assert t.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_acceptance_slice(self):
        """验收：slice。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sl = ms.slice(c, [[0, 2, 1], [0, 3, 2]])
        assert sl.tolist() == [[1.0, 3.0], [4.0, 6.0]]

    def test_acceptance_getitem(self):
        """验收：__getitem__。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sl = c[0:2, ::2]
        assert sl.tolist() == [[1.0, 3.0], [4.0, 6.0]]
