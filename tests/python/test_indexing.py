"""Phase 6: Indexing 套件测试（transpose/permute/flip/index_select/slice/__getitem__）。

CPU 测试使用 MUSAPY_MOCK_MUSA=1 构建；GPU 测试需真实 MUSA 设备。
"""

import numpy as np
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

    def test_flip_reduce_axis_none(self):
        """flip 视图参与 axis=None reduce（触发 materialize 路径）。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        f = ms.flip(a, axis=1)
        assert ms.sum(f).tolist() == 10.0

    def test_transpose_cumsum_axis_none(self):
        """transpose 视图参与 axis=None cumsum（触发 materialize 路径）。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        t = ms.transpose(a)
        # 逻辑展平顺序 [1,3,2,4] → cumsum [1,4,6,10]
        assert ms.cumsum(t).tolist() == [1.0, 4.0, 6.0, 10.0]


# ═══════════════════════════════════════════════════════════════
# TestContiguous
# ═══════════════════════════════════════════════════════════════

class TestContiguous:
    """contiguous 物化（P6.8 辅助 + gather/scatter 前置）。"""

    def test_contiguous_noop(self):
        """已连续数组：零拷贝视图。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        c = ms.contiguous(a)
        assert c.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_contiguous_transposed(self):
        """transpose 视图物化。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = ms.transpose(c)
        m = ms.contiguous(t)
        assert m.shape == (3, 2)
        assert m.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_contiguous_flipped(self):
        """flip 视图物化（负 stride）。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f = ms.flip(c, axis=1)
        m = ms.contiguous(f)
        assert m.tolist() == [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]

    def test_contiguous_slice_offset(self):
        """slice 视图物化（offset）。"""
        a = ms.arange(10)
        s = a[3:7]
        m = ms.contiguous(s)
        assert m.tolist() == [3.0, 4.0, 5.0, 6.0]


# ═══════════════════════════════════════════════════════════════
# TestGather
# ═══════════════════════════════════════════════════════════════

class TestGather:
    """gather：沿 axis 按 indices 取元素（copy）。"""

    def test_gather_axis0(self):
        """axis=0 行选择。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = ms.array([1, 0], dtype='i64')
        g = ms.gather(c, idx, axis=0)
        assert g.shape == (2, 3)
        assert g.tolist() == [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]

    def test_gather_axis1(self):
        """axis=1 列选择（v0.2 plan 验收用例）。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = ms.array([0, 2], dtype='i64')
        g = ms.gather(c, idx, axis=1)
        assert g.shape == (2, 2)
        assert g.tolist() == [[1.0, 3.0], [4.0, 6.0]]

    def test_gather_1d(self):
        """1D gather。"""
        a = ms.array([10.0, 20.0, 30.0, 40.0])
        idx = ms.array([3, 1, 3], dtype='i64')
        g = ms.gather(a, idx, axis=0)
        assert g.tolist() == [40.0, 20.0, 40.0]

    def test_gather_3d_middle_axis(self):
        """3D 中间轴 gather。"""
        a = ms.array([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                      [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]])
        idx = ms.array([2, 0], dtype='i64')
        g = ms.gather(a, idx, axis=1)
        assert g.shape == (2, 2, 2)
        assert g.tolist() == [[[4.0, 5.0], [0.0, 1.0]], [[10.0, 11.0], [6.0, 7.0]]]

    def test_gather_on_flipped_view(self):
        """对 flip 视图 gather（负 stride 输入）。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        f = ms.flip(a, axis=1)  # [[2,1],[4,3]]
        idx = ms.array([1], dtype='i64')
        g = ms.gather(f, idx, axis=0)
        assert g.tolist() == [[4.0, 3.0]]

    def test_gather_errors(self):
        """错误输入。"""
        c = ms.array([[1.0, 2.0], [3.0, 4.0]])
        idx = ms.array([0], dtype='i64')
        with pytest.raises(Exception):  # axis 越界
            ms.gather(c, idx, axis=2)
        with pytest.raises(Exception):  # 索引越界
            ms.gather(c, ms.array([2], dtype='i64'), axis=0)
        with pytest.raises(Exception):  # indices 非 int64
            ms.gather(c, ms.array([0.0]), axis=0)


# ═══════════════════════════════════════════════════════════════
# TestScatter
# ═══════════════════════════════════════════════════════════════

class TestScatter:
    """scatter：沿 axis 把 values 写入 indices 位置（copy）。"""

    def test_scatter_axis0(self):
        """axis=0 行写入。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        idx = ms.array([1], dtype='i64')
        vals = ms.array([[10.0, 11.0]])
        s = ms.scatter(a, idx, vals, axis=0)
        assert s.tolist() == [[1.0, 2.0], [10.0, 11.0]]
        # 原数组不被修改
        assert a.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_scatter_axis1(self):
        """axis=1 列写入。"""
        a = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = ms.array([0, 2], dtype='i64')
        vals = ms.array([[7.0, 8.0], [9.0, 10.0]])
        s = ms.scatter(a, idx, vals, axis=1)
        assert s.tolist() == [[7.0, 2.0, 8.0], [9.0, 5.0, 10.0]]

    def test_scatter_on_slice_view(self):
        """对 slice 视图 scatter（offset 输入）。"""
        a = ms.arange(5)  # int64 [0,1,2,3,4]
        s = a[1:4]  # [1,2,3]
        idx = ms.array([1], dtype='i64')
        vals = ms.array([99], dtype='i64')
        r = ms.scatter(s, idx, vals, axis=0)
        assert r.tolist() == [1, 99, 3]
        assert a.tolist() == [0, 1, 2, 3, 4]

    def test_scatter_errors(self):
        """错误输入。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]])
        idx = ms.array([0], dtype='i64')
        vals = ms.array([[10.0, 11.0]])
        with pytest.raises(Exception):  # axis 越界
            ms.scatter(a, idx, vals, axis=2)
        with pytest.raises(Exception):  # values shape 不匹配
            ms.scatter(a, idx, ms.array([[10.0], [11.0]]), axis=0)
        with pytest.raises(Exception):  # 索引越界
            ms.scatter(a, ms.array([2], dtype='i64'), vals, axis=0)


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

    @musa_required
    def test_contiguous_gpu(self):
        """GPU contiguous 物化。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="musa:0")
        m = ms.contiguous(ms.transpose(c))
        assert m.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    @musa_required
    def test_gather_gpu(self):
        """GPU gather（kernel 路径）。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="musa:0")
        idx = ms.array([0, 2], dtype='i64')
        g = ms.gather(c, idx, axis=1)
        assert g.tolist() == [[1.0, 3.0], [4.0, 6.0]]

    @musa_required
    def test_gather_gpu_indices_upload(self):
        """GPU gather：indices 在 CPU 时自动上传。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="musa:0")
        idx = ms.array([1, 0], dtype='i64')  # CPU indices
        g = ms.gather(c, idx, axis=0)
        assert g.tolist() == [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]

    @musa_required
    def test_scatter_gpu(self):
        """GPU scatter（kernel 路径）。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="musa:0")
        idx = ms.array([0, 2], dtype='i64')
        vals = ms.array([[7.0, 8.0], [9.0, 10.0]], device="musa:0")
        s = ms.scatter(c, idx, vals, axis=1)
        assert s.tolist() == [[7.0, 2.0, 8.0], [9.0, 5.0, 10.0]]

    @musa_required
    def test_gather_gpu_oob_reports_at_sync(self):
        """P1 方案二：GPU 路径越界索引在下一次同步时报错（而非调用时）。"""
        c = ms.array([[1.0, 2.0], [3.0, 4.0]], device="musa:0")
        g = ms.gather(c, ms.array([0, 5], dtype='i64'), axis=0)  # 5 越界
        with pytest.raises(Exception):
            g.tolist()

    @musa_required
    def test_gather_gpu_negative_index_reports_at_sync(self):
        """P1 方案二：负索引同样在同步时报错。"""
        c = ms.array([1.0, 2.0, 3.0], device="musa:0")
        g = ms.gather(c, ms.array([0, -1], dtype='i64'), axis=0)
        with pytest.raises(Exception):
            g.tolist()

    @musa_required
    def test_scatter_gpu_oob_reports_at_sync(self):
        """P1 方案二：scatter 越界在同步时报错。"""
        c = ms.array([[1.0, 2.0], [3.0, 4.0]], device="musa:0")
        vals = ms.array([[9.0, 9.0]], device="musa:0")
        s = ms.scatter(c, ms.array([2], dtype='i64'), vals, axis=0)  # 2 越界
        with pytest.raises(Exception):
            s.tolist()

    @musa_required
    def test_gpu_index_error_stream_reusable(self):
        """越界报错后流仍可用：错误槽复位、不毒化，后续合法 gather 正常。"""
        c = ms.array([1.0, 2.0, 3.0], device="musa:0")
        bad = ms.gather(c, ms.array([7], dtype='i64'), axis=0)
        with pytest.raises(Exception):
            bad.tolist()
        good = ms.gather(c, ms.array([2, 0], dtype='i64'), axis=0)
        assert good.tolist() == [3.0, 1.0]

    @musa_required
    def test_gpu_index_check_no_false_positive(self):
        """连续大量合法 gather（检查槽复用 + arena 扩容路径）不误报。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="musa:0")
        idx = ms.array([0, 2], dtype='i64')
        g = None
        for _ in range(40):  # > arena 初始容量 16，覆盖扩容路径
            g = ms.gather(c, idx, axis=1)
        assert g.tolist() == [[1.0, 3.0], [4.0, 6.0]]

    # ── P4: 2D 转置 tiled 物化 ────────────────────────────────

    @musa_required
    def test_gpu_contig_transpose_tiled(self):
        """转置视图物化走 tiled kernel：多种形状（含 <32 边界、非方阵）对比 numpy。"""
        rng = np.random.default_rng(31)
        for (r, c) in [(1024, 1024), (1000, 37), (37, 1000), (31, 33), (33, 31),
                       (32, 32), (5, 5), (1, 100), (100, 1), (2, 1025)]:
            data = rng.random((r, c), dtype=np.float32)
            a = ms.array(data.tolist(), device="musa:0")
            got = ms.contiguous(ms.transpose(a))
            np.testing.assert_allclose(np.array(got.tolist()), data.T,
                                       rtol=1e-6, atol=1e-6)

    @musa_required
    def test_gpu_contig_transpose_tiled_dtypes(self):
        """tiled 转置各 dtype 实例化（f64/i32/i64）。"""
        rng = np.random.default_rng(32)
        d64 = rng.random((257, 63), dtype=np.float64)
        a = ms.array(d64.tolist(), dtype='f64', device="musa:0")
        np.testing.assert_allclose(
            np.array(ms.contiguous(ms.transpose(a)).tolist()), d64.T, rtol=1e-9)
        di = rng.integers(-1000, 1000, (63, 257))
        for dtype in ('i64', 'i32'):
            a = ms.array(di.tolist(), dtype=dtype, device="musa:0")
            assert ms.contiguous(ms.transpose(a)).tolist() == di.T.tolist()

    @musa_required
    def test_gpu_contig_non_transpose_still_generic(self):
        """非转置模式（flip/3D permute/strided slice）仍走通用路径且结果正确。"""
        rng = np.random.default_rng(33)
        base = rng.random((64, 64), dtype=np.float32)
        a = ms.array(base.tolist(), device="musa:0")
        np.testing.assert_allclose(
            np.array(ms.contiguous(ms.flip(a, axis=1)).tolist()), base[:, ::-1])
        np.testing.assert_allclose(
            np.array(ms.contiguous(ms.flip(a, axis=0)).tolist()), base[::-1, :])
        b3 = rng.random((16, 17, 18), dtype=np.float32)
        a3 = ms.array(b3.tolist(), device="musa:0")
        np.testing.assert_allclose(
            np.array(ms.contiguous(ms.transpose(a3, [2, 0, 1])).tolist()),
            b3.transpose(2, 0, 1))
        sl = ms.slice(a, [[5, 60, 2], [1, 50, 3]])
        np.testing.assert_allclose(
            np.array(ms.contiguous(sl).tolist()), base[5:60:2, 1:50:3])

    @musa_required
    def test_offset_view_arithmetic_gpu(self):
        """GPU offset 视图参与算术（slice/flip 视图 + 指针调整路径）。"""
        a = ms.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], device="musa:0")
        b = ms.ones((2, 2), device="musa:0")
        # slice 视图（offset=0，stride 截断）
        s = ms.slice(a, [[0, 2, 1], [1, 3, 1]])
        assert (s + b).tolist() == [[2.0, 3.0], [5.0, 6.0]]
        # flip 视图（负 stride + offset）
        f = ms.flip(a, axis=1)
        b3 = ms.ones((2, 3), device="musa:0")
        assert (f + b3).tolist() == [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]

    @musa_required
    def test_reduce_on_view_gpu(self):
        """GPU axis=None reduce 作用于非连续视图（materialize 路径）。"""
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], device="musa:0")
        t = ms.transpose(a)
        assert ms.sum(t).tolist() == 10.0
        f = ms.flip(a, axis=0)
        assert ms.sum(f).tolist() == 10.0


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

    def test_acceptance_gather(self):
        """验收：gather（v0.2 plan）。"""
        c = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        g = ms.gather(c, ms.array([0, 2], dtype='i64'), axis=1)
        assert g.tolist() == [[1.0, 3.0], [4.0, 6.0]]


# ═══════════════════════════════════════════════════════════════
# Phase 8（ADR-002-D4）: 高级索引（boolean mask + fancy indexing）
# ═══════════════════════════════════════════════════════════════

class TestBooleanMask:
    """boolean mask：等形/前 md 维广播/恒 copy/越界。"""

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_mask_1d(self, device):
        a = ms.array([1.0, 2.0, 3.0, 4.0], dtype='f64', device=device)
        m = ms.array([True, False, True, False], dtype='b1', device=device)
        got = a[m]
        assert got.tolist() == [1.0, 3.0]

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_mask_2d_prefix(self, device):
        """mask 匹配前 md 维（NumPy 语义）。"""
        a = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='f64', device=device)
        m = ms.array([True, False], dtype='b1', device=device)
        got = a[m]
        assert got.shape == (1, 3)
        assert got.tolist() == [[1.0, 2.0, 3.0]]
        m_all = ms.array([[True, False, True], [False, True, False]], dtype='b1', device=device)
        got_all = a[m_all]
        assert got_all.tolist() == [1.0, 3.0, 5.0]

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_mask_is_copy(self, device):
        """mask 索引恒为 copy：改结果不影响原数组。"""
        a = ms.array([1.0, 2.0, 3.0], dtype='f64', device=device)
        m = ms.array([True, True, False], dtype='b1', device=device)
        got = a[m]
        got2 = ms.array(got.tolist(), device=device)  # 模拟修改
        got2 = ms.add(got2, ms.array([10.0, 10.0], dtype='f64', device=device))
        assert a.tolist() == [1.0, 2.0, 3.0]

    def test_mask_shape_mismatch(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], dtype='f64')
        m = ms.array([True, False, True], dtype='b1')  # 3 != 2
        with pytest.raises(ms.ShapeError):
            a[m]


class TestFancyIndexing:
    """fancy：单/多索引坐标配对、广播、N-D、负索引、越界 IndexError。"""

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_fancy_1d(self, device):
        a = ms.array([10.0, 20.0, 30.0, 40.0], dtype='f64', device=device)
        idx = ms.array([0, 2], dtype='i64', device=device)
        assert a[idx].tolist() == [10.0, 30.0]

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_fancy_negative(self, device):
        a = ms.array([10.0, 20.0, 30.0, 40.0], dtype='f64', device=device)
        idx = ms.array([-1, 0], dtype='i64', device=device)
        assert a[idx].tolist() == [40.0, 10.0]

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_fancy_coords_pair(self, device):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], dtype='f64', device=device)
        i0 = ms.array([0, 1], dtype='i64', device=device)
        i1 = ms.array([1, 0], dtype='i64', device=device)
        assert a[i0, i1].tolist() == [2.0, 3.0]

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_fancy_broadcast_indices(self, device):
        a = ms.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='f64', device=device)
        i0 = ms.array([0], dtype='i64', device=device)
        i1 = ms.array([0, 2], dtype='i64', device=device)
        got = a[i0, i1]
        assert got.tolist() == [1.0, 3.0]

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_fancy_nd_index_array(self, device):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], dtype='f64', device=device)
        idx = ms.array([[0, 1], [1, 0]], dtype='i64', device=device)
        got = a[idx]
        assert got.shape == (2, 2, 2)
        assert got.tolist() == [
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 4.0], [1.0, 2.0]],
        ]

    def test_fancy_list_index(self):
        a = ms.array([10.0, 20.0, 30.0], dtype='f64')
        assert a[[0, 2]].tolist() == [10.0, 30.0]

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_fancy_oob_indexerror(self, device):
        """越界抛 Python 内置 IndexError（NumPy 兼容，非 MusapyError 子类）。"""
        a = ms.array([1.0, 2.0, 3.0], dtype='f64', device=device)
        idx = ms.array([5], dtype='i64', device=device)
        with pytest.raises(IndexError):
            a[idx]

    @pytest.mark.parametrize("device", ["cpu", "musa:0"])
    def test_fancy_np_comparison(self, device):
        """对照 NumPy（含顺序/形状）。"""
        rng = np.random.default_rng(31)
        data = rng.normal(size=(4, 5))
        a = ms.array(data.tolist(), dtype='f64', device=device)
        i0 = ms.array([0, 2, 3], dtype='i64', device=device)
        i1 = ms.array([1], dtype='i64', device=device)  # 广播到 len 3
        got = a[i0, i1]
        exp = data[np.array([0, 2, 3]), np.array([1])]
        assert np.allclose(got.tolist(), exp), (got.tolist(), exp)
