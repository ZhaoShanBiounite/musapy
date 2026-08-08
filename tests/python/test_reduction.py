"""Phase 4: Reduction 套件测试（sum/prod/max/min/mean/argmax/argmin/cumsum）。

CPU 测试使用 MUSAPY_MOCK_MUSA=1 构建；GPU 测试需真实 MUSA 设备。
"""

import math
import numpy as np
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
        a = ms.array([1, 2, 3], dtype='i64')
        result = ms.sum(a)
        assert result.dtype == 'i64'
        assert result.item() == 6

    def test_sum_int8_accumulates_i64(self):
        """int8 输入 sum → int64 输出（cast + 累加）。"""
        a = ms.array([1, 2, 3], dtype='i8')
        result = ms.sum(a)
        assert result.dtype == 'i64'
        assert result.item() == 6

    def test_mean_int_gives_float64(self):
        """整数输入 mean → float64 输出。"""
        a = ms.array([1, 2, 3, 4], dtype='i64')
        result = ms.mean(a)
        assert result.dtype == 'f64'
        assert result.item() == pytest.approx(2.5)

    def test_mean_f32_stays_f32(self):
        """float32 输入 mean → float32 输出。"""
        a = ms.array([1.0, 2.0, 3.0], dtype='f32')
        result = ms.mean(a)
        assert result.dtype == 'f32'

    def test_max_int_gives_i64(self):
        """整数输入 max → int64 输出（alpha 简化）。"""
        a = ms.array([3, 1, 4], dtype='i64')
        result = ms.max(a)
        assert result.dtype == 'i64'
        assert result.item() == 4

    def test_sum_f32_stays_f32(self):
        """float32 输入 sum → float32 输出。"""
        a = ms.array([1.0, 2.0, 3.0], dtype='f32')
        result = ms.sum(a)
        assert result.dtype == 'f32'


# ═══════════════════════════════════════════════════════════════
# TestArgReduce — argmax / argmin
# ═══════════════════════════════════════════════════════════════

class TestArgReduce:
    """argmax / argmin 测试。"""

    def test_argmax_global(self):
        a = ms.array([1.0, 5.0, 3.0, 2.0])
        result = ms.argmax(a)
        assert result.item() == 1
        assert result.dtype == 'i64'

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
        a = ms.array([10, 30, 20], dtype='i64')
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
        a = ms.array([1, 2, 3], dtype='i64')
        result = ms.cumsum(a, axis=0)
        assert result.dtype == 'i64'
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
        out = ms.array([0, 0], dtype='i64')  # wrong dtype for float sum
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
        a = ms.array([1, 2, 3], dtype='i8', device="musa:0")
        result = ms.sum(a)
        assert result.dtype == 'i64'
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

    # ── P0: cumsum 长 axis（分层扫描）──────────────────────────

    def test_gpu_cumsum_boundary_65536(self):
        """axis_len = 65536（blocks_per_row = 256，单级路径上限）。"""
        n = 65536
        a = ms.ones(n, dtype='i64', device="musa:0")
        result = ms.cumsum(a, axis=0)
        # cumsum(ones) = [1..n]：总和 + 末元素抽查
        assert ms.sum(result).item() == n * (n + 1) // 2
        idx = ms.array([n - 1], dtype='i64', device="musa:0")
        assert ms.gather(result, idx, axis=0).item() == n

    def test_gpu_cumsum_hierarchical_65537(self):
        """axis_len = 65537（blocks_per_row = 257，进入分层路径的首例）。"""
        n = 65537
        a = ms.ones(n, dtype='i64', device="musa:0")
        result = ms.cumsum(a, axis=0)
        assert ms.sum(result).item() == n * (n + 1) // 2
        idx = ms.array([n - 1], dtype='i64', device="musa:0")
        assert ms.gather(result, idx, axis=0).item() == n

    def test_gpu_cumsum_1m_i64(self):
        """axis_len = 1M（blocks_per_row = 3907；P0 报告中的 benchmark 规模）。"""
        n = 1_000_000
        a = ms.ones(n, dtype='i64', device="musa:0")
        result = ms.cumsum(a, axis=0)
        assert ms.sum(result).item() == n * (n + 1) // 2
        # 抽查：单级边界末位、分层路径首位、末元素
        idx = ms.array([65535, 65536, n - 1], dtype='i64', device="musa:0")
        assert ms.gather(result, idx, axis=0).tolist() == [65536, 65537, n]

    def test_gpu_cumsum_long_axis_random_f32(self):
        """100000 元素 f32 随机数据，对比 numpy（blocks_per_row = 391）。"""
        import numpy as np

        rng = np.random.default_rng(42)
        data = rng.random(100000, dtype=np.float32)
        a = ms.array(data.tolist(), dtype='f32', device="musa:0")
        result = ms.cumsum(a, axis=0)
        got = np.array(result.tolist(), dtype=np.float32)
        assert np.allclose(got, np.cumsum(data), rtol=1e-3, atol=1e-3)

    def test_gpu_cumsum_long_axis_f64(self):
        """100000 元素 f64，覆盖 f64 实例化。"""
        import numpy as np

        rng = np.random.default_rng(43)
        data = rng.random(100000, dtype=np.float64)
        a = ms.array(data.tolist(), dtype='f64', device="musa:0")
        result = ms.cumsum(a, axis=0)
        got = np.array(result.tolist(), dtype=np.float64)
        assert np.allclose(got, np.cumsum(data), rtol=1e-9, atol=1e-9)

    def test_gpu_cumsum_multirow_long_axis(self):
        """多行 × 长 axis：(3, 70000) axis=1（blocks_per_row = 274）。"""
        import numpy as np

        rng = np.random.default_rng(7)
        data = rng.random((3, 70000), dtype=np.float32)
        a = ms.array(data.tolist(), dtype='f32', device="musa:0")
        result = ms.cumsum(a, axis=1)
        got = np.array(result.tolist(), dtype=np.float32)
        assert got.shape == (3, 70000)
        assert np.allclose(got, np.cumsum(data, axis=1), rtol=1e-3, atol=1e-3)

    def test_gpu_cumsum_non_pow2_blocks_per_row(self):
        """blocks_per_row 非 2 的幂（bpr = 6/7）。

        旧 Phase 2 用带 guard 的 Blelloch 树按 bpr 扫描，非 2 的幂时
        产生错误前缀；修复后固定按 256 槽位（补 0）扫描。
        """
        import numpy as np

        for n in (1300, 1537):  # bpr = 6, 7
            rng = np.random.default_rng(n)
            data = rng.random(n, dtype=np.float32)
            a = ms.array(data.tolist(), dtype='f32', device="musa:0")
            result = ms.cumsum(a, axis=0)
            got = np.array(result.tolist(), dtype=np.float32)
            assert np.allclose(got, np.cumsum(data), rtol=1e-3, atol=1e-3), (
                f"n={n}"
            )

    def test_gpu_cumsum_axis_too_long_raises(self):
        """axis_len > 256^3 超出分层扫描容量，应明确报错。"""
        n = 256**3 + 1  # 16777217
        a = ms.ones(n, dtype='f32', device="musa:0")
        with pytest.raises(Exception):
            ms.cumsum(a, axis=0)

    def test_gpu_prod(self):
        a = ms.array([2.0, 3.0, 4.0], device="musa:0")
        result = ms.prod(a)
        assert result.item() == pytest.approx(24.0)

    # ── P2: 小 axis 并行（每输出多线程组）+ partial 增强 ──────

    def test_gpu_small_axis_256x256(self):
        """256×256 axis=0/1：naive(256 线程) → small_axis(G=256)。"""
        rng = np.random.default_rng(7)
        data = rng.random((256, 256), dtype=np.float32)
        a = ms.array(data.tolist(), device="musa:0")
        for op, wf in ((ms.sum, np.sum), (ms.max, np.max), (ms.min, np.min),
                       (ms.prod, np.prod), (ms.mean, np.mean)):
            for ax in (0, 1):
                got = op(a, axis=ax)
                np.testing.assert_allclose(np.array(got.tolist()),
                                           wf(data, axis=ax), rtol=1e-5, atol=1e-5)

    def test_gpu_small_axis_group_boundaries(self):
        """G 边界 32/64/128/256（axis_len=17/33/129/1024）+ 阈值两侧。"""
        rng = np.random.default_rng(8)
        for (r, c) in [(100, 17), (33, 33), (129, 129), (7, 1024), (1025, 5)]:
            data = rng.random((r, c), dtype=np.float32)
            a = ms.array(data.tolist(), device="musa:0")
            for op, wf in ((ms.sum, np.sum), (ms.max, np.max), (ms.min, np.min),
                           (ms.mean, np.mean)):
                for ax in (0, 1):
                    got = op(a, axis=ax)
                    np.testing.assert_allclose(np.array(got.tolist()),
                                               wf(data, axis=ax), rtol=1e-5, atol=1e-5)

    def test_gpu_small_axis_i64(self):
        """i64 小 axis（ReduceLimits 单位元路径）。"""
        rng = np.random.default_rng(9)
        data = rng.integers(-1000, 1000, (256, 256))
        a = ms.array(data.tolist(), dtype='i64', device="musa:0")
        for op, wf in ((ms.sum, np.sum), (ms.prod, np.prod),
                       (ms.max, np.max), (ms.min, np.min)):
            for ax in (0, 1):
                assert op(a, axis=ax).tolist() == wf(data, axis=ax).tolist()

    def test_gpu_reduce_1m_partial(self):
        """1M 全局归约（partial 新路径：ITEMS=4 + shuffle），对比 numpy。"""
        rng = np.random.default_rng(10)
        data = rng.random(1_000_000, dtype=np.float32)
        a = ms.array(data.tolist(), device="musa:0")
        for op, wf in ((ms.sum, np.sum), (ms.max, np.max), (ms.min, np.min),
                       (ms.mean, np.mean), (ms.prod, np.prod)):
            got = op(a)
            np.testing.assert_allclose(np.array(got.tolist()),
                                       wf(data), rtol=1e-5, atol=1e-5)

    def test_gpu_reduce_axis_5000x3(self):
        """axis_len=5000（partial 路径）+ out_size=3 的轴归约。"""
        rng = np.random.default_rng(11)
        data = rng.random((5000, 3), dtype=np.float32)
        a = ms.array(data.tolist(), device="musa:0")
        for op, wf in ((ms.sum, np.sum), (ms.max, np.max), (ms.min, np.min)):
            for ax in (0, 1):
                got = op(a, axis=ax)
                np.testing.assert_allclose(np.array(got.tolist()),
                                           wf(data, axis=ax), rtol=1e-5, atol=1e-5)

    def test_gpu_reduce_strided_view(self):
        """非连续视图（flip，stride 为负）走小 axis/partial 路径。"""
        rng = np.random.default_rng(12)
        base = rng.random((64, 64), dtype=np.float32)
        a = ms.array(base.tolist(), device="musa:0")
        fv = ms.flip(a, axis=1)
        for op, wf in ((ms.sum, np.sum), (ms.max, np.max), (ms.min, np.min)):
            for ax in (0, 1):
                got = op(fv, axis=ax)
                np.testing.assert_allclose(np.array(got.tolist()),
                                           wf(base[:, ::-1], axis=ax),
                                           rtol=1e-5, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# Phase 7 (P7.1/P7.2): axis=tuple 多轴归约 + 复数 sum/mean/prod
# ═══════════════════════════════════════════════════════════════

class TestReductionMultiAxis:
    """axis=tuple 多轴归约（P7.1）：sum/prod/max/min/mean 逐轴迭代，
    argmax/argmin transpose+合并轴。CPU + MUSA 双路径。"""

    @pytest.mark.parametrize("dtype", ['f32', 'f64'])
    @pytest.mark.parametrize("axes", [(0, 1), (1, 2), (0, 2), (0, 1, 2), (-1, 0)])
    def test_multi_axis_reduce_ops(self, dtype, axes):
        rng = np.random.default_rng(21)
        data = rng.normal(size=(2, 3, 4)).astype(
            np.float32 if dtype == 'f32' else np.float64
        )
        tol = 1e-4 if dtype == 'f32' else 1e-10
        for dev in ("cpu", "musa:0"):
            x = ms.array(data.tolist(), dtype=dtype, device=dev)
            for name, f in [("sum", ms.sum), ("mean", ms.mean), ("prod", ms.prod),
                            ("max", ms.max), ("min", ms.min)]:
                for keep in (False, True):
                    got = np.array(f(x, axis=axes, keepdims=keep).tolist())
                    exp = getattr(np, name)(data, axis=axes, keepdims=keep)
                    assert np.allclose(got, exp, rtol=tol, atol=tol), \
                        (dev, name, axes, keep, got, exp)

    @pytest.mark.parametrize("axes", [(0, 1), (1, 2), (0, 2), (0, 1, 2), (-1, 0)])
    def test_multi_axis_argreduce(self, axes):
        """argmax/argmin 多轴：transpose+合并轴，索引为展平指定轴的扁平索引
        （NumPy 2.0+ 语义）。"""
        rng = np.random.default_rng(22)
        data = rng.normal(size=(2, 3, 4))
        # 参照：指定轴移到末尾、合并为单轴后 argmax
        norm = tuple(sorted(ax % 3 for ax in axes))
        perm = [i for i in range(3) if i not in norm] + list(norm)
        t = data.transpose(perm)
        merged = t.reshape(t.shape[:3 - len(norm)] + (-1,))
        exp = np.argmax(merged, axis=-1)
        for dev in ("cpu", "musa:0"):
            x = ms.array(data.tolist(), dtype='f64', device=dev)
            got = np.array(ms.argmax(x, axis=axes).tolist())
            assert np.allclose(got, exp), (dev, axes, got, exp)

    def test_multi_axis_argreduce_keepdims(self):
        """argmax 多轴 keepdims：被归约轴处恢复为 1。"""
        rng = np.random.default_rng(23)
        data = rng.normal(size=(2, 3, 4))
        x = ms.array(data.tolist(), dtype='f64')
        got = ms.argmax(x, axis=(0, 1), keepdims=True)
        assert got.shape == (1, 1, 4)
        # 参照：合并轴 argmax 后 reshape 回 (1,1,4)
        t = data.transpose(2, 0, 1)  # 非轴维 2 在前，轴 0,1 在末尾
        merged = t.reshape(4, -1)
        exp = np.argmax(merged, axis=-1).reshape(1, 1, 4)
        assert np.allclose(got.tolist(), exp), (got.tolist(), exp)

    def test_multi_axis_errors(self):
        x = ms.array(np.zeros((2, 3, 4)).tolist(), dtype='f64')
        with pytest.raises(ms.ShapeError):
            ms.sum(x, axis=(0, 0))  # 重复轴
        with pytest.raises(ms.ShapeError):
            ms.sum(x, axis=(0, 5))  # 越界

    def test_multi_axis_out_rejected(self):
        """多轴 + out= 暂不支持（中间轮 shape 不同）。"""
        x = ms.array(np.zeros((2, 3)).tolist(), dtype='f64')
        out = ms.array(np.zeros(2).tolist(), dtype='f64')
        with pytest.raises(ms.ShapeError):
            ms.sum(x, axis=(0, 1), out=out)


class TestReductionComplex:
    """复数 sum/mean/prod（P7.2）；max/min/argmax/argmin 拒绝。CPU + MUSA。"""

    @pytest.mark.parametrize("dtype", ['c64', 'c128'])
    def test_complex_reduce_ops(self, dtype):
        rng = np.random.default_rng(24)
        data = (rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))).astype(
            np.complex64 if dtype == 'c64' else np.complex128
        )
        tol = 1e-4 if dtype == 'c64' else 1e-10
        for dev in ("cpu", "musa:0"):
            x = ms.array(data.tolist(), dtype=dtype, device=dev)
            for axes in (None, 0, 1, (0, 1), -1):
                for name, f in [("sum", ms.sum), ("mean", ms.mean), ("prod", ms.prod)]:
                    got = np.array(f(x, axis=axes).tolist(), dtype=complex)
                    exp = getattr(np, name)(data, axis=axes)
                    assert np.allclose(got, exp, rtol=tol, atol=tol), \
                        (dev, name, axes, got, exp)

    def test_complex_ordering_rejected(self):
        """复数 max/min/argmax/argmin 抛 DtypeError（复数无全序）。"""
        x = ms.array(np.array([1 + 2j, 3 + 4j]).tolist(), dtype='c128')
        for f in (ms.max, ms.min, ms.argmax, ms.argmin):
            with pytest.raises(ms.DtypeError):
                f(x)

    def test_complex_global_and_keepdims(self):
        x = ms.array(np.array([1 + 2j, 3 + 4j]).tolist(), dtype='c128')
        assert ms.sum(x).dtype == 'c128'
        assert ms.sum(x).item() == 4 + 6j
        got = ms.mean(x, axis=0, keepdims=True)
        assert got.shape == (1,)
        assert np.allclose(got.tolist(), [2 + 3j], rtol=1e-10)
