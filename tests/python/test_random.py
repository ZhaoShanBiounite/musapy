"""v0.3 Phase 4 (P4.7): random 套件测试（rand/randn/uniform/normal/bernoulli）。

GPU-only（003-D4 修订，P4.6 CPU fallback 已取消）：CPU 设备调用抛
DeviceError；分布统计 / seed 复现性走真机 GPU（musa_required 门控），
mock 模式只跑形状与拒绝用例（mock stub 为确定性填充，见 musa_x_ffi.rs）。
"""

import pytest
import numpy as np
import musapy as ms

# GPU 探测（与 test_linalg.py 同模式；mock 模式下 Device("musa:0") 亦有效）
try:
    _dev = ms.Device("musa:0")
    _gpu_available = True
except Exception:
    _gpu_available = False

musa_required = pytest.mark.skipif(not _gpu_available, reason="MUSA device not available")


@musa_required
class TestRandomCpuRejected:
    """v0.3 GPU-only：CPU 设备调用必须拒绝（DeviceError）。"""

    def test_rand_cpu_rejected(self):
        with pytest.raises(ms.DeviceError):
            ms.random.rand(4, device="cpu")

    def test_randn_cpu_rejected(self):
        with pytest.raises(ms.DeviceError):
            ms.random.randn(4, device="cpu")

    def test_uniform_cpu_rejected(self):
        with pytest.raises(ms.DeviceError):
            ms.random.uniform(0.0, 1.0, shape=(4,), device="cpu")

    def test_normal_cpu_rejected(self):
        with pytest.raises(ms.DeviceError):
            ms.random.normal(shape=(4,), device="cpu")

    def test_bernoulli_cpu_rejected(self):
        with pytest.raises(ms.DeviceError):
            ms.random.bernoulli(shape=(4,), device="cpu")


@musa_required
class TestRandomShapes:
    """形状 / dtype 管线（mock 与真机均验证）。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    def test_rand_shape_matrix(self):
        for shape in [(4,), (2, 3), (3, 2, 2), (1,), (2, 0), (0, 3)]:
            a = ms.random.rand(*shape)
            assert a.shape == shape, f"rand{shape}: got {a.shape}"
            # 值域 [0, 1)
            if len(shape) == 1 and shape[0] > 0:
                vals = a.tolist()
                assert all(0.0 <= v < 1.0 for v in vals)

    def test_rand_tuple_shape_form(self):
        """rand((2, 3)) 与 rand(2, 3) 等价（random.py 归一化）。"""
        a = ms.random.rand((2, 3))
        b = ms.random.rand(2, 3)
        assert a.shape == b.shape == (2, 3)

    def test_rand_zero_dim(self):
        a = ms.random.rand()
        assert a.shape == ()

    def test_rand_dtype_matrix(self):
        a32 = ms.random.rand(8, dtype=ms.float32)
        a64 = ms.random.rand(8, dtype=ms.float64)
        assert a32.dtype == ms.float32
        assert a64.dtype == ms.float64

    def test_rand_bad_dtype_rejected(self):
        with pytest.raises(ms.DtypeError):
            ms.random.rand(8, dtype=ms.int64)

    def test_randn_shape_dtype(self):
        a = ms.random.randn(3, 4, dtype=ms.float64)
        assert a.shape == (3, 4)
        assert a.dtype == ms.float64

    def test_uniform_shape_and_defaults(self):
        a = ms.random.uniform(-1.0, 1.0, shape=(4, 4))
        assert a.shape == (4, 4)
        vals = np.array(a.tolist())
        assert vals.min() >= -1.0 and vals.max() < 1.0
        # 默认参数：uniform() = uniform(0, 1) → 0-dim
        assert ms.random.uniform().shape == ()

    def test_normal_shape_and_defaults(self):
        a = ms.random.normal(loc=5.0, scale=2.0, shape=(2, 3))
        assert a.shape == (2, 3)
        assert ms.random.normal().shape == ()

    def test_bernoulli_bool_output(self):
        a = ms.random.bernoulli(p=0.5, shape=(6,))
        assert a.shape == (6,)
        assert a.dtype == ms.bool_
        assert set(a.tolist()) <= {False, True}
        assert ms.random.bernoulli().shape == ()

    def test_bad_shape_rejected(self):
        with pytest.raises(TypeError):
            ms.random.rand(-1)


@musa_required
class TestRandomReproducibility:
    """seed 复现性（真机）：同 seed 紧邻两次逐元素相等；无 seed 两次不同。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    @pytest.mark.parametrize("dtype", [ms.float32, ms.float64])
    def test_rand_seed_reproducible(self, dtype):
        a1 = ms.random.rand(100, dtype=dtype, seed=42)
        a2 = ms.random.rand(100, dtype=dtype, seed=42)
        assert a1.tolist() == a2.tolist()

    @pytest.mark.parametrize("dtype", [ms.float32, ms.float64])
    def test_randn_seed_reproducible(self, dtype):
        a1 = ms.random.randn(100, dtype=dtype, seed=7)
        a2 = ms.random.randn(100, dtype=dtype, seed=7)
        assert a1.tolist() == a2.tolist()

    @pytest.mark.parametrize("dtype", [ms.float32, ms.float64])
    def test_uniform_seed_reproducible(self, dtype):
        a1 = ms.random.uniform(-2.0, 3.0, shape=(64,), dtype=dtype, seed=11)
        a2 = ms.random.uniform(-2.0, 3.0, shape=(64,), dtype=dtype, seed=11)
        assert a1.tolist() == a2.tolist()

    @pytest.mark.parametrize("dtype", [ms.float32, ms.float64])
    def test_normal_seed_reproducible(self, dtype):
        a1 = ms.random.normal(loc=2.0, scale=3.0, shape=(64,), dtype=dtype, seed=23)
        a2 = ms.random.normal(loc=2.0, scale=3.0, shape=(64,), dtype=dtype, seed=23)
        assert a1.tolist() == a2.tolist()

    def test_bernoulli_seed_reproducible(self):
        a1 = ms.random.bernoulli(p=0.5, shape=(64,), seed=5)
        a2 = ms.random.bernoulli(p=0.5, shape=(64,), seed=5)
        assert a1.tolist() == a2.tolist()

    def test_different_seed_differs(self):
        a1 = ms.random.rand(100, seed=1)
        a2 = ms.random.rand(100, seed=2)
        assert a1.tolist() != a2.tolist()

    def test_no_seed_differs(self):
        """无 seed：generator 自然推进，紧邻两次调用结果不同。"""
        a1 = ms.random.rand(100)
        a2 = ms.random.rand(100)
        assert a1.tolist() != a2.tolist()


@musa_required
class TestRandomDistribution:
    """分布统计（真机）：1e6 样本均值/方差在 3σ 容差内（验收标准）。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    N = 1_000_000

    def test_uniform_stats_f64(self):
        a = np.array(ms.random.rand(self.N, dtype=ms.float64, seed=1).tolist())
        assert abs(a.mean() - 0.5) < 0.01          # 3σ ≈ 8.7e-4
        assert abs(a.var() - 1.0 / 12.0) < 0.005   # 3σ ≈ 7.3e-4

    def test_uniform_stats_f32(self):
        a = np.array(ms.random.rand(self.N, dtype=ms.float32, seed=2).tolist())
        assert abs(a.mean() - 0.5) < 0.02
        assert abs(a.var() - 1.0 / 12.0) < 0.01

    def test_randn_stats_f64(self):
        a = np.array(ms.random.randn(self.N, dtype=ms.float64, seed=3).tolist())
        assert abs(a.mean()) < 0.01                # 3σ ≈ 3e-3
        assert abs(a.var() - 1.0) < 0.01           # 3σ ≈ 4.2e-3

    def test_randn_stats_f32(self):
        a = np.array(ms.random.randn(self.N, dtype=ms.float32, seed=4).tolist())
        assert abs(a.mean()) < 0.02
        assert abs(a.var() - 1.0) < 0.02

    def test_uniform_range_transform(self):
        """uniform(low, high)：值域 [low, high) 且均值 (low+high)/2。"""
        low, high = -3.0, 5.0
        a = np.array(
            ms.random.uniform(low, high, shape=(self.N,), dtype=ms.float64, seed=5).tolist()
        )
        assert a.min() >= low and a.max() < high
        assert abs(a.mean() - (low + high) / 2.0) < 0.01

    def test_normal_loc_scale(self):
        """normal(loc, scale)：均值 loc、方差 scale²（原生 mean/stddev）。"""
        loc, scale = 4.0, 2.0
        a = np.array(
            ms.random.normal(loc=loc, scale=scale, shape=(self.N,), dtype=ms.float64, seed=6).tolist()
        )
        assert abs(a.mean() - loc) < 0.02
        assert abs(a.var() - scale * scale) < 0.05

    def test_bernoulli_stats(self):
        a = np.array(ms.random.bernoulli(p=0.3, shape=(self.N,), seed=8).tolist())
        assert abs(a.mean() - 0.3) < 0.01
        assert set(np.unique(a)) <= {0.0, 1.0}
