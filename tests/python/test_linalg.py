"""v0.3 Phase 1 (P1.7): MUSA-X 数学库句柄生命周期冒烟测试。

验证内容（ADR-003 003-D2）：
  - 4 库（muBLAS/muRAND/muFFT/muSPARSE）版本查询在真机上可走通；
  - 句柄「懒创建 → SetStream → evict → 延迟销毁」闭环不泄漏
    （mem_stats 持平、延迟销毁队列归零）。

需真实 MUSA 设备；无 GPU 环境自动 skip（与 test_indexing.py 同模式）。
注：冒烟入口 `_core._math_handle_smoke` 仅测试用，不在公开 API。
"""

import pytest
import musapy as ms
from musapy import _core

# GPU 探测（与 test_indexing.py 一致）
try:
    _dev = ms.Device("musa:0")
    _gpu_available = True
except Exception:
    _gpu_available = False

musa_required = pytest.mark.skipif(not _gpu_available, reason="MUSA device not available")


@musa_required
class TestMathHandleSmoke:
    """MUSA-X 句柄生命周期冒烟。"""

    def test_library_versions(self):
        """4 库版本查询成功，版本号为正（SDK 3.1.0 → 30100 格式）。"""
        r = _core._math_handle_smoke(device="musa:0", iters=0)
        for lib in ("mublas", "murand", "mufft", "musparse"):
            assert r["versions"][lib] > 0, f"{lib} version should be positive"

    def test_handle_cycle_mem_flat(self):
        """1e3 次句柄创建/销毁循环后 mem_stats 持平、销毁队列归零。"""
        before = _core._math_handle_smoke(device="musa:0", iters=0)
        r = _core._math_handle_smoke(device="musa:0", iters=1000)

        # 延迟销毁队列必须清空（synchronize 触发 reclaim_destroys）
        assert r["pending_destroys_after"] == 0

        # musapy 记账的设备内存回到循环前水平（无泄漏）
        assert r["mem_allocated_bytes_after"] == r["mem_allocated_bytes_before"]
        assert r["mem_allocated_buffers_after"] == r["mem_allocated_buffers_before"]
        # deferred-free 队列不残留
        assert r["mem_cached_bytes_after"] == 0
        _ = before  # iters=0 仅用于显式首次初始化

    def test_handle_cycle_vram_flat(self):
        """句柄 create/destroy 循环不应累积驱动级显存（VRAM 无净增长）。

        VRAM 基线取在首次懒创建之后，故循环后空闲显存应 ≥ 基线
        （允许少量驱动/碎片波动容差）。
        """
        r = _core._math_handle_smoke(device="musa:0", iters=1000)
        vb, va = r["vram_free_bytes_before"], r["vram_free_bytes_after"]
        if vb is None or va is None:
            pytest.skip("VRAM info unavailable on this device")
        # 容差 64 MiB：覆盖驱动缓存/碎片等小幅波动；真泄漏远超此量级
        assert va >= vb - 64 * 1024 * 1024, (
            f"VRAM leaked during handle cycle: free {vb} -> {va} "
            f"(delta {vb - va} bytes)"
        )

    def test_cpu_device_rejected(self):
        """CPU 设备无 MUSA-X 句柄，应抛 DeviceError。"""
        with pytest.raises(ms.DeviceError):
            _core._math_handle_smoke(device="cpu", iters=1)


# ============================================================
# v0.3 Phase 2 (P2.7): matmul / dot / solve 验收（ADR-003 003-D3/D6）
#
# v0.3 策略修订（003-D4）：数学库算子 GPU-only，无 CPU fallback——
# CPU 设备上调用抛 DeviceError（TestLinalgCpuRejected）；
# 数值对照全部走真机 GPU（TestLinalgGpu，musa_required 门控）。
# ============================================================

import numpy as np


class TestLinalgCpuRejected:
    """v0.3 GPU-only：CPU 设备输入必须拒绝（DeviceError）。"""

    def test_matmul_cpu_rejected(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        b = ms.array([[5.0, 6.0], [7.0, 8.0]], device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.matmul(a, b)

    def test_dot_cpu_rejected(self):
        a = ms.array([1.0, 2.0, 3.0], device="cpu")
        b = ms.array([4.0, 5.0, 6.0], device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.dot(a, b)

    def test_solve_cpu_rejected(self):
        a = ms.array([[1.0, 0.0], [0.0, 1.0]], device="cpu")
        b = ms.array([1.0, 2.0], device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.solve(a, b)

    def test_matmul_cross_device_rejected(self):
        """musa×cpu 混合输入：device mismatch 报 DeviceError。"""
        a_cpu = ms.array([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        b_cpu = ms.array([[5.0, 6.0], [7.0, 8.0]], device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.matmul(a_cpu, b_cpu)



class TestLinalgGpu:
    """GPU 真机数值对照（gemm 转置技巧 / getrf+getrs 全链路）。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    def test_matmul_f64(self):
        rng = np.random.default_rng(11)
        A = rng.normal(size=(5, 3))
        B = rng.normal(size=(3, 4))
        got = ms.matmul(
            ms.array(A.tolist(), dtype=ms.float64),
            ms.array(B.tolist(), dtype=ms.float64),
        )
        assert np.allclose(got.tolist(), A @ B, atol=1e-10)

    def test_matmul_f32_non_square(self):
        rng = np.random.default_rng(13)
        A = rng.normal(size=(3, 7))
        B = rng.normal(size=(7, 2))
        got = ms.matmul(
            ms.array(A.tolist(), dtype=ms.float32),
            ms.array(B.tolist(), dtype=ms.float32),
        )
        assert np.allclose(got.tolist(), A @ B, atol=1e-5)

    def test_matmul_1d(self):
        v = ms.array([1.0, 2.0, 3.0], dtype=ms.float64)
        m = ms.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=ms.float64)
        r = ms.matmul(m, v)
        assert np.allclose(r.tolist(), [7.0, 11.0])

    def test_dot(self):
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float64)
        b = ms.array([4.0, 5.0, 6.0], dtype=ms.float64)
        assert abs(ms.dot(a, b).item() - 32.0) < 1e-10

    def test_solve_f64(self):
        rng = np.random.default_rng(17)
        A = rng.normal(size=(5, 5))
        B = rng.normal(size=(5, 2))
        x = ms.solve(
            ms.array(A.tolist(), dtype=ms.float64),
            ms.array(B.tolist(), dtype=ms.float64),
        )
        assert np.allclose(x.tolist(), np.linalg.solve(A, B), atol=1e-8)

    def test_solve_f32_singular(self):
        a = ms.array([[1.0, 2.0], [2.0, 4.0]], dtype=ms.float32)
        b = ms.array([1.0, 2.0], dtype=ms.float32)
        with pytest.raises(ms.LinAlgError):
            ms.solve(a, b)

    def test_solve_large_n128(self):
        """n≥128 回归：muSOLVER 3.1.0 不写 getrf info（SDK 缺陷，
        奇异检测走 LU 对角 D2H，2026-08-07 真机 C 探针实锤）。
        A = J + I 解析解 x = 1/(n+1)。"""
        n = 128
        a = ms.add(ms.full([n, n], 1.0, dtype=ms.float64), ms.eye(n, dtype=ms.float64))
        b = ms.ones([n], dtype=ms.float64)
        x = ms.solve(a, b)
        exp = 1.0 / (n + 1.0)
        assert np.allclose(x.tolist(), np.full(n, exp), atol=1e-9)

        # 大奇异矩阵（末行 = 首行）仍须检出
        big = np.ones((n, n)) + np.eye(n)
        big[-1] = big[0]
        with pytest.raises(ms.LinAlgError):
            ms.solve(ms.array(big.tolist(), dtype=ms.float64), b)

    def test_solve_large_2d_rhs(self):
        """n≥128 + 多 rhs：getrs 列主序拷贝路径大矩阵回归。"""
        n = 128
        a = ms.add(ms.full([n, n], 1.0, dtype=ms.float64), ms.eye(n, dtype=ms.float64))
        b = ms.ones([n, 4], dtype=ms.float64)
        x = ms.solve(a, b)
        exp = 1.0 / (n + 1.0)
        assert x.shape == (n, 4)
        assert np.allclose(x.tolist(), np.full((n, 4), exp), atol=1e-9)
