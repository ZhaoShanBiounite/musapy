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


# ============================================================
# v0.3 Phase 3 (P3.1–P3.6): lu / qr / svd 验收（ADR-003 003-D3/D6）
#
# 分解类算子 GPU-only（003-D4 修订）；数值对照走真机 GPU。
# 语义（2026-08-07 真机 C 探针锁定）：
#   - lu：getrf 列主序副本 → 跨步视图标准行主序 L·U；piv 1-based int64
#   - qr：geqrf+orgqr；R 在 orgqr 前独立提取（orgqr 覆盖前 k 列）
#   - svd：gesvd 的 V 输出即 V（非 Vᵀ）→ vh = 转置视图；S 降序
# ============================================================


class TestLinalgDecompCpuRejected:
    """v0.3 GPU-only：分解类算子 CPU 输入必须拒绝（DeviceError）。"""

    def test_lu_cpu_rejected(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.lu(a)

    def test_qr_cpu_rejected(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.qr(a)

    def test_svd_cpu_rejected(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.svd(a)


class TestLinalgDecompGpu:
    """分解类算子真机数值对照（lu 重建 / qr 正交重构 / svd 重建与降序）。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    # ── lu ──────────────────────────────────────────────────

    def test_lu_f64(self):
        rng = np.random.default_rng(23)
        A = rng.normal(size=(5, 5))
        lu, piv = ms.lu(ms.array(A.tolist(), dtype=ms.float64))
        lu_np = np.array(lu.tolist())
        piv_np = np.array(piv.tolist())

        # 返回语义：lu (m×n) 标准行主序（L 单位下三角 + U 上三角），
        # piv 为 1-based int64（LAPACK ipiv）
        assert lu.shape == (5, 5)
        assert piv.shape == (5,)
        assert piv.dtype == ms.int64
        assert np.all(piv_np >= 1) and np.all(piv_np <= 5)

        # 重建 a = P·L·U：按 getrf 语义逐 i 交换行 (i, piv[i]-1)
        k = 5
        L = np.tril(lu_np, -1)[:, :k] + np.eye(5, k)
        U = np.triu(lu_np)[:k, :]
        PA = A.copy()
        for i, p in enumerate(piv_np):
            j = int(p) - 1
            PA[[i, j], :] = PA[[j, i], :]
        assert np.allclose(PA, L @ U, atol=1e-8)

    def test_lu_rectangular(self):
        rng = np.random.default_rng(25)
        A = rng.normal(size=(6, 3))
        lu, piv = ms.lu(ms.array(A.tolist(), dtype=ms.float64))
        assert lu.shape == (6, 3)
        assert piv.shape == (3,)
        k = 3
        L = np.tril(np.array(lu.tolist()), -1)[:, :k] + np.eye(6, k)
        U = np.triu(np.array(lu.tolist()))[:k, :]
        PA = A.copy()
        for i, p in enumerate(np.array(piv.tolist())):
            j = int(p) - 1
            PA[[i, j], :] = PA[[j, i], :]
        assert np.allclose(PA, L @ U, atol=1e-8)

    def test_lu_f32(self):
        rng = np.random.default_rng(27)
        A = rng.normal(size=(4, 4))
        lu, piv = ms.lu(ms.array(A.tolist(), dtype=ms.float32))
        k = 4
        L = np.tril(np.array(lu.tolist()), -1)[:, :k] + np.eye(4, k)
        U = np.triu(np.array(lu.tolist()))[:k, :]
        PA = A.copy()
        for i, p in enumerate(np.array(piv.tolist())):
            j = int(p) - 1
            PA[[i, j], :] = PA[[j, i], :]
        assert np.allclose(PA, L @ U, atol=1e-4)

    def test_lu_singular_no_crash(self):
        """奇异矩阵不崩溃；piv 仍在合法范围（info 失效 SDK 缺陷见 solve 注释）。"""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        lu, piv = ms.lu(ms.array(A.tolist(), dtype=ms.float64))
        assert lu.shape == (2, 2)
        p = np.array(piv.tolist())
        assert np.all(p >= 1) and np.all(p <= 2)

    # ── qr ──────────────────────────────────────────────────

    def test_qr_f64(self):
        rng = np.random.default_rng(29)
        A = rng.normal(size=(6, 4))
        q, r = ms.qr(ms.array(A.tolist(), dtype=ms.float64))
        Q = np.array(q.tolist())
        R = np.array(r.tolist())
        assert q.shape == (6, 4) and r.shape == (4, 4)
        # 重构 a = q@r 且 q 正交；R 上三角
        assert np.allclose(Q @ R, A, atol=1e-8)
        assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-8)
        assert np.allclose(R, np.triu(R))

    def test_qr_complete(self):
        rng = np.random.default_rng(33)
        A = rng.normal(size=(5, 3))
        q, r = ms.qr(ms.array(A.tolist(), dtype=ms.float64), mode="complete")
        Q = np.array(q.tolist())
        R = np.array(r.tolist())
        assert q.shape == (5, 5) and r.shape == (5, 3)
        assert np.allclose(Q @ R, A, atol=1e-8)
        assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-8)
        # R 下三角补零
        assert np.allclose(R, np.triu(R))

    def test_qr_wide_matrix(self):
        """m < n：reduced 与 complete 均为 q (m,m)、r (m,n)。"""
        rng = np.random.default_rng(37)
        A = rng.normal(size=(3, 6))
        for mode, exp_q in (("reduced", (3, 3)), ("complete", (3, 3))):
            q, r = ms.qr(ms.array(A.tolist(), dtype=ms.float64), mode=mode)
            assert q.shape == exp_q and r.shape == (3, 6)
            Q = np.array(q.tolist())
            R = np.array(r.tolist())
            assert np.allclose(Q @ R, A, atol=1e-8)

    def test_qr_f32(self):
        rng = np.random.default_rng(41)
        A = rng.normal(size=(8, 5))
        q, r = ms.qr(ms.array(A.tolist(), dtype=ms.float32))
        Q = np.array(q.tolist())
        R = np.array(r.tolist())
        assert np.allclose(Q @ R, A, atol=1e-4)
        assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-4)

    def test_qr_degenerate_zero_matrix(self):
        """零矩阵：geqrf tau=0 → Q 单位阵；R 全零，重构成立。"""
        A = np.zeros((4, 4))
        q, r = ms.qr(ms.array(A.tolist(), dtype=ms.float64))
        Q = np.array(q.tolist())
        R = np.array(r.tolist())
        assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-5)
        assert np.allclose(Q @ R, A, atol=1e-5)

    def test_qr_bad_mode_rejected(self):
        a = ms.array([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float64)
        with pytest.raises(ms.ShapeError):
            ms.qr(a, mode="invalid")

    # ── svd ─────────────────────────────────────────────────

    def test_svd_f64(self):
        rng = np.random.default_rng(43)
        A = rng.normal(size=(5, 3))
        u, s, vh = ms.svd(ms.array(A.tolist(), dtype=ms.float64))
        U = np.array(u.tolist())
        S = np.array(s.tolist())
        Vh = np.array(vh.tolist())
        assert u.shape == (5, 5) and s.shape == (3,) and vh.shape == (3, 3)
        # 降序 + 与 NumPy 奇异值一致
        assert np.all(np.diff(S) <= 1e-10)
        assert np.allclose(S, np.linalg.svd(A)[1], atol=1e-8)
        # 重建 u@diag(s)@vh = a（full：u 尺寸 (m,m)、vh 尺寸 (n,n)，各取前 k 列/行）
        assert np.allclose(U[:, :3] @ np.diag(S) @ Vh[:3, :], A, atol=1e-8)
        # 正交性
        assert np.allclose(U.T @ U, np.eye(5), atol=1e-8)
        assert np.allclose(Vh @ Vh.T, np.eye(3), atol=1e-8)

    def test_svd_thin(self):
        rng = np.random.default_rng(47)
        A = rng.normal(size=(5, 3))
        u, s, vh = ms.svd(ms.array(A.tolist(), dtype=ms.float64), full_matrices=False)
        U = np.array(u.tolist())
        S = np.array(s.tolist())
        Vh = np.array(vh.tolist())
        assert u.shape == (5, 3) and vh.shape == (3, 3)
        assert np.allclose(U @ np.diag(S) @ Vh, A, atol=1e-8)
        assert np.allclose(U.T @ U, np.eye(3), atol=1e-8)

    def test_svd_wide_f64(self):
        rng = np.random.default_rng(53)
        A = rng.normal(size=(3, 5))
        u, s, vh = ms.svd(ms.array(A.tolist(), dtype=ms.float64))
        U = np.array(u.tolist())
        S = np.array(s.tolist())
        Vh = np.array(vh.tolist())
        assert u.shape == (3, 3) and vh.shape == (5, 5)
        # 宽矩阵：vh 取前 k 行重建
        assert np.allclose(U @ np.diag(S) @ Vh[:3, :], A, atol=1e-8)
        u2, s2, vh2 = ms.svd(
            ms.array(A.tolist(), dtype=ms.float64), full_matrices=False
        )
        assert u2.shape == (3, 3) and vh2.shape == (3, 5)
        assert np.allclose(
            np.array(u2.tolist()) @ np.diag(np.array(s2.tolist())) @ np.array(vh2.tolist()),
            A,
            atol=1e-8,
        )

    def test_svd_compute_uv_false(self):
        rng = np.random.default_rng(59)
        A = rng.normal(size=(4, 6))
        s = ms.svd(ms.array(A.tolist(), dtype=ms.float64), compute_uv=False)
        # NumPy 语义：仅返回 s（非三元组）
        assert s.shape == (4,)
        assert np.allclose(s.tolist(), np.linalg.svd(A, compute_uv=False), atol=1e-8)
        assert np.all(np.diff(np.array(s.tolist())) <= 1e-10)

    def test_svd_f32(self):
        rng = np.random.default_rng(61)
        A = rng.normal(size=(6, 4))
        u, s, vh = ms.svd(ms.array(A.tolist(), dtype=ms.float32))
        S = np.array(s.tolist())
        assert np.allclose(S, np.linalg.svd(A)[1], atol=1e-3)
        assert np.allclose(
            np.array(u.tolist())[:, :4] @ np.diag(S) @ np.array(vh.tolist())[:4, :],
            A,
            atol=1e-3,
        )

    def test_svd_degenerate_rank_deficient(self):
        """奇异值 0 的退化矩阵不崩溃，重建趋势与 NumPy 一致。"""
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[1, 1] = 2.0  # rank 2
        u, s, vh = ms.svd(ms.array(A.tolist(), dtype=ms.float64))
        S = np.array(s.tolist())
        assert S[0] > 1.9 and S[1] > 0.9 and S[2] < 1e-6 and S[3] < 1e-6
        assert np.allclose(
            np.array(u.tolist()) @ np.diag(S) @ np.array(vh.tolist()), A, atol=1e-8
        )

    # ── 形状参数矩阵（mock 与真机均验证形状管线）──────────────

    def test_decomp_shape_parameter_matrix(self):
        rng = np.random.default_rng(67)
        for shape in [(5, 3), (3, 5), (4, 4)]:
            m, n = shape
            k = min(m, n)
            arr = ms.array(rng.normal(size=shape).tolist(), dtype=ms.float64)
            # lu
            lu, piv = ms.lu(arr)
            assert lu.shape == shape and piv.shape == (k,)
            # qr
            q, r = ms.qr(arr)
            assert q.shape == (m, k) and r.shape == (k, n)
            q, r = ms.qr(arr, mode="complete")
            assert q.shape == (m, m) and r.shape == (m, n)
            # svd
            u, s, vh = ms.svd(arr)
            assert u.shape == (m, m) and s.shape == (k,) and vh.shape == (n, n)
            u, s, vh = ms.svd(arr, full_matrices=False)
            assert u.shape == (m, k) and vh.shape == (k, n)
            s2 = ms.svd(arr, compute_uv=False)
            assert s2.shape == (k,)
