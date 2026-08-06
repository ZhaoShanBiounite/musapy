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
