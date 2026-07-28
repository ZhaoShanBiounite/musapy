"""设备/dtype/stream 解析链验收测试。

对应 ADR：L0-6（5 级 device 解析）、L0-7（dtype 解析）、L0-8（反馈原则）、
L0-9（DeviceNotConfigured）、L0-11（线程继承）、L2-7（context 对称）。
"""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import musapy as ms
from musapy import Device, Dtype, Stream


# ============================================================
# 5 级解析优先级
# ============================================================


class TestResolutionPriority:
    """resolve_device/resolve_dtype 的 5 级优先级。"""

    def test_level1_arg_wins_over_default(self):
        """显式 device 参数优先于全局默认。"""
        ms.set_default_device("cpu")
        a = ms.array([1.0, 2.0], dtype=ms.float32, device="cpu")
        r = repr(a.device)
        # 显式参数 → source = arg
        assert "resolved from: arg" in r

    def test_level4_global_default(self):
        """无显式参数时用全局默认。"""
        ms.set_default_device("cpu")
        a = ms.array([1.0, 2.0], dtype=ms.float32)
        r = repr(a.device)
        assert "resolved from: global_default" in r

    def test_level2_context_overrides_default(self):
        """context manager 优先于全局默认。"""
        ms.set_default_device("cpu")
        with ms.device("cpu"):
            a = ms.array([1.0], dtype=ms.float32)
            r = repr(a.device)
            # context 优先级高于 global_default
            assert "resolved from: context" in r
        # context 退出后恢复 global_default
        b = ms.array([1.0], dtype=ms.float32)
        assert "resolved from: global_default" in repr(b.device)

    def test_level1_arg_wins_over_context(self):
        """显式参数优先于 context。"""
        ms.set_default_device("cpu")
        with ms.device("cpu"):
            a = ms.array([1.0], dtype=ms.float32, device="cpu")
            assert "resolved from: arg" in repr(a.device)

    def test_dtype_float32_fallback(self):
        """dtype 未指定时兜底为 float32（L0-7）。"""
        ms.set_default_device("cpu")
        a = ms.array([1.0, 2.0])
        assert a.dtype == ms.float32


# ============================================================
# DeviceNotConfigured（L0-9）
# ============================================================


class TestDeviceNotConfigured:
    """未设默认 device 时 ms.array() 抛 DeviceNotConfiguredError。

    由于 Rust 侧全局 SEED 是进程级共享的，其他测试 set_default_device 后
    SEED 不为空。所以用 subprocess 在全新进程中测试。
    """

    def test_array_without_default_raises(self):
        """全新进程中未 set_default_device → array() 抛 DeviceNotConfiguredError。"""
        code = (
            "from musapy import array, float32\n"
            "try:\n"
            "    a = array([1.0, 2.0], dtype=float32)\n"
            "    print('NO_ERROR')\n"
            "except Exception as e:\n"
            "    print(type(e).__name__)\n"
            "    print(str(e))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        stdout = result.stdout.strip()
        assert "NO_ERROR" not in stdout
        assert "DeviceNotConfiguredError" in stdout

    def test_exception_inheritance(self):
        """DeviceNotConfiguredError 是 DeviceError 的子类。"""
        assert issubclass(ms.DeviceNotConfiguredError, ms.DeviceError)
        assert issubclass(ms.DeviceError, ms.MusapyError)


# ============================================================
# Context managers（L2-7 对称可组合）
# ============================================================


class TestContextManagers:
    """device/dtype/stream context manager 对称性。"""

    def test_device_context_enters_and_exits(self):
        ms.set_default_device("cpu")
        before = ms.array([1.0], dtype=ms.float32)
        assert "global_default" in repr(before.device)

        with ms.device("cpu"):
            inside = ms.array([1.0], dtype=ms.float32)
            assert "context" in repr(inside.device)

        after = ms.array([1.0], dtype=ms.float32)
        assert "global_default" in repr(after.device)

    def test_dtype_context(self):
        ms.set_default_device("cpu")
        with ms.dtype(ms.float64):
            a = ms.array([1.0, 2.0])
            assert a.dtype == ms.float64
            assert a.dtype_resolution_source == "context"

        # 退出后 dtype 兜底回 float32
        b = ms.array([1.0, 2.0])
        assert b.dtype == ms.float32

    def test_stream_context(self):
        ms.set_default_device("cpu")
        s = Stream("cpu", priority=0)
        with ms.stream(s):
            a = ms.array([1.0], dtype=ms.float32)
            # stream context 不改变 device 解析，但绑定了 stream
            assert a.stream.priority == 0

    def test_nested_device_contexts(self):
        ms.set_default_device("cpu")
        with ms.device("cpu"):
            a = ms.array([1.0], dtype=ms.float32)
            assert "context" in repr(a.device)
            with ms.device("cpu"):
                b = ms.array([1.0], dtype=ms.float32)
                assert "context" in repr(b.device)
            c = ms.array([1.0], dtype=ms.float32)
            assert "context" in repr(c.device)
        d = ms.array([1.0], dtype=ms.float32)
        assert "global_default" in repr(d.device)


# ============================================================
# Device 类
# ============================================================


class TestDeviceClass:
    """Device 类的基本行为。"""

    def test_device_construct_from_string(self):
        d = Device("cpu")
        assert d.is_musa is False
        assert d.musa_id is None

    def test_device_musa(self):
        d = Device("musa:0")
        assert d.is_musa is True
        assert d.musa_id == 0

    def test_device_equality(self):
        assert Device("cpu") == Device("cpu")
        assert Device("musa:0") == Device("musa:0")
        assert Device("cpu") != Device("musa:0")

    def test_device_hash(self):
        """Device 可用作 dict key。"""
        d = {Device("cpu"): 1, Device("musa:0"): 2}
        assert d[Device("cpu")] == 1
        assert d[Device("musa:0")] == 2


# ============================================================
# 异常层次
# ============================================================


class TestExceptionHierarchy:
    """ADR L3-5/L3-7 异常层次结构。"""

    def test_all_exceptions_are_musapy_errors(self):
        for exc in [
            ms.DeviceError,
            ms.DtypeError,
            ms.ShapeError,
            ms.MemoryError,
            ms.StreamError,
            ms.KernelError,
            ms.InteropError,
        ]:
            assert issubclass(exc, ms.MusapyError)

    def test_device_not_configured_is_device_error(self):
        assert issubclass(ms.DeviceNotConfiguredError, ms.DeviceError)

    def test_out_of_memory_is_memory_error(self):
        assert issubclass(ms.OutOfMemoryError, ms.MemoryError)
        # ADR L3-7: 不继承 Python's MemoryError
        assert not issubclass(ms.OutOfMemoryError, MemoryError)


# ============================================================
# Stream 类
# ============================================================


class TestStreamClass:
    """Stream 类的基本行为。"""

    def test_stream_create_cpu(self):
        s = Stream("cpu")
        assert s.priority == 0
        assert s.is_poisoned is False
        assert str(s.device) == "cpu"

    def test_stream_synchronize(self):
        s = Stream("cpu")
        s.synchronize()  # 不应抛异常

    def test_stream_repr(self):
        s = Stream("cpu", priority=0)
        r = repr(s)
        assert "Stream(" in r
        assert "device=cpu" in r
        assert "priority=0" in r

    def test_stream_id_unique(self):
        s1 = Stream("cpu")
        s2 = Stream("cpu")
        assert s1.id != s2.id


# ============================================================
# Thread-local 默认隔离（ADR L0-11）
# ============================================================


class TestThreadIsolation:
    """thread-local 默认 device/dtype 隔离验证（P7.3）。

    ADR L0-11：
    - 新线程在 spawn 时继承父线程当前的默认（值快照，之后解耦）
    - 兄弟线程互不影响
    """

    def test_child_inherits_parent_default(self):
        """子线程继承父线程的 default device。"""
        ms.set_default_device("cpu")
        result = {}

        def worker():
            # 子线程应继承父线程的 cpu 默认
            a = ms.array([1.0], dtype=ms.float32)
            result["device"] = str(a.device)
            result["source"] = repr(a.device)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert "cpu" in result["device"]

    def test_child_change_does_not_affect_parent(self):
        """子线程修改 default 不影响主线程。"""
        ms.set_default_device("cpu")

        def worker():
            # 子线程重新设置（对主线程无影响）
            ms.set_default_device("cpu")
            a = ms.array([1.0], dtype=ms.float32)
            assert "cpu" in str(a.device)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # 主线程仍然是 cpu
        a = ms.array([2.0], dtype=ms.float32)
        assert "cpu" in str(a.device)
        assert "global_default" in repr(a.device)

    def test_sibling_threads_independent(self):
        """兄弟线程互不影响。"""
        ms.set_default_device("cpu")
        results = {}
        barrier = threading.Barrier(3)

        def worker(name):
            # 所有线程继承 cpu
            a = ms.array([1.0], dtype=ms.float32)
            results[f"{name}_before"] = str(a.device)
            # 同步点：确保所有线程都读了初始值
            barrier.wait()
            # 每个线程重新设置（不影响其他线程）
            ms.set_default_device("cpu")
            b = ms.array([2.0], dtype=ms.float32)
            results[f"{name}_after"] = str(b.device)

        threads = [
            threading.Thread(target=worker, args=(f"t{i}",))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有线程都成功创建了 array
        for i in range(3):
            assert "cpu" in results[f"t{i}_before"]
            assert "cpu" in results[f"t{i}_after"]

    def test_thread_pool_isolation(self):
        """ThreadPoolExecutor 中各任务线程隔离。"""
        ms.set_default_device("cpu")

        def task(idx):
            a = ms.array([float(idx)], dtype=ms.float32)
            return (idx, a.tolist()[0], str(a.device))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(task, i) for i in range(8)]
            results = [f.result() for f in futures]

        for idx, val, dev in results:
            assert val == float(idx)
            assert "cpu" in dev

    def test_dtype_thread_isolation(self):
        """dtype 默认也是 thread-local 隔离的。"""
        ms.set_default_device("cpu")
        results = {}

        def worker(name):
            # 继承全局 float32 兜底
            a = ms.array([1.0])
            results[f"{name}_dtype"] = a.dtype

        threads = [
            threading.Thread(target=worker, args=(f"t{i}",))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(3):
            assert results[f"t{i}_dtype"] == ms.float32
