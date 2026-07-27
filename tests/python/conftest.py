"""pytest 配置：确保 musapy 可导入，提供全局默认重置。"""

import pytest


@pytest.fixture(autouse=True)
def reset_defaults():
    """每个测试前设置一个已知默认 device，避免跨测试状态污染。

    Rust 侧的 thread-local 栈 + SEED 是进程级共享的，
    用 set_default_device("cpu") 覆盖为已知状态。
    需要测试 DeviceNotConfigured 的用例用 subprocess 隔离。
    """
    import musapy as ms

    ms.set_default_device("cpu")
    yield
    ms.set_default_device("cpu")
