"""ms.array() 验收测试 — 基本创建、属性、repr、dtype 支持。

对应 ADR：L0-6（device 解析）、L0-7（dtype 解析）、L0-8（反馈原则）、
L1-11（0-dim Array）、L3-27（Array naming）。
"""

import math

import pytest

import musapy as ms
from musapy import Device, Dtype


# ============================================================
# 基本创建 + repr
# ============================================================


class TestArrayCreate:
    """ms.array() 基本创建。"""

    def test_create_float32_on_cpu(self):
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="cpu")
        assert a.shape == (3,)
        assert a.ndim == 1
        assert a.size == 3

    def test_create_float64(self):
        a = ms.array([1.0, 2.0], dtype=ms.float64, device="cpu")
        assert a.dtype == ms.float64
        assert a.nbytes == 16  # 2 elements * 8 bytes

    def test_create_int32(self):
        a = ms.array([10, 20, 30], dtype=ms.int32, device="cpu")
        assert a.dtype == ms.int32
        assert a.size == 3
        assert a.nbytes == 12  # 3 * 4

    def test_create_int64(self):
        a = ms.array([1, 2, 3, 4], dtype=ms.int64, device="cpu")
        assert a.dtype == ms.int64
        assert a.nbytes == 32  # 4 * 8

    def test_create_bool(self):
        a = ms.array([True, False, True], dtype=ms.bool_, device="cpu")
        assert a.dtype == ms.bool_
        assert a.nbytes == 3  # 3 * 1

    def test_create_uint8(self):
        a = ms.array([0, 128, 255], dtype=ms.uint8, device="cpu")
        assert a.dtype == ms.uint8
        assert a.nbytes == 3

    def test_create_with_default_device(self):
        """使用 conftest 设置的默认 device='cpu' 创建数组。"""
        a = ms.array([1.0, 2.0], dtype=ms.float32)
        assert a.shape == (2,)
        # 默认 device 来源应该是 global_default
        assert "global_default" in repr(a.device)

    def test_create_with_device_object(self):
        """传入 Device 对象而非字符串。"""
        a = ms.array([1.0], dtype=ms.float32, device=Device("cpu"))
        assert a.shape == (1,)


# ============================================================
# repr 格式（L0-8 反馈原则）
# ============================================================


class TestArrayRepr:
    """Array 和 Device 的 __repr__ 格式验证。"""

    def test_array_repr_format(self):
        a = ms.array([1.0, 2.0, 3.0], dtype=ms.float32, device="cpu")
        r = repr(a)
        assert "Array(" in r
        assert "shape=(3,)" in r
        assert "dtype=float32" in r
        assert "device=cpu" in r

    def test_array_str_same_as_repr(self):
        a = ms.array([1.0, 2.0], dtype=ms.float32, device="cpu")
        assert str(a) == repr(a)

    def test_device_repr_with_resolution(self):
        """从 array 解析出的 device 应显示 resolution source。"""
        a = ms.array([1.0], dtype=ms.float32, device="cpu")
        r = repr(a.device)
        assert "Device(cpu)" in r
        assert "resolved from:" in r

    def test_device_repr_without_resolution(self):
        """直接构造的 Device 不显示 resolution。"""
        d = Device("cpu")
        assert repr(d) == "Device(cpu)"

    def test_device_str(self):
        assert str(Device("cpu")) == "cpu"
        assert str(Device("musa:0")) == "musa:0"


# ============================================================
# 属性访问
# ============================================================


class TestArrayAttributes:
    """Array 属性：shape, ndim, size, dtype, nbytes, is_contiguous, is_0d。"""

    def test_1d_attributes(self):
        a = ms.array([1.0, 2.0, 3.0, 4.0], dtype=ms.float32, device="cpu")
        assert a.shape == (4,)
        assert a.ndim == 1
        assert a.size == 4
        assert a.nbytes == 16
        assert a.is_contiguous is True
        assert a.is_0d is False

    def test_single_element(self):
        a = ms.array([42.0], dtype=ms.float32, device="cpu")
        assert a.shape == (1,)
        assert a.size == 1
        assert a.nbytes == 4

    def test_dtype_property(self):
        a = ms.array([1.0], dtype=ms.float32, device="cpu")
        assert isinstance(a.dtype, Dtype)
        assert a.dtype == ms.float32
        assert a.dtype.name == "float32"
        assert a.dtype.element_size == 4
        assert a.dtype.is_floating is True

    def test_stream_property(self):
        a = ms.array([1.0], dtype=ms.float32, device="cpu")
        s = a.stream
        assert s.priority == 0
        assert str(s.device) == "cpu"


# ============================================================
# Array naming（L3-27）
# ============================================================


class TestArrayNaming:
    """Array name 管理。"""

    def test_default_name_none(self):
        a = ms.array([1.0], dtype=ms.float32, device="cpu")
        assert a.name is None

    def test_set_name(self):
        a = ms.array([1.0], dtype=ms.float32, device="cpu")
        a.name = "my_array"
        assert a.name == "my_array"

    def test_clear_name(self):
        a = ms.array([1.0], dtype=ms.float32, device="cpu")
        a.name = "temp"
        a.clear_name()
        assert a.name is None


# ============================================================
# Dtype 常量
# ============================================================


class TestDtypeConstants:
    """15 个 dtype 常量注册正确。"""

    @pytest.mark.parametrize(
        "name,constant",
        [
            ("bool", ms.bool_),
            ("int8", ms.int8),
            ("int16", ms.int16),
            ("int32", ms.int32),
            ("int64", ms.int64),
            ("uint8", ms.uint8),
            ("uint16", ms.uint16),
            ("uint32", ms.uint32),
            ("uint64", ms.uint64),
            ("float16", ms.float16),
            ("float32", ms.float32),
            ("float64", ms.float64),
            ("bfloat16", ms.bfloat16),
            ("complex64", ms.complex64),
            ("complex128", ms.complex128),
        ],
    )
    def test_dtype_name(self, name, constant):
        assert constant.name == name

    def test_dtype_construct_from_string(self):
        d = Dtype("float32")
        assert d == ms.float32

    def test_dtype_equality(self):
        assert ms.float32 == ms.float32
        assert ms.float32 != ms.float64

    def test_dtype_repr(self):
        assert repr(ms.float32) == "float32"
        assert str(ms.float32) == "float32"

    def test_dtype_element_sizes(self):
        assert ms.bool_.element_size == 1
        assert ms.int32.element_size == 4
        assert ms.int64.element_size == 8
        assert ms.float32.element_size == 4
        assert ms.float64.element_size == 8
        assert ms.complex64.element_size == 8
        assert ms.complex128.element_size == 16

    def test_dtype_categories(self):
        assert ms.bool_.is_integer and not ms.bool_.is_floating
        assert ms.int32.is_integer and not ms.int32.is_floating
        assert ms.float32.is_floating and not ms.float32.is_integer
        assert ms.complex64.is_complex
