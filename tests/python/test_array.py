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
        a = ms.array([1.0, 2.0, 3.0], dtype='f32', device="cpu")
        assert a.shape == (3,)
        assert a.ndim == 1
        assert a.size == 3

    def test_create_float64(self):
        a = ms.array([1.0, 2.0], dtype='f64', device="cpu")
        assert a.dtype == 'f64'
        assert a.nbytes == 16  # 2 elements * 8 bytes

    def test_create_int32(self):
        a = ms.array([10, 20, 30], dtype='i32', device="cpu")
        assert a.dtype == 'i32'
        assert a.size == 3
        assert a.nbytes == 12  # 3 * 4

    def test_create_int64(self):
        a = ms.array([1, 2, 3, 4], dtype='i64', device="cpu")
        assert a.dtype == 'i64'
        assert a.nbytes == 32  # 4 * 8

    def test_create_bool(self):
        a = ms.array([True, False, True], dtype='b1', device="cpu")
        assert a.dtype == 'b1'
        assert a.nbytes == 3  # 3 * 1

    def test_create_uint8(self):
        a = ms.array([0, 128, 255], dtype='u8', device="cpu")
        assert a.dtype == 'u8'
        assert a.nbytes == 3

    def test_create_with_default_device(self):
        """使用 conftest 设置的默认 device='cpu' 创建数组。"""
        a = ms.array([1.0, 2.0], dtype='f32')
        assert a.shape == (2,)
        # 默认 device 来源应该是 global_default
        assert "global_default" in repr(a.device)

    def test_create_with_device_object(self):
        """传入 Device 对象而非字符串。"""
        a = ms.array([1.0], dtype='f32', device=Device("cpu"))
        assert a.shape == (1,)


# ============================================================
# repr 格式（L0-8 反馈原则）
# ============================================================


class TestArrayRepr:
    """Array 和 Device 的 __repr__ 格式验证。"""

    def test_array_repr_format(self):
        a = ms.array([1.0, 2.0, 3.0], dtype='f32', device="cpu")
        r = repr(a)
        assert "Array(" in r
        assert "shape=(3,)" in r
        assert "dtype=float32" in r
        assert "device=cpu" in r

    def test_array_str_same_as_repr(self):
        a = ms.array([1.0, 2.0], dtype='f32', device="cpu")
        assert str(a) == repr(a)

    def test_device_repr_with_resolution(self):
        """从 array 解析出的 device 应显示 resolution source。"""
        a = ms.array([1.0], dtype='f32', device="cpu")
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
        a = ms.array([1.0, 2.0, 3.0, 4.0], dtype='f32', device="cpu")
        assert a.shape == (4,)
        assert a.ndim == 1
        assert a.size == 4
        assert a.nbytes == 16
        assert a.is_contiguous is True
        assert a.is_0d is False

    def test_single_element(self):
        a = ms.array([42.0], dtype='f32', device="cpu")
        assert a.shape == (1,)
        assert a.size == 1
        assert a.nbytes == 4

    def test_dtype_property(self):
        a = ms.array([1.0], dtype='f32', device="cpu")
        assert isinstance(a.dtype, Dtype)
        assert a.dtype == 'f32'
        assert a.dtype.name == "float32"
        assert a.dtype.element_size == 4
        assert a.dtype.is_floating is True

    def test_stream_property(self):
        a = ms.array([1.0], dtype='f32', device="cpu")
        s = a.stream
        assert s.priority == 0
        assert str(s.device) == "cpu"


# ============================================================
# Array naming（L3-27）
# ============================================================


class TestArrayNaming:
    """Array name 管理。"""

    def test_default_name_none(self):
        a = ms.array([1.0], dtype='f32', device="cpu")
        assert a.name is None

    def test_set_name(self):
        a = ms.array([1.0], dtype='f32', device="cpu")
        a.name = "my_array"
        assert a.name == "my_array"

    def test_clear_name(self):
        a = ms.array([1.0], dtype='f32', device="cpu")
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
        assert d == 'f32'

    def test_dtype_equality(self):
        # Dtype 与字符串别名/全名可互比（v0.3）
        assert ms.float32 == 'f32'
        assert ms.float32 == 'float32'
        assert ms.float32 != 'f64'
        assert Dtype("f64") == 'f64'

    @pytest.mark.parametrize(
        "alias,full",
        [
            ("b1", "bool"),
            ("i8", "int8"),
            ("i16", "int16"),
            ("i32", "int32"),
            ("i64", "int64"),
            ("u8", "uint8"),
            ("u16", "uint16"),
            ("u32", "uint32"),
            ("u64", "uint64"),
            ("f16", "float16"),
            ("f32", "float32"),
            ("f64", "float64"),
            ("bf16", "bfloat16"),
            ("c64", "complex64"),
            ("c128", "complex128"),
            ("half", "float16"),
            ("single", "float32"),
            ("double", "float64"),
        ],
    )
    def test_dtype_alias_parse(self, alias, full):
        """字符串别名（含全名/兼容名）解析到正确 dtype。"""
        assert Dtype(alias).name == full

    def test_dtype_string_arg_short_aliases(self):
        """dtype='f32' 字符串参数语法（v0.3 主推形式）。"""
        assert ms.array([1.0], dtype='f32').dtype == 'f32'
        assert ms.array([1], dtype='i64').dtype == 'i64'
        assert ms.array([True], dtype='b1').dtype == 'b1'
        assert ms.array([1 + 2j], dtype='c64').dtype == 'c64'
        assert ms.zeros(2, dtype='f64').dtype == 'f64'

    def test_dtype_string_arg_full_names(self):
        """dtype='float32' 全名（向后兼容）。"""
        assert ms.array([1.0], dtype='float32').dtype == 'f32'
        assert ms.array([1.0], dtype='float64').dtype == 'f64'
        assert ms.array([1], dtype='int64').dtype == 'i64'
        assert ms.array([1 + 2j], dtype='complex128').dtype == 'c128'

    def test_dtype_string_case_insensitive(self):
        assert ms.array([1.0], dtype='F32').dtype == 'f32'
        assert ms.array([1.0], dtype='Float64').dtype == 'f64'

    def test_dtype_string_error(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            ms.array([1.0], dtype='fancy64')
        with pytest.raises(TypeError):
            ms.array([1.0], dtype=123)

    def test_astype_string(self):
        """astype('f64') 字符串参数。"""
        a = ms.array([1.0, 2.0], dtype='f32')
        b = a.astype('f64')
        assert b.dtype == 'f64'
        assert b.dtype == 'double'
        c = ms.array([1.0, 2.0], dtype='f64').astype('f32')
        assert c.dtype == 'f32'

    def test_dtype_context_string(self):
        """with ms.dtype('f64'): 字符串参数。"""
        ms.set_default_device("cpu")
        with ms.dtype('f64'):
            a = ms.zeros(2)
            assert a.dtype == 'f64'
        b = ms.zeros(2)
        assert b.dtype == 'f32'

    def test_set_default_dtype_string(self):
        """set_default_dtype('f64') 字符串参数。"""
        ms.set_default_dtype('f64')
        try:
            assert ms.zeros(2).dtype == 'f64'
        finally:
            ms.set_default_dtype(ms.float32)

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
