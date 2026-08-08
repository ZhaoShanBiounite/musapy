"""v0.3 Phase 5 (P5.7): ms.fft 命名空间验收（ADR-003 003-D5/D7）。

覆盖：
  - fft/ifft 数值对照 NumPy（complex 输入；real 输入内部扩 complex）
  - 圆整性 ifft(fft(x)) ≈ x
  - rfft 形状 (..., N//2+1) + 数值对照
  - norm='backward'/'ortho'/'forward' 三值
  - n 截断/补零
  - 2D 输入（axis=-1 逐行）
  - GPU-only：CPU 设备抛 DeviceError
  - 错误路径：非法 norm、axis!=−1、rfft complex 输入拒绝

mock 模式（MUSAPY_MOCK_MUSA=1）下 mufft stub 用 naive DFT 数值仿真，
本文件在无 GPU CI 同样可跑（与 test_random.py 同模式）。
"""

import numpy as np
import pytest

import musapy as ms

# GPU 探测（mock 模式下 Device("musa:0") 亦有效，见 test_random.py L12 注释）
try:
    _dev = ms.Device("musa:0")
    _gpu_available = True
except Exception:
    _gpu_available = False

musa_required = pytest.mark.skipif(not _gpu_available, reason="MUSA device not available")


def _as_np(a):
    return np.array(a.tolist(), dtype=complex)


@musa_required
class TestFftGpu:
    """fft/ifft/rfft 数值对照 NumPy。"""

    @pytest.fixture(autouse=True)
    def _gpu_default(self):
        ms.set_default_device("musa:0")
        yield
        ms.set_default_device("cpu")

    # ── fft / ifft ──

    def test_fft_c128(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], dtype='c128')
        y = ms.fft.fft(x)
        assert y.dtype == 'c128'
        assert np.allclose(_as_np(y), np.fft.fft([1, 2, 3, 4]), atol=1e-10)

    def test_fft_c64(self):
        x = ms.array([1.0 + 1j, 2.0 - 1j, 3.0, 4.0], dtype='c64')
        y = ms.fft.fft(x)
        assert y.dtype == 'c64'
        assert np.allclose(_as_np(y), np.fft.fft(np.array(x.tolist(), dtype=complex)), atol=1e-5)

    def test_fft_real_f64(self):
        y = ms.fft.fft(ms.array([1.0, 2.0, 3.0, 4.0], dtype='f64'))
        assert y.dtype == 'c128'
        assert np.allclose(_as_np(y), np.fft.fft([1, 2, 3, 4]), atol=1e-10)

    def test_fft_real_f32(self):
        y = ms.fft.fft(ms.array([1.0, 2.0, 3.0, 4.0], dtype='f32'))
        assert y.dtype == 'c64'
        assert np.allclose(
            _as_np(y), np.fft.fft(np.array([1, 2, 3, 4], dtype=np.float32)), atol=1e-5
        )

    def test_ifft_roundtrip(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], dtype='c128')
        assert np.allclose(_as_np(ms.fft.ifft(ms.fft.fft(x))), [1, 2, 3, 4], atol=1e-10)

    def test_fft_2d_rows(self):
        m = ms.array([[1.0 + 0j, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='c128')
        y = ms.fft.fft(m)
        assert y.shape == (2, 3)
        assert np.allclose(
            _as_np(y), np.fft.fft(np.array([[1.0, 2, 3], [4, 5, 6]]), axis=-1), atol=1e-10
        )

    def test_norm_values(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], dtype='c128')
        for norm in ("backward", "ortho", "forward"):
            y = ms.fft.fft(x, norm=norm)
            assert np.allclose(
                _as_np(y), np.fft.fft([1, 2, 3, 4], norm=norm), atol=1e-10
            ), norm
            b = ms.fft.ifft(x, norm=norm)
            assert np.allclose(
                _as_np(b), np.fft.ifft([1, 2, 3, 4], norm=norm), atol=1e-10
            ), norm

    def test_n_pad_and_truncate(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], dtype='c128')
        # 补零
        y = ms.fft.fft(x, n=8)
        assert y.shape == (8,)
        assert np.allclose(_as_np(y), np.fft.fft([1, 2, 3, 4], n=8), atol=1e-10)
        # 截断
        y = ms.fft.fft(x, n=3)
        assert y.shape == (3,)
        assert np.allclose(_as_np(y), np.fft.fft([1, 2, 3, 4], n=3), atol=1e-10)

    # ── rfft ──

    def test_rfft_shape_and_values(self):
        y = ms.fft.rfft(ms.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='f64'))
        assert y.shape == (4,), "N//2+1 = 4"
        assert y.dtype == 'c128'
        assert np.allclose(_as_np(y), np.fft.rfft([1.0, 2, 3, 4, 5, 6]), atol=1e-10)

    def test_rfft_f32(self):
        y = ms.fft.rfft(ms.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='f32'))
        assert y.dtype == 'c64'
        assert np.allclose(
            _as_np(y), np.fft.rfft(np.array([1.0, 2, 3, 4, 5, 6], dtype=np.float32)), atol=1e-5
        )

    def test_rfft_2d(self):
        m = ms.array([[1.0, 2, 3, 4], [5, 6, 7, 8]], dtype='f64')
        y = ms.fft.rfft(m, norm="ortho")
        assert y.shape == (2, 3)
        assert np.allclose(
            _as_np(y), np.fft.rfft(np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]), axis=-1, norm="ortho"), atol=1e-10
        )

    def test_rfft_n_pad(self):
        y = ms.fft.rfft(ms.array([1.0, 2, 3, 4], dtype='f64'), n=8)
        assert y.shape == (5,), "8//2+1 = 5"
        assert np.allclose(_as_np(y), np.fft.rfft([1.0, 2, 3, 4], n=8), atol=1e-10)

    # ── 错误路径 ──

    def test_invalid_norm_rejected(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], dtype='c128')
        with pytest.raises(ms.ShapeError):
            ms.fft.fft(x, norm="bogus")

    def test_axis_not_last_rejected(self):
        m = ms.array([[1.0, 2.0], [3.0, 4.0]], dtype='c128')
        with pytest.raises(ms.ShapeError):
            ms.fft.fft(m, axis=0)

    def test_rfft_complex_rejected(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], dtype='c128')
        with pytest.raises(ms.DtypeError):
            ms.fft.rfft(x)

    def test_scalar_input_rejected(self):
        s = ms.array(1.0 + 0j)
        with pytest.raises(ms.ShapeError):
            ms.fft.fft(s)

    # ── out= ──

    def test_out(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], dtype='c128')
        out = ms.array([0.0 + 0j] * 4, dtype='c128')
        y = ms.fft.fft(x, out=out)
        assert np.allclose(_as_np(y), np.fft.fft([1, 2, 3, 4]), atol=1e-10)
        # out 被就地写入
        assert np.allclose(_as_np(out), _as_np(y), atol=0)


class TestFftCpuRejected:
    """v0.3 GPU-only（003-D4）：CPU 设备输入必须拒绝（DeviceError）。"""

    def test_fft_cpu_rejected(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], dtype='c128', device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.fft.fft(x)

    def test_rfft_cpu_rejected(self):
        x = ms.array([1.0, 2.0, 3.0, 4.0], device="cpu")
        with pytest.raises(ms.DeviceError):
            ms.fft.rfft(x)
