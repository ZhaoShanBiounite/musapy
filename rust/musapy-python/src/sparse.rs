//! ms.sparse 命名空间的 _core 层（ADR-003 003-D7，v0.3 Phase 6）
//!
//! PyCsrMatrix 类（`#[pyclass]`，仿 PyArray 约定）+ `csr_matrix` 工厂。
//! Python 层 python/musapy/sparse.py 包装为 `ms.sparse.*` 命名空间。
//! GPU-only（003-D4）：CPU 设备上调用抛 DeviceError。

use crate::array::PyArray;
use crate::device::PyDevice;
use crate::dtype::PyDtype;
use crate::error;
use pyo3::prelude::*;

/// Python CsrMatrix 类。
#[pyclass(name = "CsrMatrix", module = "musapy")]
pub struct PyCsrMatrix {
    pub(crate) inner: musapy_ops::CsrMatrix,
}

impl PyCsrMatrix {
    pub fn from_csr(csr: musapy_ops::CsrMatrix) -> Self {
        Self { inner: csr }
    }
}

/// 从 3 个 Python Array 构造 CSR。
///
/// 要求 data 已为 Array（f32/f64），indices/indptr 为 Array（int32）。
/// 非 Array 输入（list/tuple）由 Python 层 sparse.py 先 ms.array() 转换。
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape=None, dtype=None))]
pub fn csr_matrix(
    py: Python<'_>,
    data: PyRef<'_, PyArray>,
    indices: PyRef<'_, PyArray>,
    indptr: PyRef<'_, PyArray>,
    shape: Option<(usize, usize)>,
    dtype: Option<PyDtype>,
) -> PyResult<PyCsrMatrix> {
    // dtype 参数（若给，须与 data.dtype 一致）
    if let Some(d) = dtype {
        if d.0 != data.inner.dtype() {
            return Err(error::to_pyerr(musapy_core::MusapyError::Dtype(
                musapy_core::error::DtypeError::Unsupported(format!(
                    "csr_matrix: dtype {} != data dtype {}",
                    d.0,
                    data.inner.dtype()
                )),
            )));
        }
    }
    let _ = py;

    // 解析 shape（默认 rows=len(indptr)-1, cols = max(indices)+1）
    let nnz = data.inner.shape()[0];
    let rows;
    let cols;
    match shape {
        Some((r, c)) => {
            rows = r;
            cols = c;
        }
        None => {
            // 从 indptr/indices 推断
            let n = indptr.inner.shape()[0];
            if n == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "csr_matrix: indptr must be non-empty",
                ));
            }
            rows = n - 1;
            // cols 需 D2H 读 indices 最大值；简单起见要求显式 shape（或 0 元素时 =0）
            if nnz == 0 {
                cols = 0;
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "csr_matrix: shape must be provided when nnz > 0 (cols not inferable from device data)",
                ));
            }
        }
    }

    let csr = musapy_ops::csr_from_arrays(&data.inner, &indices.inner, &indptr.inner, rows, cols)
        .map_err(error::to_pyerr)?;
    Ok(PyCsrMatrix::from_csr(csr))
}

#[pymethods]
impl PyCsrMatrix {
    /// `csr @ vec / csr @ dense` — spmv（1D）/ spmm（2D）。
    ///
    /// 右侧可为 ms.Array（device 直连）或 numpy ndarray/list
    /// （经 tolist → ms.array 构造临时 device Array）。
    fn __matmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        // 尝试 Array 直连
        let arr: PyRef<'_, PyArray> = match other.extract() {
            Ok(arr) => arr,
            Err(_) => {
                // 非 Array：list/tuple/scalar 直接作 ms.array 输入（dtype 沿用 mat）；
                // numpy ndarray 无 sequence 类型但支持 .tolist()，走第二条路径。
                let dtype = Some(crate::dtype::PyDtype(self.inner.dtype()));
                if let Ok(temp) = crate::ops::array(py, other, dtype, None) {
                    return self.matmul_inner_array(&temp.inner);
                }
                let list = other.call_method0("tolist").map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "csr @ : rhs must be ms.Array or numpy.ndarray/list",
                    )
                })?;
                let temp = crate::ops::array(py, &list, dtype, None)?;
                return self.matmul_inner_array(&temp.inner);
            }
        };
        self.matmul_inner(&arr)
    }

    /// `toarray()` — 物化稠密 Array。
    fn toarray(&self) -> PyResult<PyArray> {
        let result = musapy_ops::toarray(&self.inner).map_err(error::to_pyerr)?;
        Ok(PyArray::from_array(result))
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[getter]
    fn dtype(&self) -> PyDtype {
        PyDtype(self.inner.dtype())
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_device(self.inner.device().clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "CsrMatrix(shape={:?}, dtype={}, nnz={}, device={})",
            self.inner.shape(),
            self.inner.dtype(),
            self.inner.nnz(),
            self.inner.device()
        )
    }
}

impl PyCsrMatrix {
    /// 内部 spmv/spmm 分派（按 rhs ndim；PyArray 引用版本）。
    fn matmul_inner(&self, rhs: &PyRef<'_, PyArray>) -> PyResult<PyArray> {
        self.matmul_inner_array(&rhs.inner)
    }

    /// 内部 spmv/spmm 分派（按 rhs ndim；core Array 版本）。
    fn matmul_inner_array(&self, rhs: &musapy_core::Array) -> PyResult<PyArray> {
        let result = match rhs.shape().len() {
            1 => musapy_ops::spmv(&self.inner, rhs),
            2 => musapy_ops::spmm(&self.inner, rhs),
            nd => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "csr @ : rhs must be 1D (vec) or 2D (dense), got {nd}D"
                )))
            }
        };
        Ok(PyArray::from_array(result.map_err(error::to_pyerr)?))
    }
}

/// `ms.sparse.spmv(mat, vec)` — 显式函数形式（Python 层包装）。
#[pyfunction]
pub fn spmv(mat: &PyCsrMatrix, vec: &PyArray) -> PyResult<PyArray> {
    let result = musapy_ops::spmv(&mat.inner, &vec.inner).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}

/// `ms.sparse.spmm(mat, dense)` — 显式函数形式（Python 层包装）。
#[pyfunction]
pub fn spmm(mat: &PyCsrMatrix, dense: &PyArray) -> PyResult<PyArray> {
    let result = musapy_ops::spmm(&mat.inner, &dense.inner).map_err(error::to_pyerr)?;
    Ok(PyArray::from_array(result))
}
