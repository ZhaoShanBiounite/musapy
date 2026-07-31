//! Array：用户面对的张量类型（ADR L1-11, L3-27, L1-10）
//!
//! 职责：
//!   1. Array 结构：data(BufferRef) + layout + dtype + device + stream + resolution + name
//!   2. 基本访问器：shape/ndim/size/dtype/device/stream/name
//!   3. DeviceResolution 附着（ADR L0-8：每次解析可追溯）
//!
//! Phase 2 约束（ADR L2-3）：
//!   - 不执行任何 op（Phase 6）
//!   - 不调用 MUSA API
//!   - Array 是纯数据载体，OpBuilder（Phase 6）负责填充数据
//!
//! 设计依据：
//!   - L1-11：0-dim 无特殊路径，shape=[] 就是 0-dim
//!   - L1-10：data 用 BufferRef（只读共享），op 输出是新 Buffer
//!   - L3-27：name 存在 Array 层而非 Buffer 层（同一 buffer 可有多个不同命名的 view）
//!   - L0-8：device_resolution 附着，可追溯解析来源

use crate::buffer::BufferRef;
use crate::device::DeviceResolution;
use crate::dtype::Dtype;
use crate::layout::{Layout, Shape};
use crate::stream::Stream;
use std::fmt;
use std::sync::Arc;

// ============================================================
// 1. DtypeResolution（对称于 DeviceResolution，ADR L0-8）
// ============================================================

/// Dtype 解析记录（ADR L0-8，对称于 DeviceResolution）。
///
/// 每次数组创建时的 dtype 解析都生成此记录。
/// 与 DeviceResolution 不同，dtype 总有 fallback（float32），不会 DeviceNotConfigured。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DtypeResolution {
    pub dtype: Dtype,
    pub source: crate::device::ResolutionSource,
    pub source_location: Option<crate::device::SourceLocation>,
}

impl DtypeResolution {
    pub fn new(dtype: Dtype, source: crate::device::ResolutionSource) -> Self {
        Self {
            dtype,
            source,
            source_location: None,
        }
    }

    pub fn with_location(mut self, loc: crate::device::SourceLocation) -> Self {
        self.source_location = Some(loc);
        self
    }
}

impl fmt::Display for DtypeResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}  # resolved from: {}", self.dtype, self.source)?;
        if let Some(ref loc) = self.source_location {
            write!(f, " at {}", loc)?;
        }
        Ok(())
    }
}

// ============================================================
// 2. Array（ADR L1-11, L3-27, L1-10）
// ============================================================

/// 用户面对的张量类型（ADR L1-11）。
///
/// 字段说明：
/// - `data`：BufferRef（只读共享视图）。op 输入用此字段。
///   注意：op 输出会创建新 Buffer，再包成新 Array。
/// - `layout`：shape/strides/offset，描述数据在内存中的排列
/// - `dtype`：数据类型
/// - `device`：所属设备（从 data.device 解析，但显式存储便于快速访问）
/// - `stream`：所属流（op 执行的默认流）
/// - `device_resolution`：设备解析记录（ADR L0-8，可追溯）
/// - `dtype_resolution`：dtype 解析记录
/// - `name`：可选名称（ADR L3-27，存在 Array 层而非 Buffer 层）
///
/// **0-dim 语义**（ADR L1-11）：shape=[] 是 0-dim，无特殊路径。
/// `.item()`/`__float__` 显式触发 sync + D2H（Phase 6+ 实现）。
#[derive(Debug)]
pub struct Array {
    data: BufferRef,
    layout: Layout,
    dtype: Dtype,
    device: crate::device::Device,
    stream: Arc<Stream>,
    device_resolution: DeviceResolution,
    dtype_resolution: DtypeResolution,
    name: Option<String>,
}

impl Array {
    /// 创建 Array（完整构造，所有字段显式传入）。
    ///
    /// 通常由 `ms.array()` / op 输出构造调用。Phase 5 的 PyO3 绑定会封装更友好的 API。
    pub fn new(
        data: BufferRef,
        layout: Layout,
        dtype: Dtype,
        stream: Arc<Stream>,
        device_resolution: DeviceResolution,
        dtype_resolution: DtypeResolution,
    ) -> Self {
        let device = data.buffer().device().clone();
        Self {
            data,
            layout,
            dtype,
            device,
            stream,
            device_resolution,
            dtype_resolution,
            name: None,
        }
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.layout.shape
    }

    /// 维度数。
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// 元素总数（shape 各维度乘积）。
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// 数据类型。
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// 所属设备。
    pub fn device(&self) -> &crate::device::Device {
        &self.device
    }

    /// 所属流。
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// 内存布局。
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// 底层数据引用（只读，op 输入用）。
    pub fn data(&self) -> &BufferRef {
        &self.data
    }

    /// 设备解析记录（ADR L0-8）。
    pub fn device_resolution(&self) -> &DeviceResolution {
        &self.device_resolution
    }

    /// Dtype 解析记录。
    pub fn dtype_resolution(&self) -> &DtypeResolution {
        &self.dtype_resolution
    }

    /// 名称（ADR L3-27）。
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// 设置名称。
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// 清除名称。
    pub fn clear_name(&mut self) {
        self.name = None;
    }

    /// 是否为 0-dim（ADR L1-11）。
    pub fn is_0d(&self) -> bool {
        self.layout.ndim() == 0
    }

    /// 是否为连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// 字节大小（元素数 × 元素大小）。
    pub fn nbytes(&self) -> usize {
        self.size() * self.dtype.element_size()
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Array(shape={:?}, dtype={}, device={})",
            self.layout.shape, self.dtype, self.device
        )?;
        if let Some(ref name) = self.name {
            write!(f, ", name='{}'", name)?;
        }
        Ok(())
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::device::{Device, DeviceResolution, ResolutionSource};
    use crate::dtype::Dtype;
    use crate::layout::Layout;
    use crate::stream::Stream;

    // 测试辅助：创建一个 Array
    fn make_array(shape: Shape, dtype: Dtype, device: Device) -> Array {
        let size: usize = shape.iter().product::<usize>().max(1);
        let nbytes = size * dtype.element_size();
        let buffer = Arc::new(Buffer::placeholder(nbytes, device.clone()));
        let data = BufferRef::new(buffer);
        let layout = Layout::from_shape(shape);
        let stream = Arc::new(Stream::new(device.clone(), 0).unwrap());
        let device_res = DeviceResolution::new(device.clone(), ResolutionSource::Arg);
        let dtype_res = DtypeResolution::new(dtype, ResolutionSource::Arg);
        Array::new(data, layout, dtype, stream, device_res, dtype_res)
    }

    // --- 基本属性 ---

    #[test]
    fn array_shape_and_ndim() {
        let a = make_array(vec![2, 3, 4], Dtype::Float32, Device::Musa(0));
        assert_eq!(a.shape(), &vec![2, 3, 4]);
        assert_eq!(a.ndim(), 3);
    }

    #[test]
    fn array_size_1d() {
        let a = make_array(vec![10], Dtype::Float32, Device::Musa(0));
        assert_eq!(a.size(), 10);
    }

    #[test]
    fn array_size_3d() {
        let a = make_array(vec![2, 3, 4], Dtype::Float32, Device::Musa(0));
        assert_eq!(a.size(), 24);
    }

    #[test]
    fn array_dtype() {
        let a = make_array(vec![3], Dtype::Float64, Device::Cpu);
        assert_eq!(a.dtype(), Dtype::Float64);
    }

    #[test]
    fn array_device() {
        // 单 GPU 环境用 Musa(0)；测试目的是验证 device 字段被正确保留。
        let a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        assert_eq!(*a.device(), Device::Musa(0));
    }

    #[test]
    fn array_stream() {
        let a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        assert_eq!(*a.stream().device(), Device::Musa(0));
    }

    // --- 0-dim（ADR L1-11）---

    #[test]
    fn array_0d_shape() {
        let a = make_array(vec![], Dtype::Float32, Device::Musa(0));
        assert_eq!(a.shape(), &vec![]);
        assert_eq!(a.ndim(), 0);
        assert_eq!(a.size(), 1); // 0-dim 元素数为 1
        assert!(a.is_0d());
    }

    #[test]
    fn array_0d_not_special() {
        // 0-dim 就是一个普通 Array，没有特殊路径
        let a = make_array(vec![], Dtype::Float32, Device::Musa(0));
        assert!(a.is_0d());
        assert!(a.is_contiguous()); // 空 strides 是连续的
    }

    // --- 连续性 ---

    #[test]
    fn array_is_contiguous_default() {
        let a = make_array(vec![2, 3, 4], Dtype::Float32, Device::Musa(0));
        assert!(a.is_contiguous());
    }

    // --- 字节大小 ---

    #[test]
    fn array_nbytes_float32() {
        let a = make_array(vec![2, 3], Dtype::Float32, Device::Musa(0));
        // 6 元素 × 4 字节 = 24
        assert_eq!(a.nbytes(), 24);
    }

    #[test]
    fn array_nbytes_complex128() {
        let a = make_array(vec![4], Dtype::Complex128, Device::Musa(0));
        // 4 元素 × 16 字节 = 64
        assert_eq!(a.nbytes(), 64);
    }

    #[test]
    fn array_nbytes_0d() {
        let a = make_array(vec![], Dtype::Float32, Device::Musa(0));
        // 1 元素 × 4 字节 = 4
        assert_eq!(a.nbytes(), 4);
    }

    // --- name（ADR L3-27）---

    #[test]
    fn array_name_default_none() {
        let a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        assert!(a.name().is_none());
    }

    #[test]
    fn array_set_and_get_name() {
        let mut a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        a.set_name("weights.layer1");
        assert_eq!(a.name(), Some("weights.layer1"));
    }

    #[test]
    fn array_clear_name() {
        let mut a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        a.set_name("temp");
        a.clear_name();
        assert!(a.name().is_none());
    }

    #[test]
    fn array_set_name_overwrite() {
        let mut a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        a.set_name("first");
        a.set_name("second");
        assert_eq!(a.name(), Some("second"));
    }

    // --- resolution（ADR L0-8）---

    #[test]
    fn array_device_resolution() {
        let a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        let res = a.device_resolution();
        assert_eq!(res.device, Device::Musa(0));
        assert_eq!(res.source, ResolutionSource::Arg);
    }

    #[test]
    fn array_dtype_resolution() {
        let a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        let res = a.dtype_resolution();
        assert_eq!(res.dtype, Dtype::Float32);
        assert_eq!(res.source, ResolutionSource::Arg);
    }

    // --- Display ---

    #[test]
    fn array_display_without_name() {
        let a = make_array(vec![2, 3], Dtype::Float32, Device::Musa(0));
        let s = a.to_string();
        assert!(s.contains("shape=[2, 3]"));
        assert!(s.contains("dtype=float32"));
        assert!(s.contains("device=musa:0"));
        assert!(!s.contains("name"));
    }

    #[test]
    fn array_display_with_name() {
        let mut a = make_array(vec![2, 3], Dtype::Float32, Device::Musa(0));
        a.set_name("my_tensor");
        let s = a.to_string();
        assert!(s.contains("name='my_tensor'"));
    }

    // --- data 访问（BufferRef）---

    #[test]
    fn array_data_bufferref() {
        let a = make_array(vec![3], Dtype::Float32, Device::Musa(0));
        let data = a.data();
        assert_eq!(data.buffer().size(), 3 * 4); // 3 × f32
    }

    // --- DtypeResolution ---

    #[test]
    fn dtype_resolution_display() {
        let r = DtypeResolution::new(Dtype::Float32, ResolutionSource::GlobalDefault);
        assert_eq!(r.to_string(), "float32  # resolved from: global_default");
    }

    #[test]
    fn dtype_resolution_with_location() {
        let r = DtypeResolution::new(Dtype::Float32, ResolutionSource::Arg).with_location(
            crate::device::SourceLocation {
                file: "test.py".to_string(),
                line: 5,
                column: 0,
            },
        );
        assert_eq!(r.to_string(), "float32  # resolved from: arg at test.py:5");
    }

    #[test]
    fn dtype_resolution_equality() {
        let r1 = DtypeResolution::new(Dtype::Int32, ResolutionSource::Context);
        let r2 = DtypeResolution::new(Dtype::Int32, ResolutionSource::Context);
        assert_eq!(r1, r2);
    }

    // --- 多个 Array 共享同一 Buffer（view 场景，L3-27）---

    #[test]
    fn multiple_arrays_share_buffer_with_different_names() {
        // L3-27 rationale: 同一 buffer 可有多个不同命名的 view
        let dtype = Dtype::Float32;
        let device = Device::Musa(0);
        let nbytes = 24; // 6 × f32
        let buffer = Arc::new(Buffer::placeholder(nbytes, device.clone()));

        // view 1: shape [2, 3]
        let data1 = BufferRef::new(buffer.clone());
        let layout1 = Layout::from_shape(vec![2, 3]);
        let stream = Arc::new(Stream::new(device.clone(), 0).unwrap());
        let mut a1 = Array::new(
            data1,
            layout1,
            dtype,
            stream.clone(),
            DeviceResolution::new(device.clone(), ResolutionSource::Arg),
            DtypeResolution::new(dtype, ResolutionSource::Arg),
        );
        a1.set_name("view_2x3");

        // view 2: shape [3, 2]（转置，同一 buffer）
        let data2 = BufferRef::new(buffer);
        let layout2 = Layout::from_shape_and_strides(vec![3, 2], vec![1, 3]).unwrap();
        let mut a2 = Array::new(
            data2,
            layout2,
            dtype,
            stream,
            DeviceResolution::new(device, ResolutionSource::Arg),
            DtypeResolution::new(dtype, ResolutionSource::Arg),
        );
        a2.set_name("view_3x2_t");

        // 两个 Array 共享同一 buffer（不同 view），各自有不同 name
        assert_eq!(a1.name(), Some("view_2x3"));
        assert_eq!(a2.name(), Some("view_3x2_t"));
        assert!(a1.data() == a2.data()); // 同一 BufferRef
        assert!(a1.is_contiguous()); // view_2x3 是连续布局
        assert!(!a2.is_contiguous()); // view_3x2_t 转置后非连续
    }
}
