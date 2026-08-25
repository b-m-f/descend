//! # Descend Host
//!
//! `descend_host` provides the stable host-side orchestration API for the Descend language.
//! It serves as the primary runtime interface for CPU-bound Rust programs to safely and seamlessly
//! launch, manage, and interact with compiled Descend GPU kernels.

pub use cudarc;
use cudarc::driver::{CudaContext, CudaSlice, DeviceRepr};
use std::sync::Arc;

#[derive(Clone)]
pub struct DescendContext(pub Arc<CudaContext>);

impl Default for DescendContext {
    fn default() -> Self {
        Self::new(0)
    }
}

impl DescendContext {
    pub fn new(device_id: u32) -> Self {
        Self(CudaContext::new(device_id as usize).expect("Failed to create CUDA context"))
    }
}

// TODO: both needed?
pub struct CpuShrd<T: Clone + Default + DeviceRepr, const N: usize>(pub Box<[T; N]>);
pub struct CpuUniq<T: Clone + Default + DeviceRepr, const N: usize>(pub Box<[T; N]>);

pub struct GpuShrd<T: DeviceRepr, const N: usize> {
    pub dev_buf: CudaSlice<T>,
}

pub struct GpuUniq<T: DeviceRepr, const N: usize> {
    pub dev_buf: CudaSlice<T>,
}

impl<T: Clone + Default + DeviceRepr, const N: usize> CpuShrd<T, N> {
    pub fn new(arr: Box<[T; N]>) -> Self {
        CpuShrd(arr)
    }
    pub fn to_gpu(&self, context: &DescendContext) -> GpuShrd<T, N> {
        let stream = context.0.default_stream();
        let dev_buf = stream.clone_htod(&*self.0).expect("Failed HtoD");
        let dev_buf: CudaSlice<T> = dev_buf.try_into().unwrap();
        stream.synchronize().unwrap();
        GpuShrd { dev_buf }
    }
}

impl<T: Clone + Default + DeviceRepr, const N: usize> CpuUniq<T, N> {
    pub fn new(arr: Box<[T; N]>) -> Self {
        CpuUniq(arr)
    }
    pub fn to_gpu(&self, context: &DescendContext) -> GpuUniq<T, N> {
        let stream = context.0.default_stream();
        let dev_buf = stream.clone_htod(&*self.0).expect("Failed HtoD");
        let dev_buf: CudaSlice<T> = dev_buf.try_into().unwrap();
        stream.synchronize().unwrap();
        GpuUniq { dev_buf }
    }
}

impl<T: DeviceRepr + Clone + Default, const N: usize> GpuUniq<T, N> {
    pub fn to_cpu(&self) -> CpuUniq<T, N> {
        let stream = self.dev_buf.stream();
        let vec = stream.clone_dtoh(&self.dev_buf).expect("Failed DtoH");
        // TODO: dont hide this. maybe async?
        // TODO: CUDA events?
        stream.synchronize().unwrap();
        let boxed_slice = vec.into_boxed_slice();
        let boxed_array: Box<[T; N]> = boxed_slice
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to convert slice to array"));
        CpuUniq(boxed_array)
    }
}

// Deref
impl<T: Clone + Default + DeviceRepr, const N: usize> std::ops::Deref for CpuUniq<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T: Clone + Default + DeviceRepr, const N: usize> std::ops::DerefMut for CpuUniq<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<T: Clone + Default + DeviceRepr, const N: usize> std::ops::Deref for CpuShrd<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
