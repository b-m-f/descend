//! # Descend Host
//!
//! `descend_host` provides the stable host-side orchestration API for the Descend language.
//! It serves as the primary runtime interface for CPU-bound Rust programs to safely and seamlessly
//! launch, manage, and interact with compiled Descend GPU kernels.

pub use cudarc;
use cudarc::driver::{CudaContext, CudaSlice, DeviceRepr, DeviceSlice};
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

<<<<<<< HEAD
// TODO: both needed?
pub struct CpuShrd<T: Clone + Default + DeviceRepr, const N: usize>(pub Box<[T; N]>);
pub struct CpuUniq<T: Clone + Default + DeviceRepr, const N: usize>(pub Box<[T; N]>);

pub struct GpuShrd<T: DeviceRepr, const N: usize> {
=======
pub struct GpuBuffer<T: DeviceRepr, const N: usize> {
>>>>>>> 7cc7078 (use standard rust types and simple transparent wrapper. add more test cases)
    pub dev_buf: CudaSlice<T>,
}

impl<T: Clone + Default + DeviceRepr, const N: usize> GpuBuffer<T, N> {
    pub fn from_slice(context: &DescendContext, slice: &[T; N]) -> Self {
        let stream = context.0.default_stream();
        let dev_buf = stream.clone_htod(slice.as_slice()).expect("Failed HtoD");
        let dev_buf: CudaSlice<T> = dev_buf.try_into().unwrap();
        stream.synchronize().unwrap();
        Self { dev_buf }
    }

    pub fn to_cpu(&self) -> Box<[T; N]> {
        let stream = self.dev_buf.stream();
        let vec = stream.clone_dtoh(&self.dev_buf).expect("Failed DtoH");
        // TODO: dont hide this. maybe async?
        // TODO: CUDA events?
        stream.synchronize().unwrap();
        vec.into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to convert slice to array"))
    }
}
