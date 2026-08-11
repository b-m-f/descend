use std::sync::Arc;
use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy};

pub struct CpuShrd<T: Clone + Default + DeviceCopy, const N: usize>(pub Box<[T; N]>);
pub struct CpuUniq<T: Clone + Default + DeviceCopy, const N: usize>(pub Box<[T; N]>);

pub struct GpuShrd<T: DeviceCopy, const N: usize> { 
    pub dev_buf: DeviceBuffer<T> 
}
pub struct GpuUniq<T: DeviceCopy, const N: usize> { 
    pub dev_buf: DeviceBuffer<T> 
}

impl<T: Clone + Default + DeviceCopy, const N: usize> CpuShrd<T, N> {
    pub fn new(arr: Box<[T; N]>) -> Self {
        CpuShrd(arr)
    }

    pub fn to_gpu(&self, ctx: &Arc<CudaContext>) -> GpuShrd<T, N> {
        let stream = ctx.new_stream().expect("Failed to create stream");
        let dev_buf = DeviceBuffer::from_host(&stream, &*self.0).expect("Failed to copy HtoD");
        stream.synchronize().expect("Failed to synchronize stream");
        GpuShrd { dev_buf }
    }
}

impl<T: Clone + Default + DeviceCopy, const N: usize> CpuUniq<T, N> {
    pub fn new(arr: Box<[T; N]>) -> Self {
        CpuUniq(arr)
    }

    pub fn to_gpu(&self, ctx: &Arc<CudaContext>) -> GpuUniq<T, N> {
        let stream = ctx.new_stream().expect("Failed to create stream");
        let dev_buf = DeviceBuffer::from_host(&stream, &*self.0).expect("Failed to copy HtoD");
        stream.synchronize().expect("Failed to synchronize stream");
        GpuUniq { dev_buf }
    }
}

impl<T: DeviceCopy + Clone + Default, const N: usize> GpuUniq<T, N> {
    pub fn to_cpu(&self, ctx: &Arc<CudaContext>) -> CpuUniq<T, N> {
        let stream = ctx.new_stream().expect("Failed to create stream");
        let mut vec = vec![T::default(); N];
        self.dev_buf.copy_to_host(&stream, &mut vec).expect("Failed to copy DtoH");
        stream.synchronize().expect("Failed to synchronize stream");
        
        let boxed_slice = vec.into_boxed_slice();
        let boxed_array: Box<[T; N]> = boxed_slice.try_into().unwrap_or_else(|_| panic!("Failed to convert slice to array"));
        CpuUniq(boxed_array)
    }
}

// Deref is intentionally NOT implemented for GpuShrd and GpuUniq to prevent accidental host accesses.

// Implement Deref/DerefMut for CPU arrays for convenience
impl<T: Clone + Default + DeviceCopy, const N: usize> std::ops::Deref for CpuUniq<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T: Clone + Default + DeviceCopy, const N: usize> std::ops::DerefMut for CpuUniq<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<T: Clone + Default + DeviceCopy, const N: usize> std::ops::Deref for CpuShrd<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
