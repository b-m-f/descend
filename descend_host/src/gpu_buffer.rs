use crate::context::Context;
use crate::stream::Stream;
use cuda_core::simt::{DeviceBuffer, DeviceCopy};

pub struct GpuBuffer<T: DeviceCopy, const N: usize> {
    pub(crate) dev_buf: DeviceBuffer<T>,
}

use crate::error::HostError;

impl<T: DeviceCopy, const N: usize> GpuBuffer<T, N> {
    pub fn context(&self) -> Context {
        Context {
            inner: self.dev_buf.context().clone(),
        }
    }

    pub fn from_host(data: &[T; N], stream: &Stream) -> Result<Self, HostError> {
        if N == 0 {
            return Err(HostError::InvalidSize("N == 0 is not supported"));
        }
        let size = std::mem::size_of::<T>();
        if size == 0 {
            return Err(HostError::InvalidSize("Zero-sized types are not supported"));
        }
        if size.checked_mul(N).is_none() {
            return Err(HostError::InvalidSize("Byte size overflow"));
        }
        
        // cuda-core guarantees alignment to 256 for malloc, but let's just reject anything > 256
        let align = std::mem::align_of::<T>();
        if align > 256 {
            return Err(HostError::InvalidAlignment("Alignment > 256 is not supported by driver"));
        }

        stream.context.inner.bind_to_thread().map_err(|e| HostError::Driver(e, "bind context for from_host"))?;

        let dev_buf = DeviceBuffer::from_host(&stream.inner, data)
            .map_err(|e| HostError::Driver(e, "host-to-device transfer"))?;
        Ok(Self { dev_buf })
    }

    pub fn to_host(&self, stream: &Stream) -> Result<Box<[T; N]>, HostError> {
        if !std::sync::Arc::ptr_eq(&self.context().inner, &stream.context.inner) {
            return Err(HostError::ContextMismatch("Stream context does not match buffer context"));
        }
        stream.context.inner.bind_to_thread().map_err(|e| HostError::Driver(e, "bind context for to_host"))?;

        let host_vec = self
            .dev_buf
            .to_host_vec(&stream.inner)
            .map_err(|e| HostError::Driver(e, "device-to-host transfer"))?;
        Ok(host_vec
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Size mismatch on array conversion")))
    }
}

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    use super::*;
    #[test]
    fn test_accept_static_arrays() {
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        // 1. Stack Array Reference
        let stack_arr = [1i32; 10];
        let _buf1 = GpuBuffer::from_host(&stack_arr, &stream).unwrap();

        // 2. Heap Array Reference (tests Deref coercion)
        let bx: Box<[i32; 10]> = vec![1i32; 10].into_boxed_slice().try_into().unwrap();
        let _buf2 = GpuBuffer::from_host(&bx, &stream).unwrap();
    }
}
