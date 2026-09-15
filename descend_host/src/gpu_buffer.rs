use crate::context::Context;
use crate::stream::Stream;
use crate::traits::{DescendType, DescendTypeWrapper};
use cuda_core::simt::DeviceBuffer;
use crate::error::HostError;

pub struct GpuBuffer<T: DescendType, const N: usize> {
    pub(crate) dev_buf: DeviceBuffer<DescendTypeWrapper<T>>,
}

impl<T: DescendType, const N: usize> GpuBuffer<T, N> {
    pub fn context(&self) -> Context {
        Context {
            inner: self.dev_buf.context().clone(),
        }
    }

    pub fn validate_buffer_params() -> Result<(), HostError> {
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
        let align = std::mem::align_of::<T>();
        if align > 256 {
            return Err(HostError::InvalidAlignment("Alignment > 256 is not supported by driver"));
        }
        Ok(())
    }

    pub fn from_host(data: &[T; N], stream: &Stream) -> Result<Self, HostError> {
        Self::validate_buffer_params()?;

        stream.context.inner.bind_to_thread().map_err(|e| HostError::driver(e, "bind context for from_host"))?;

        // Safely transmute due to #[repr(transparent)]
        let wrapper_data: &[DescendTypeWrapper<T>; N] = unsafe { std::mem::transmute(data) };

        let dev_buf = DeviceBuffer::from_host(&stream.inner, wrapper_data)
            .map_err(|e| HostError::driver(e, "host-to-device transfer"))?;
        Ok(Self { dev_buf })
    }

    pub fn to_host(&self, stream: &Stream) -> Result<Box<[T; N]>, HostError> {
        if self.context().inner != stream.context.inner {
            return Err(HostError::ContextMismatch("Stream context does not match buffer context"));
        }
        stream.context.inner.bind_to_thread().map_err(|e| HostError::driver(e, "bind context for to_host"))?;

        let host_vec = self
            .dev_buf
            .to_host_vec(&stream.inner)
            .map_err(|e| HostError::driver(e, "device-to-host transfer"))?;

        // Safely transmute Vec<DescendTypeWrapper<T>> to Vec<T>
        let mut host_vec = std::mem::ManuallyDrop::new(host_vec);
        let ptr = host_vec.as_mut_ptr() as *mut T;
        let len = host_vec.len();
        let cap = host_vec.capacity();
        let host_vec_t = unsafe { Vec::from_raw_parts(ptr, len, cap) };

        Self::validate_readback(host_vec_t)
    }

    pub fn validate_readback(host_vec_t: Vec<T>) -> Result<Box<[T; N]>, HostError> {
        host_vec_t
            .into_boxed_slice()
            .try_into()
            .map_err(|_| HostError::ConversionError("Size mismatch on array conversion"))
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

        let stack_arr = [1i32; 10];
        let _buf1 = GpuBuffer::from_host(&stack_arr, &stream).unwrap();

        let bx: Box<[i32; 10]> = vec![1i32; 10].into_boxed_slice().try_into().unwrap();
        let _buf2 = GpuBuffer::from_host(&bx, &stream).unwrap();
    }

    #[test]
    fn test_reject_zero_extent() {
        assert!(matches!(
            GpuBuffer::<i32, 0>::validate_buffer_params(),
            Err(HostError::InvalidSize("N == 0 is not supported"))
        ));
    }

    #[derive(Clone, Copy)]
    struct ZST;
    unsafe impl DescendType for ZST {}

    #[test]
    fn test_reject_zst() {
        assert!(matches!(
            GpuBuffer::<ZST, 10>::validate_buffer_params(),
            Err(HostError::InvalidSize("Zero-sized types are not supported"))
        ));
    }

    #[derive(Clone, Copy)]
    #[repr(align(512))]
    struct OverAligned(i32);
    unsafe impl DescendType for OverAligned {}

    #[test]
    fn test_reject_over_aligned() {
        assert!(matches!(
            GpuBuffer::<OverAligned, 10>::validate_buffer_params(),
            Err(HostError::InvalidAlignment(_))
        ));
    }

    #[test]
    fn test_context_match_readback() {
        let ctx0 = Context::new(0).unwrap();
        let stream0 = ctx0.new_stream().unwrap();
        
        let ctx1 = Context::new(0).unwrap();
        let stream1 = ctx1.new_stream().unwrap();

        let arr = [1i32; 10];
        let buf = GpuBuffer::from_host(&arr, &stream0).unwrap();

        let res = buf.to_host(&stream1);
        assert!(res.is_ok());
    }
}
