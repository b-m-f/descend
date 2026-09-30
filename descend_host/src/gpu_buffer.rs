use crate::backend::DeviceBuffer;
use crate::context::Context;
use crate::error::HostError;
use crate::stream::Stream;
use crate::traits::{DescendType, DescendTypeWrapper};

pub struct GpuBuffer<T: DescendType> {
    pub(crate) dev_buf: DeviceBuffer<DescendTypeWrapper<T>>,
}

impl<T: DescendType> GpuBuffer<T> {
    pub fn context(&self) -> Context {
        Context {
            inner: self.dev_buf.context().clone(),
        }
    }

    pub fn validate_buffer_params() -> Result<(), HostError> {
        let size = std::mem::size_of::<T>();
        if size == 0 {
            return Err(HostError::InvalidSize("Zero-sized types are not supported"));
        }
        Ok(())
    }

    fn upload(data: &T, stream: &Stream) -> Result<Self, HostError> {
        Self::validate_buffer_params()?;

        stream
            .context
            .inner
            .bind_to_thread()
            .map_err(|e| HostError::driver(e, "bind context for from_host"))?;

        let source: &DescendTypeWrapper<T> =
            unsafe { &*std::ptr::from_ref(data).cast::<DescendTypeWrapper<T>>() };

        let dev_buf = DeviceBuffer::from_host(&stream.inner, std::slice::from_ref(source))
            .map_err(|e| HostError::driver(e, "device buffer upload"))?;
        Ok(Self { dev_buf })
    }

    pub fn to_host(&self, stream: &Stream) -> Result<Box<T>, HostError> {
        if self.context().inner != stream.context.inner {
            return Err(HostError::ContextMismatch(
                "Stream context does not match buffer context",
            ));
        }
        stream
            .context
            .inner
            .bind_to_thread()
            .map_err(|e| HostError::driver(e, "bind context for readback"))?;

        let host = self
            .dev_buf
            .to_host_vec(&stream.inner)
            .map_err(|e| HostError::driver(e, "device-to-host transfer"))?;
        let actual_len = host.len();
        let host: Box<[DescendTypeWrapper<T>; 1]> = host
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| {
                panic!(
                    "DeviceBuffer::to_host_vec returned {actual_len} values for a GpuBuffer; expected exactly 1"
                )
            });

        let host = Box::into_raw(host).cast::<T>();
        Ok(unsafe { Box::from_raw(host) })
    }
}

impl<T: DescendType, const N: usize> GpuBuffer<[T; N]> {
    pub fn from_host(data: &[T; N], stream: &Stream) -> Result<Self, HostError> {
        Self::upload(data, stream)
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
    fn test_reject_zero_size() {
        assert!(matches!(
            GpuBuffer::<[i32; 0]>::validate_buffer_params(),
            Err(HostError::InvalidSize("Zero-sized types are not supported"))
        ));
    }

    #[derive(Clone, Copy)]
    struct ZeroSizeTrait;
    unsafe impl DescendType for ZeroSizeTrait {}

    #[test]
    fn test_reject_zst() {
        assert!(matches!(
            GpuBuffer::<[ZeroSizeTrait; 10]>::validate_buffer_params(),
            Err(HostError::InvalidSize("Zero-sized types are not supported"))
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

    #[test]
    fn test_roundtrip_preserves_values() {
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let mut data = [0u64; 4096];
        for (i, value) in data.iter_mut().enumerate() {
            *value = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        let buf = GpuBuffer::from_host(&data, &stream).unwrap();
        assert_eq!(*buf.to_host(&stream).unwrap(), data);
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct Particle {
        position: [f32; 3],
        id: u32,
    }
    unsafe impl DescendType for Particle {}

    #[test]
    fn test_roundtrip_struct_elements() {
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let mut data = [Particle {
            position: [0.0; 3],
            id: 0,
        }; 257];
        for (i, particle) in data.iter_mut().enumerate() {
            *particle = Particle {
                position: [i as f32, -(i as f32), 0.5],
                id: i as u32,
            };
        }
        let buf = GpuBuffer::from_host(&data, &stream).unwrap();
        assert_eq!(*buf.to_host(&stream).unwrap(), data);
    }

    #[test]
    fn device_oom_on_upload_is_recoverable() {
        const N: usize = 20 << 30;
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let free = crate::backend::free_device_memory(&ctx.inner).unwrap();
        if free >= N {
            println!("skipping: device has {free} free bytes, at least {N} requested");
            return;
        }

        let layout = std::alloc::Layout::new::<[u8; N]>();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<[u8; N]>();
        if ptr.is_null() {
            println!("skipping: host could not reserve {N} bytes");
            return;
        }
        let host = unsafe { Box::from_raw(ptr) };

        match GpuBuffer::from_host(&host, &stream) {
            Err(HostError::Driver(_, "device buffer upload")) => {}
            Err(other) => panic!("expected a device buffer upload error, got {other:?}"),
            Ok(_) => panic!("a {N}-byte allocation unexpectedly succeeded"),
        }
        drop(host);

        let small = [7u8; 64];
        let buf = GpuBuffer::from_host(&small, &stream).unwrap();
        assert_eq!(*buf.to_host(&stream).unwrap(), small);
    }

    fn roundtrip<T, const N: usize>(data: [T; N], stream: &Stream)
    where
        T: DescendType + PartialEq + std::fmt::Debug,
    {
        let buf = GpuBuffer::from_host(&data, stream).unwrap();
        assert_eq!(
            *buf.to_host(stream).unwrap(),
            data,
            "a roundtrip of [{}; {N}] must preserve every value",
            std::any::type_name::<T>()
        );
    }

    #[test]
    fn every_primitive_descend_type_survives_a_roundtrip() {
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        roundtrip([i8::MIN, -1, 0, 1, i8::MAX], &stream);
        roundtrip([i16::MIN, -1, 0, 1, i16::MAX], &stream);
        roundtrip([i32::MIN, -1, 0, 1, i32::MAX], &stream);
        roundtrip([i64::MIN, -1, 0, 1, i64::MAX], &stream);
        roundtrip([u8::MIN, 1, 0x7f, 0x80, u8::MAX], &stream);
        roundtrip([u16::MIN, 1, 0x7fff, 0x8000, u16::MAX], &stream);
        roundtrip([u32::MIN, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX], &stream);
        roundtrip(
            [
                u64::MIN,
                1,
                0x7fff_ffff_ffff_ffff,
                0x8000_0000_0000_0000,
                u64::MAX,
            ],
            &stream,
        );
        roundtrip([f32::MIN, -1.5, 0.0, 1.5, f32::MAX], &stream);
        roundtrip([f64::MIN, -1.5, 0.0, 1.5, f64::MAX], &stream);
        roundtrip([true, false, true, true, false], &stream);
    }

    #[test]
    fn nested_arrays_survive_a_roundtrip() {
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let mut data = [[0i32; 4]; 8];
        for (row_index, row) in data.iter_mut().enumerate() {
            for (cell_index, cell) in row.iter_mut().enumerate() {
                *cell = (row_index * 4 + cell_index) as i32;
            }
        }
        roundtrip(data, &stream);
    }

    #[test]
    fn dropping_a_buffer_releases_device_memory() {
        const N: usize = 256 << 20;
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let before = crate::backend::free_device_memory(&ctx.inner).unwrap();
        if before < 4 * N {
            println!("skipping: {before} bytes free, {} wanted", 4 * N);
            return;
        }

        let host: Box<[u8; N]> = vec![0u8; N].into_boxed_slice().try_into().unwrap();
        let buffer = GpuBuffer::from_host(&host, &stream).unwrap();

        let allocated = crate::backend::free_device_memory(&ctx.inner).unwrap();
        assert!(
            before - allocated >= N / 2,
            "a {N}-byte buffer must be visible in free device memory: before={before} allocated={allocated}"
        );

        drop(buffer);
        ctx.synchronize().unwrap();

        let released = crate::backend::free_device_memory(&ctx.inner).unwrap();
        assert!(
            released + N / 2 >= before,
            "dropping the buffer must return its device memory: before={before} released={released}"
        );
    }
}
