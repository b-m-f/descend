use crate::backend::DeviceBuffer;
use crate::context::Context;
use crate::error::HostError;
use crate::stream::Stream;
use crate::traits::{DescendType, DescendTypeWrapper};

/// Device storage holding exactly one value of the contents type `T`.
///
/// `T` is the buffer's whole contents type, so an array buffer is
/// `GpuBuffer<[i32; 1024]>` and its size is part of its type. The device
/// allocation is one value of `T`, and its device address is the address of
/// that value, which for an array is the address of its first element — the
/// address a kernel's pointer parameter expects.
pub struct GpuBuffer<T: DescendType> {
    pub(crate) dev_buf: DeviceBuffer<DescendTypeWrapper<T>>,
}

/// A device buffer associated with an exclusively borrowed host value.
///
/// The host value is used for the initial upload and can later receive an
/// explicit readback without allocating another host region. The exclusive
/// borrow prevents host access while the buffer may be used by a kernel.
pub struct BorrowedGpuBuffer<'a, T: DescendType> {
    buffer: GpuBuffer<T>,
    host: &'a mut T,
}

impl<T: DescendType> GpuBuffer<T> {
    pub fn context(&self) -> Context {
        Context {
            inner: crate::backend::buffer_context(&self.dev_buf),
        }
    }

    pub fn validate_buffer_params() -> Result<(), HostError> {
        let size = std::mem::size_of::<T>();
        if size == 0 {
            return Err(HostError::InvalidSize("Zero-sized types are not supported"));
        }
        let align = std::mem::align_of::<T>();
        if align > 256 {
            return Err(HostError::InvalidAlignment(
                "Alignment > 256 is not supported by driver",
            ));
        }
        Ok(())
    }

    /// Allocates device storage and uploads the single value `data`.
    ///
    /// Validation, context binding and device allocation failures are returned
    /// as errors. Once `data` has been handed to the driver, a failure means the
    /// copy's completion is unknown, and the process terminates while `data` is
    /// still borrowed.
    fn upload(data: &T, stream: &Stream) -> Result<Self, HostError> {
        Self::validate_buffer_params()?;

        crate::backend::bind_context_to_thread(&stream.context.inner)
            .map_err(|e| HostError::driver(e, "bind context for from_host"))?;

        let mut dev_buf = crate::backend::buffer_uninitialized(&stream.inner, 1)
            .map_err(|e| HostError::driver(e, "device allocation"))?;

        // SAFETY: DescendTypeWrapper<T> is repr(transparent) over T, so the
        // values have identical layout.
        let source: &DescendTypeWrapper<T> =
            unsafe { &*std::ptr::from_ref(data).cast::<DescendTypeWrapper<T>>() };

        if crate::backend::buffer_copy_from_host(
            &mut dev_buf,
            &stream.inner,
            std::slice::from_ref(source),
        )
        .is_err()
        {
            crate::backend::fail_stop("host-to-device transfer");
        }
        Ok(Self { dev_buf })
    }

    /// Copies the device contents into a new host allocation.
    pub fn to_host(&self, stream: &Stream) -> Result<Box<T>, HostError> {
        // SAFETY: the DescendType contract requires the all-zero bit pattern to
        // be a valid T, so a zeroed T is initialized.
        let mut host = unsafe { Box::<T>::new_zeroed().assume_init() };
        self.copy_into(stream, &mut host, "device-to-host transfer")?;
        Ok(host)
    }

    fn copy_into(
        &self,
        stream: &Stream,
        destination: &mut T,
        operation: &'static str,
    ) -> Result<(), HostError> {
        if self.context().inner != stream.context.inner {
            return Err(HostError::ContextMismatch(
                "Stream context does not match buffer context",
            ));
        }
        crate::backend::bind_context_to_thread(&stream.context.inner)
            .map_err(|e| HostError::driver(e, "bind context for readback"))?;

        // SAFETY: DescendTypeWrapper<T> is transparent over T, so the values
        // have identical layout. `destination` stays borrowed until this
        // function returns or the process terminates.
        let destination: &mut DescendTypeWrapper<T> =
            unsafe { &mut *std::ptr::from_mut(destination).cast::<DescendTypeWrapper<T>>() };
        if crate::backend::buffer_copy_to_host(
            &self.dev_buf,
            &stream.inner,
            std::slice::from_mut(destination),
        )
        .is_err()
        {
            crate::backend::fail_stop(operation);
        }
        Ok(())
    }
}

/// Array constructors. `N` is unified from the array argument and never written
/// by a caller, so no const generic appears in a public type or signature.
impl<T: DescendType, const N: usize> GpuBuffer<[T; N]> {
    /// Allocates device storage and uploads `data`.
    ///
    /// Validation, context binding and device allocation failures are returned
    /// as errors. Once `data` has been handed to the driver, a failure means the
    /// copy's completion is unknown, and the process terminates while `data` is
    /// still borrowed.
    pub fn from_host(data: &[T; N], stream: &Stream) -> Result<Self, HostError> {
        Self::upload(data, stream)
    }

    /// Uploads `data` and retains its exclusive borrow for optional in-place
    /// readback through [`BorrowedGpuBuffer::read_back`].
    pub fn from_host_mut<'a>(
        data: &'a mut [T; N],
        stream: &Stream,
    ) -> Result<BorrowedGpuBuffer<'a, [T; N]>, HostError> {
        let buffer = Self::upload(data, stream)?;
        Ok(BorrowedGpuBuffer { buffer, host: data })
    }
}

impl<'a, T: DescendType> BorrowedGpuBuffer<'a, T> {
    pub fn context(&self) -> Context {
        self.buffer.context()
    }

    pub fn gpu_buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    pub fn gpu_buffer_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.buffer
    }

    /// Copies the device contents into the originally borrowed host value and
    /// returns that exact reference. No host allocation is performed.
    pub fn read_back(self, stream: &Stream) -> Result<&'a mut T, HostError> {
        self.buffer
            .copy_into(stream, self.host, "in-place device-to-host transfer")?;
        Ok(self.host)
    }
}

impl<'a, T: DescendType> std::ops::Deref for BorrowedGpuBuffer<'a, T> {
    type Target = GpuBuffer<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<'a, T: DescendType> std::ops::DerefMut for BorrowedGpuBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
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
    struct ZST;
    unsafe impl DescendType for ZST {}

    #[test]
    fn test_reject_zst() {
        assert!(matches!(
            GpuBuffer::<[ZST; 10]>::validate_buffer_params(),
            Err(HostError::InvalidSize("Zero-sized types are not supported"))
        ));
    }

    #[derive(Clone, Copy)]
    #[repr(align(512))]
    struct OverAligned(#[allow(dead_code)] i32);
    unsafe impl DescendType for OverAligned {}

    #[test]
    fn test_reject_over_aligned() {
        assert!(matches!(
            GpuBuffer::<[OverAligned; 10]>::validate_buffer_params(),
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

        // Zeroed pages are mapped lazily, so this reserves address space only.
        let layout = std::alloc::Layout::new::<[u8; N]>();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<[u8; N]>();
        if ptr.is_null() {
            println!("skipping: host could not reserve {N} bytes");
            return;
        }
        // SAFETY: allocated with the layout of [u8; N] and zero-initialized.
        let host = unsafe { Box::from_raw(ptr) };

        match GpuBuffer::from_host(&host, &stream) {
            Err(HostError::Driver(_, "device allocation")) => {}
            Err(other) => panic!("expected a device allocation error, got {other:?}"),
            Ok(_) => panic!("a {N}-byte allocation unexpectedly succeeded"),
        }
        drop(host);

        let small = [7u8; 64];
        let buf = GpuBuffer::from_host(&small, &stream).unwrap();
        assert_eq!(*buf.to_host(&stream).unwrap(), small);
    }
}
