//! Private facade for the selected CUDA backend.
//!
//! Keeping these names private prevents backend types from becoming part of the
//! supported application interface and gives a replacement backend one module
//! to target.
//!
//! Host-side modules use the backend's typed abstractions directly. Raw driver
//! access remains only where no suitable abstraction exists.

pub(crate) use cuda_core::simt::{
    launch_kernel_on_stream, CudaContext, CudaModule, CudaStream, DeviceBuffer, DeviceCopy,
};

#[cfg(test)]
use cuda_core::simt::{DriverError, IntoResult};
#[cfg(test)]
use cuda_core::sys;

#[cfg(test)]
pub(crate) fn free_device_memory(context: &CudaContext) -> Result<usize, DriverError> {
    context.bind_to_thread()?;
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: both pointers are valid for writes and the context is current.
    unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }.result()?;
    Ok(free)
}
