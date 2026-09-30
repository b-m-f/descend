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
    unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }.result()?;
    Ok(free)
}
