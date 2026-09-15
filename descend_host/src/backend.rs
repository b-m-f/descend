//! Private adapter for the selected CUDA backend.
//!
//! Keeping these names private prevents backend types from becoming part of the
//! supported application interface and gives a replacement backend one module
//! to target.
//!
//! Context binding, transfers, kernel submission and stream synchronization use
//! the backend's raw driver calls. The backend's convenience methods first
//! report any error recorded by an earlier resource release and then skip the
//! driver call. On a path that must terminate when completion is unknown, such
//! a recorded error would terminate the process although the operation itself
//! did not fail.

use std::sync::Arc;

pub(crate) use cuda_core::simt::{CudaContext, CudaModule, CudaStream, DeviceBuffer, DeviceCopy};
use cuda_core::simt::{CudaFunction, DriverError, IntoResult};
use cuda_core::sys;

pub(crate) fn new_context(device_id: usize) -> Result<Arc<CudaContext>, DriverError> {
    CudaContext::new(device_id)
}

pub(crate) fn new_stream(context: &Arc<CudaContext>) -> Result<Arc<CudaStream>, DriverError> {
    context.new_stream()
}

/// Makes `context` current on the calling thread.
pub(crate) fn bind_context_to_thread(context: &CudaContext) -> Result<(), DriverError> {
    // SAFETY: `context` retains a valid primary context handle.
    unsafe { sys::cuCtxSetCurrent(context.cu_ctx()) }.result()
}

/// Waits for all work on `stream`. The stream's context must be current.
/// `Ok` means the driver established completion.
pub(crate) fn synchronize_stream(stream: &CudaStream) -> Result<(), DriverError> {
    // SAFETY: `stream` retains a valid stream handle.
    unsafe { sys::cuStreamSynchronize(stream.cu_stream()) }.result()
}

/// Reads and clears an error recorded by a backend destructor.
pub(crate) fn take_recorded_error(context: &CudaContext) -> Result<(), DriverError> {
    context.check_err()
}

/// Allocates uninitialized, stream-ordered device storage for `len` elements.
///
/// The allocation reads no host memory, and on failure no device pointer
/// exists. The caller must initialize every byte before the buffer is read.
pub(crate) fn buffer_uninitialized<T: DeviceCopy>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    // SAFETY: the only caller writes the whole buffer with a synchronizing
    // host-to-device copy and terminates if that copy fails.
    unsafe { DeviceBuffer::uninitialized_async(stream, len) }
}

/// Copies `src` into `buffer` and synchronizes `stream`. The stream's context
/// must be current. An error means completion was not established.
pub(crate) fn buffer_copy_from_host<T: DeviceCopy>(
    buffer: &mut DeviceBuffer<T>,
    stream: &CudaStream,
    src: &[T],
) -> Result<(), DriverError> {
    // SAFETY: `src` stays borrowed until the synchronization below returns, and
    // the caller does not release it before terminating on error.
    let enqueued = unsafe { buffer.copy_from_host_async_unchecked(stream, src) };
    let completed = synchronize_stream(stream);
    enqueued.and(completed)
}

/// Copies `buffer` into `destination` and synchronizes `stream`. The stream's
/// context must be current. An error means completion was not established.
pub(crate) fn buffer_copy_to_host<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    stream: &CudaStream,
    destination: &mut [T],
) -> Result<(), DriverError> {
    assert_eq!(destination.len(), buffer.len());
    // SAFETY: `destination` holds exactly `buffer.num_bytes()` bytes and stays
    // borrowed until the synchronization below returns.
    let enqueued = unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            destination.as_mut_ptr(),
            buffer.cu_deviceptr(),
            buffer.num_bytes(),
            stream.cu_stream(),
        )
    };
    let completed = synchronize_stream(stream);
    enqueued.and(completed)
}

pub(crate) fn buffer_context<T: DeviceCopy>(buffer: &DeviceBuffer<T>) -> Arc<CudaContext> {
    buffer.context().clone()
}

pub(crate) fn buffer_device_ptr<T: DeviceCopy>(buffer: &DeviceBuffer<T>) -> u64 {
    buffer.cu_deviceptr()
}

pub(crate) fn load_module_from_ptx(
    context: &Arc<CudaContext>,
    ptx_str: &str,
) -> Result<Arc<CudaModule>, DriverError> {
    context.load_module_from_ptx_src(ptx_str)
}

pub(crate) fn load_function(
    module: &Arc<CudaModule>,
    name: &str,
) -> Result<CudaFunction, DriverError> {
    module.load_function(name)
}

/// Submits a kernel on `stream`. The stream's context must be current.
///
/// # Safety
/// The obligations documented by `KernelLauncher::launch` must hold.
pub(crate) unsafe fn launch(
    function: &CudaFunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: &CudaStream,
    parameters: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    // SAFETY: `function` outlives the submission; the remaining obligations are
    // the caller's.
    unsafe {
        cuda_core::launch_kernel(
            function.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            stream.cu_stream(),
            parameters,
        )
    }
}

/// Terminates before Rust can release memory whose CUDA completion state is
/// unknown. Only called when a driver transfer, submission or synchronization
/// has itself failed after host memory or kernel arguments were handed over.
pub(crate) fn fail_stop(operation: &str) -> ! {
    eprintln!(
        "fatal descend_host error: CUDA completion could not be established during {operation}"
    );
    std::process::abort()
}

#[cfg(test)]
pub(crate) fn raw_stream_handle(stream: &CudaStream) -> usize {
    stream.cu_stream() as usize
}

#[cfg(test)]
pub(crate) fn free_device_memory(context: &CudaContext) -> Result<usize, DriverError> {
    bind_context_to_thread(context)?;
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: both pointers are valid for writes and the context is current.
    unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }.result()?;
    Ok(free)
}
