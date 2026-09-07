//! # Descend Host
//!
//! `descend_host` provides the stable host-side orchestration API for the Descend language.
//! It serves as the primary runtime interface for CPU-bound Rust programs to safely and seamlessly
//! launch, manage, and interact with compiled Descend GPU kernels.

pub use cuda_core;
use cuda_core::simt::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
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


pub struct KernelLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

pub fn launch_kernel(
    context: &DescendContext,
    ptx_str: &str,
    fn_name: &str,
    config: KernelLaunchConfig,
    params: &mut [*mut std::ffi::c_void]
) {
    let stream = context.0.new_stream().expect("Failed to create stream");
    let module = context.0.load_module_from_ptx_src(ptx_str).expect("Failed to load PTX");
    let f = module.load_function(fn_name).expect("Failed to get function");
    
    unsafe {
        cuda_core::simt::launch_kernel_on_stream(
            &f,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            &stream,
            params
        )
    }.expect("Kernel launch failed");
    stream.synchronize().expect("Failed to synchronize");
}

pub struct GpuBuffer<T: DeviceCopy, const N: usize> {

    pub dev_buf: DeviceBuffer<T>,
}

impl<T: DeviceCopy, const N: usize> GpuBuffer<T, N> {
    pub fn context(&self) -> DescendContext {
        DescendContext(self.dev_buf.context().clone())
    }

    /// Synchronously uploads data to the GPU and returns a buffer.
    /// Blocks the CPU thread until the transfer is complete.
    /// Takes ownership of the Box, eagerly dropping the host memory immediately after transfer.
    pub fn from_box(context: &DescendContext, host_data: Box<[T; N]>) -> Self {
        let stream = context.0.new_stream().expect("Failed to create stream");
        let dev_buf = DeviceBuffer::from_host(&stream, host_data.as_slice()).expect("Failed HtoD Sync");
        Self { dev_buf }
    }

    /// Synchronously uploads data to the GPU from a reference.
    /// Blocks the CPU thread until the transfer is complete.
    /// The caller retains ownership of the host memory.
    pub fn from_slice(context: &DescendContext, host_data: &[T; N]) -> Self {
        let stream = context.0.new_stream().expect("Failed to create stream");
        let dev_buf = DeviceBuffer::from_host(&stream, host_data.as_slice()).expect("Failed HtoD Sync");
        Self { dev_buf }
    }

    pub fn to_cpu(&self) -> Box<[T; N]> {
        let stream = self.dev_buf.context().new_stream().expect("Failed to create stream");
        let host_vec = self.dev_buf.to_host_vec(&stream).expect("Failed DtoH Sync");
        host_vec.into_boxed_slice().try_into().unwrap_or_else(|_| panic!("Failed to convert slice to array"))
    }
}

/*
// =====================================================================
// FUTURE ASYNC API (TRUE ZERO-STALL PINNED MEMORY)
// =====================================================================
// This section implements a custom smart pointer for OS-level page-locked
// memory to bypass the CUDA driver's hidden synchronous staging buffers.
// Left commented out for future thesis iterations on high-performance DMA.

// use std::ops::{Deref, DerefMut};
// use cuda_core::simt::{PinnedHostBuffer, CudaEvent};

/// A custom Smart Pointer that owns OS-level page-locked (pinned) memory.
/// Safely wraps `cuMemAllocHost` and `cuMemFreeHost`.
// pub struct DescendPinnedBox<T: DeviceCopy> {
//     inner: PinnedHostBuffer<T>,
// }

// impl<T: DeviceCopy> DescendPinnedBox<T> {
//     pub fn new_zeroed(context: &DescendContext, len: usize) -> Self {
//         Self {
//             inner: PinnedHostBuffer::zeroed(&context.0, len)
//                 .expect("Failed to allocate pinned host memory")
//         }
//     }
// }

// impl<T: DeviceCopy> Deref for DescendPinnedBox<T> {
//     type Target = [T];
//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }

// impl<T: DeviceCopy> DerefMut for DescendPinnedBox<T> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.inner
//     }
// }

// impl<T: DeviceCopy + Send + 'static, const N: usize> GpuBuffer<T, N> {
//     /// Enqueues a true Zero-Stall asynchronous DMA transfer from Pinned Host Memory.
//     /// Because the memory is page-locked by the OS, the CUDA driver does not need
//     /// to allocate a staging buffer, allowing the CPU to return instantly (< 10µs).
//     pub fn from_pinned_async(
//         context: &DescendContext, 
//         host_data: DescendPinnedBox<T>
//     ) -> (Self, CudaEvent, DescendPinnedBox<T>) {
//         let stream = context.0.new_stream().expect("Failed to create stream");
//         
//         let dev_buf = unsafe {
//             // The CUDA driver automatically detects that the underlying pointer is pinned 
//             // and bypasses the synchronous staging copy, achieving true Zero-Stall.
//             DeviceBuffer::from_host_async_unchecked(&stream, &host_data)
//                 .expect("Failed HtoD Async")
//         };
//         
//         let event = stream.record_event(None).expect("Failed to record event");
//         
//         (Self { dev_buf }, event, host_data)
//     }
// }
*/
