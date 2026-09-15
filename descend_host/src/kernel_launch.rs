use crate::backend::CudaModule;
use crate::context::Context;
use crate::error::HostError;
use crate::gpu_buffer::GpuBuffer;
use crate::stream::Stream;
use crate::traits::DescendType;
use std::sync::Arc;

pub struct KernelLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

/// Arguments and configuration for one kernel submission.
///
/// Each pushed buffer stays borrowed for `'a`, i.e. until the `Kernel` is
/// consumed by [`KernelLauncher::launch`]. Because launch is synchronous, safe
/// code cannot drop a pushed buffer during execution, and a buffer pushed with
/// [`Kernel::push_arg_mut`] cannot be read or pushed again until completion.
pub struct Kernel<'a> {
    pub fn_name: &'a str,
    pub config: KernelLaunchConfig,
    cu_params: Vec<u64>,
    context: Option<Context>,
}

impl<'a> Kernel<'a> {
    pub fn new(fn_name: &'a str, config: KernelLaunchConfig) -> Self {
        Self {
            fn_name,
            config,
            cu_params: Vec::new(),
            context: None,
        }
    }

    /// Pushes a buffer for a parameter the kernel only reads (`shrd`).
    pub fn push_arg<T: DescendType, const N: usize>(
        &mut self,
        buf: &'a GpuBuffer<T, N>,
    ) -> Result<(), HostError> {
        self.push_device_arg(buf)
    }

    /// Pushes a buffer for a parameter the kernel may write (`uniq`).
    pub fn push_arg_mut<T: DescendType, const N: usize>(
        &mut self,
        buf: &'a mut GpuBuffer<T, N>,
    ) -> Result<(), HostError> {
        self.push_device_arg(buf)
    }

    fn push_device_arg<T: DescendType, const N: usize>(
        &mut self,
        buf: &GpuBuffer<T, N>,
    ) -> Result<(), HostError> {
        if let Some(ctx) = &self.context {
            if ctx.inner != buf.context().inner {
                return Err(HostError::ContextMismatch(
                    "Buffer context does not match existing kernel arguments context",
                ));
            }
        } else {
            self.context = Some(buf.context());
        }
        self.cu_params
            .push(crate::backend::buffer_device_ptr(&buf.dev_buf));
        Ok(())
    }
}

pub struct KernelLauncher<'a> {
    stream: &'a Stream,
    module: Arc<CudaModule>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(stream: &'a Stream, ptx_str: &str) -> Result<Self, HostError> {
        std::ffi::CString::new(ptx_str).map_err(|e| HostError::NulError(e, "PTX source"))?;
        let module = crate::backend::load_module_from_ptx(&stream.context.inner, ptx_str)
            .map_err(|e| HostError::driver(e, "load module"))?;
        Ok(Self { stream, module })
    }

    /// Submits `kernel` on this launcher's stream and waits for completion.
    ///
    /// # Safety
    ///
    /// The borrow checker already guarantees that every pushed buffer outlives
    /// this call and that buffers pushed with `push_arg_mut` are not accessed
    /// by other safe code until it returns. The caller must additionally
    /// guarantee facts that raw PTX does not expose:
    ///
    /// - **Signature**: the named kernel exists and its argument count, order,
    ///   ABI, element size, alignment, representation, extent and address space
    ///   match the pushed buffers.
    /// - **Permissions**: every parameter the kernel may write was pushed with
    ///   `push_arg_mut`, and no kernel running on another stream accesses the
    ///   same buffers in a conflicting way.
    /// - **Execution parameters**: grid, block and shared-memory sizes satisfy
    ///   the kernel's indexing assumptions.
    /// - **Output validity**: every value the kernel writes is a valid value of
    ///   its Rust `DescendType`, including the restricted encodings of `bool`.
    ///
    /// Success means all work submitted to the stream, including this kernel,
    /// has completed. If the driver's stream synchronization fails, completion
    /// is unknown and the process terminates before argument, allocation or
    /// module storage is released. Errors recorded by earlier resource releases
    /// do not cause termination; see [`Context::check_deferred_errors`].
    pub unsafe fn launch(&self, mut kernel: Kernel) -> Result<(), HostError> {
        if let Some(ctx) = &kernel.context {
            if ctx.inner != self.stream.context.inner {
                return Err(HostError::ContextMismatch(
                    "Kernel arguments context does not match execution stream context",
                ));
            }
        }
        std::ffi::CString::new(kernel.fn_name)
            .map_err(|e| HostError::NulError(e, "kernel function name"))?;
        let f = crate::backend::load_function(&self.module, kernel.fn_name)
            .map_err(|e| HostError::driver(e, "load function"))?;
        crate::backend::bind_context_to_thread(&self.stream.context.inner)
            .map_err(|e| HostError::driver(e, "bind context for kernel launch"))?;

        let mut ptrs: Vec<*mut std::ffi::c_void> = kernel
            .cu_params
            .iter_mut()
            .map(|p| p as *mut _ as *mut std::ffi::c_void)
            .collect();

        let launch_result = unsafe {
            crate::backend::launch(
                &f,
                kernel.config.grid_dim,
                kernel.config.block_dim,
                kernel.config.shared_mem_bytes,
                &self.stream.inner,
                &mut ptrs,
            )
        };

        let completion = crate::backend::synchronize_stream(&self.stream.inner);
        if completion.is_err() {
            crate::backend::fail_stop("kernel completion");
        }

        launch_result.map_err(|e| HostError::driver(e, "kernel launch"))
    }
}
