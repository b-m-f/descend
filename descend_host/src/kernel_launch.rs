use crate::gpu_buffer::GpuBuffer;
use crate::stream::Stream;
use cuda_core::simt::CudaModule;
use cuda_core::simt::DeviceCopy;
use std::sync::Arc;

pub struct KernelLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

pub struct Kernel<'a> {
    pub fn_name: &'a str,
    pub config: KernelLaunchConfig,
    pub cu_params: Vec<u64>,
}

impl<'a> Kernel<'a> {
    pub fn new(fn_name: &'a str, config: KernelLaunchConfig) -> Self {
        Self {
            fn_name,
            config,
            cu_params: Vec::new(),
        }
    }

    pub fn push_arg<T: DeviceCopy, const N: usize>(&mut self, buf: &GpuBuffer<T, N>) {
        self.cu_params.push(buf.dev_buf.cu_deviceptr());
    }
}

pub struct KernelLauncher<'a> {
    stream: &'a Stream,
    module: Arc<CudaModule>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(stream: &'a Stream, ptx_str: &str) -> Self {
        let module = stream
            .context
            .inner
            .load_module_from_ptx_src(ptx_str)
            .expect("Failed to load PTX");
        Self { stream, module }
    }

    pub fn launch(&self, mut kernel: Kernel) {
        let f = self
            .module
            .load_function(kernel.fn_name)
            .expect("Failed to get function");

        let mut ptrs: Vec<*mut std::ffi::c_void> = kernel
            .cu_params
            .iter_mut()
            .map(|p| p as *mut _ as *mut std::ffi::c_void)
            .collect();

        unsafe {
            cuda_core::simt::launch_kernel_on_stream(
                &f,
                kernel.config.grid_dim,
                kernel.config.block_dim,
                kernel.config.shared_mem_bytes,
                &self.stream.inner,
                &mut ptrs,
            )
        }
        .expect("Kernel launch failed");
    }
}
