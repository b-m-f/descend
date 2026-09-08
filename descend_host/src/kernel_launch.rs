use crate::gpu_buffer::GpuBuffer;
use crate::stream::Stream;
use cuda_core::simt::DeviceCopy;

pub struct KernelLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

pub struct KernelLauncher<'a> {
    stream: &'a Stream,
    ptx_str: &'a str,
    fn_name: &'a str,
    config: KernelLaunchConfig,
    cu_params: Vec<u64>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(
        stream: &'a Stream,
        ptx_str: &'a str,
        fn_name: &'a str,
        config: KernelLaunchConfig,
    ) -> Self {
        Self {
            stream,
            ptx_str,
            fn_name,
            config,
            cu_params: Vec::new(),
        }
    }

    pub fn push_arg<T: DeviceCopy, const N: usize>(&mut self, buf: &GpuBuffer<T, N>) {
        self.cu_params.push(buf.dev_buf.cu_deviceptr());
    }

    pub fn launch(&mut self) {
        let module = self
            .stream
            .context
            .inner
            .load_module_from_ptx_src(self.ptx_str)
            .expect("Failed to load PTX");
        let f = module
            .load_function(self.fn_name)
            .expect("Failed to get function");

        let mut ptrs: Vec<*mut std::ffi::c_void> = self
            .cu_params
            .iter_mut()
            .map(|p| p as *mut _ as *mut std::ffi::c_void)
            .collect();

        unsafe {
            cuda_core::simt::launch_kernel_on_stream(
                &f,
                self.config.grid_dim,
                self.config.block_dim,
                self.config.shared_mem_bytes,
                &self.stream.inner,
                &mut ptrs,
            )
        }
        .expect("Kernel launch failed");
    }
}
