use descend_host::{GpuBuffer, Kernel, KernelLaunchConfig, KernelLauncher, Stream};

fn buffer_dropped_before_launch(stream: &Stream, launcher: &KernelLauncher) {
    let config = KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut kernel = Kernel::new("kernel", config);
    let buffer = GpuBuffer::from_host(&[0i32; 4], stream).unwrap();
    kernel.push_arg(&buffer).unwrap();
    drop(buffer);
    let _ = unsafe { launcher.launch(kernel) };
}

fn main() {}
