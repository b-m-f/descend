use descend_host::{Context, Kernel, KernelLaunchConfig, KernelLauncher};

fn main() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let launcher = KernelLauncher::new(&stream, "dummy ptx").unwrap();
    let config = KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let kernel = Kernel::new("dummy", config);
    let _ = launcher.launch(kernel);
}
