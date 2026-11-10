use descend_host::{Kernel, KernelLaunchConfig, KernelLauncher};

fn config() -> KernelLaunchConfig {
    KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn value_dropped_before_launch(launcher: &KernelLauncher) {
    let mut kernel = Kernel::new("kernel", config());
    {
        let value = 7i32;
        kernel.push_arg_val(&value).unwrap();
    }
    let _ = unsafe { launcher.launch(kernel) };
}

fn scale() -> f32 {
    2.0
}

fn temporary_pushed(launcher: &KernelLauncher) {
    let mut kernel = Kernel::new("kernel", config());
    kernel.push_arg_val(&scale()).unwrap();
    let _ = unsafe { launcher.launch(kernel) };
}

fn main() {}
