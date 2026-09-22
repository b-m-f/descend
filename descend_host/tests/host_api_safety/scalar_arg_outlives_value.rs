use descend_host::{Kernel, KernelLaunchConfig, KernelLauncher};

fn config() -> KernelLaunchConfig {
    KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

// The by-value parallel to `kernel_arg_outlives_buffer`: a pushed value must
// outlive the kernel. A `DescendType` is `Copy`, so it cannot be moved out of;
// the borrow ends instead by the value going out of scope.
fn value_dropped_before_launch(launcher: &KernelLauncher) {
    let mut kernel = Kernel::new("kernel", config());
    {
        let value = 7i32;
        kernel.push_scalar(&value).unwrap();
    }
    let _ = unsafe { launcher.launch(kernel) };
}

fn scale() -> f32 {
    2.0
}

// A computed temporary cannot be pushed either: the caller must bind a local.
// (A borrow of a literal such as `&2.0f32` is a different case: rvalue static
// promotion gives it `'static`, so it outlives the kernel and is accepted.)
fn temporary_pushed(launcher: &KernelLauncher) {
    let mut kernel = Kernel::new("kernel", config());
    kernel.push_scalar(&scale()).unwrap();
    let _ = unsafe { launcher.launch(kernel) };
}

fn main() {}
