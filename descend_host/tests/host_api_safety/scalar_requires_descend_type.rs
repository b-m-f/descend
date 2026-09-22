use descend_host::{Kernel, KernelLaunchConfig};

fn main() {
    let config = KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut kernel = Kernel::new("kernel", config);
    let value = String::from("x");
    // A `String` owns a host allocation and is not a Descend-representable value.
    let _ = kernel.push_scalar(&value);
}
