use descend_macros::descend;

// Mock gpu module to make type checking work for our generated Rust wrapper
pub mod gpu {
    pub struct GpuBuffer<T> {
        pub device_ptr: *mut T,
        pub size: usize,
    }

    impl<T> GpuBuffer<T> {
        pub fn new(size: usize) -> Self {
            GpuBuffer {
                device_ptr: std::ptr::null_mut(), // In a real API, this allocates on the GPU
                size,
            }
        }
    }
}

descend! {
    fn gpu_vec_add<r: prv>(
        a_array: &r uniq gpu.global [i32; 16384],
        b_array: &r shrd gpu.global [i32; 16384]
    ) -[grid: gpu.grid<X<64>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
            }
        }
    }
}

fn main() {
    println!("Compiled PTX size: {} bytes", GPU_VEC_ADD_PTX.len());

    let mut a = gpu::GpuBuffer::<i32>::new(16384);
    let b = gpu::GpuBuffer::<i32>::new(16384);

    // Call the generated strongly typed launcher
    // This launcher uses raw CUDA driver API to load and launch the PTX
    // It is a mock of how `cuda-oxide` would execute it under the hood.
    // We wrap it in unsafe (if needed) but the wrapper is declared safe in our macro!

    // Uncommenting the below would attempt to run it on a real GPU!
    gpu_vec_add(&mut a, &b);

    println!("Successfully generated strongly typed macro wrapper and embedded PTX!");
}
