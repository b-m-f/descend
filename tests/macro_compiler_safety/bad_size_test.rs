use descend_host::{CpuShrd, CpuUniq, DescendContext, GpuShrd, GpuUniq};
use descend_macros::{descend_kernel, descend_launch};

// Note: This file is run via trybuild to ensure it fails to compile.
fn main() {
    descend_kernel! {
        fn gpu_vec_add_bad_size<N: nat, r: prv>(
            a_array: &r uniq gpu.global [i32; N],
            b_array: &r shrd gpu.global [i32; N]
        ) -[grid: gpu.grid<X<64>, X<256>>]-> () {
            sched block in grid {
                sched thread in block {
                    let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                    *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
                }
            }
        }
    };

    let ctx = DescendContext::new(0);

    let a_array = CpuUniq::new(Box::new([20i32; 1024]));
    let b_array = CpuShrd::new(Box::new([10i32; 1024]));

    let mut a_gpu: GpuUniq<i32, 1024> = a_array.to_gpu(&ctx);
    let b_gpu: GpuShrd<i32, 1024> = b_array.to_gpu(&ctx);

    // This will trigger the compile error E0308 because N: 16384 does not match GpuUniq<i32, 1024>
    descend_launch!(gpu_vec_add_bad_size<N: 16384>, &mut a_gpu, &b_gpu);

    let a_cpu_back: CpuUniq<i32, 1024> = a_gpu.to_cpu();
    assert_eq!(*a_cpu_back, [30i32; 1024]);
}
