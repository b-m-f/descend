use descend_host::{CpuShrd, CpuUniq, DescendContext, GpuShrd, GpuUniq};
use descend_macros::{descend_kernel, descend_launch};

#[test]
fn test_vec_add_functional_generic() {
    let s = descend_kernel! {
        fn gpu_vec_add<N: nat, r: prv>(
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
    println!("{}", s);

    let ctx = DescendContext::new(0);

    let a_array = CpuUniq::new(Box::new([20i32; 16384]));
    let b_array = CpuShrd::new(Box::new([10i32; 16384]));

    let mut a_gpu: GpuUniq<i32, 16384> = a_array.to_gpu(&ctx);
    let b_gpu: GpuShrd<i32, 16384> = b_array.to_gpu(&ctx);

    descend_launch!(gpu_vec_add<N: 16384>, &mut a_gpu, &b_gpu);

    let a_cpu_back: CpuUniq<i32, 16384> = a_gpu.to_cpu(&ctx);
    assert_eq!(*a_cpu_back, [30i32; 16384]);
}
