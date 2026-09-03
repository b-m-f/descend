use descend_host::{CpuShrd, CpuUniq, DescendContext, GpuShrd, GpuUniq};
use descend_macros::{descend, descend_launch};

<<<<<<< HEAD
=======
// -> generic descend
<<<<<<< HEAD
>>>>>>> d0a3da3 ([feedback]: notes from call)
descend_kernel! {
=======
descend! {
>>>>>>> dfb691b (refactor)
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
}

#[test]
fn test_vec_add_generic() {
    let ctx = DescendContext::new(0);

    // only 1 needed
    // borrow decides uniq/shrd
    let a_array = CpuUniq::new(Box::new([20i32; 16384]));
    let b_array = CpuShrd::new(Box::new([10i32; 16384]));

    let mut a_gpu: GpuUniq<i32, 16384> = a_array.to_gpu(&ctx);
    let b_gpu: GpuShrd<i32, 16384> = b_array.to_gpu(&ctx);
<<<<<<< HEAD
    descend_launch!(gpu_vec_add<N: 16384>, &mut a_gpu, &b_gpu);
=======

    // TODO:
    descend_launch!(gpu_vec_add, <N: 16384>, &mut a_gpu, &b_gpu);
>>>>>>> 04da9b5 (wip multi kernel)

    let a_cpu_back: CpuUniq<i32, 16384> = a_gpu.to_cpu();
    assert_eq!(*a_cpu_back, [30i32; 16384]);
}
