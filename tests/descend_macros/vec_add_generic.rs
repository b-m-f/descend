use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_launch};

descend! {
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
    let ctx = Context::new(0);
    let stream = ctx.new_stream();

    let a_array = [20i32; 16384];
    let b_array = [10i32; 16384];

    let mut a_gpu = GpuBuffer::from_host(&Box::new(a_array), &stream);
    let b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream);

    descend_launch!(gpu_vec_add, &stream, <N: 16384>, &mut a_gpu, &b_gpu);

    let a_cpu_back = a_gpu.to_host(&stream);
    assert_eq!(*a_cpu_back, [30i32; 16384]);
}
