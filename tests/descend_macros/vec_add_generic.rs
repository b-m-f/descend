use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

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

// `N` is only a literal here, so this is where the kernel becomes a function.
descend_fn!(gpu_vec_add as vec_add_16k, <N: 16384>);

#[test]
fn test_vec_add_generic() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let a_array = [20i32; 16384];
    let b_array = [10i32; 16384];

    let mut a_gpu = GpuBuffer::from_host(&Box::new(a_array), &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream).unwrap();

    vec_add_16k(&stream, &mut a_gpu, &b_gpu).unwrap();

    let a_cpu_back = a_gpu.to_host(&stream).unwrap();
    assert_eq!(*a_cpu_back, [30i32; 16384]);
}
