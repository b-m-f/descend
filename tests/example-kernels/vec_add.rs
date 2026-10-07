use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

#[test]
fn test_vec_add() {
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
    };

    let gpu_vec_add = descend_fn!(gpu_vec_add);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let a_array = [20i32; 16384];
    let b_array = [10i32; 16384];

    let mut a_gpu = GpuBuffer::from_host(&a_array, &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream).unwrap();

    gpu_vec_add(&stream, &mut a_gpu, &b_gpu).unwrap();

    let a_array = a_gpu.to_host(&stream).unwrap();
    assert_eq!(*a_array, [30i32; 16384]);
}
