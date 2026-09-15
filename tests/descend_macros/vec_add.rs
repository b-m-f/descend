use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_launch};

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

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut a_array = [20i32; 16384];
    let b_array = [10i32; 16384];

    let original_address = a_array.as_ptr();
    let mut a_gpu = GpuBuffer::from_host_mut(&mut a_array, &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream).unwrap();

    unsafe {
        descend_launch!(gpu_vec_add, &stream, , &mut a_gpu, &b_gpu).unwrap();
    }

    let a_cpu_back = a_gpu.read_back(&stream).unwrap();
    assert_eq!(a_cpu_back.as_ptr(), original_address);
    assert_eq!(*a_cpu_back, [30i32; 16384]);
}
