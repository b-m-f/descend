use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

descend! {
    fn sum_pair(a: i32, b: i32) -[t: gpu.thread]-> i32 {
        a + b
    }

    fn add_one(x: i32) -[t: gpu.thread]-> i32 {
        x + 1
    }

    fn twice(x: i32) -[t: gpu.thread]-> i32 {
        x * 2
    }

    fn gpu_vec_add<r1: prv, r2: prv>(
        a_array: &r1 uniq gpu.global [i32; 256],
        b_array: &r2 shrd gpu.global [i32; 256]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = add_one(sum_pair(*a_ref, (*b_array).to_view.grp::<256>[[block]][[thread]]))
            }
        }
    }

    fn gpu_vec_double_add<r1: prv, r2: prv>(
        a_array: &r1 uniq gpu.global [i32; 256],
        b_array: &r2 shrd gpu.global [i32; 256]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = twice(sum_pair(*a_ref, (*b_array).to_view.grp::<256>[[block]][[thread]]))
            }
        }
    }
}

#[test]
fn test_both_kernels_call_their_helpers() {
    let gpu_vec_add = descend_fn!(gpu_vec_add);
    let gpu_vec_double_add = descend_fn!(gpu_vec_double_add);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut sum_gpu = GpuBuffer::from_host(&[20i32; 256], &stream).unwrap();
    let sum_addend = GpuBuffer::from_host(&[22i32; 256], &stream).unwrap();

    gpu_vec_add(&stream, &mut sum_gpu, &sum_addend).unwrap();

    assert_eq!(*sum_gpu.to_host(&stream).unwrap(), [43i32; 256]);

    let mut doubled_gpu = GpuBuffer::from_host(&[5i32; 256], &stream).unwrap();
    let doubled_addend = GpuBuffer::from_host(&[2i32; 256], &stream).unwrap();

    gpu_vec_double_add(&stream, &mut doubled_gpu, &doubled_addend).unwrap();

    assert_eq!(*doubled_gpu.to_host(&stream).unwrap(), [14i32; 256]);
}
