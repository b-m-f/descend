#![deny(unsafe_code)]

use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_scale<r: prv>(
        a_array: &r uniq gpu.global [i32; 256],
        b_array: &r shrd gpu.global [i32; 256]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
            }
        }
    }
}

#[test]
fn generated_function_launches_without_unsafe() {
    let gpu_scale = descend_fn!(gpu_scale);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut a_gpu = GpuBuffer::from_host(&[20i32; 256], &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&[10i32; 256], &stream).unwrap();

    gpu_scale(&stream, &mut a_gpu, &b_gpu).unwrap();

    assert_eq!(*a_gpu.to_host(&stream).unwrap(), [30i32; 256]);
}
