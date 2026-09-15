use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_launch};

// Note: This file is run via trybuild to ensure it fails to compile.
fn main() {
    descend! {
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
    }

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let a_array = [20i32; 1024];
    let b_array = [10i32; 1024];

    let mut a_gpu = GpuBuffer::from_host(&Box::new(a_array), &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream).unwrap();

    // This will trigger the compile error E0308 because N: 16384 does not match GpuBuffer<i32, 1024>
    let _ = unsafe { descend_launch!(gpu_vec_add_bad_size, &stream, <N: 16384>, &mut a_gpu, &b_gpu) };
}
