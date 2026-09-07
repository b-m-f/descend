use descend_host::{DescendContext, GpuBuffer};
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

    let ctx = DescendContext::new(0);
    let stream = ctx.new_stream();

    let a_array = [20i32; 1024];
    let b_array = [10i32; 1024];

    let mut a_gpu = GpuBuffer::from_box(&stream, Box::new(a_array));
    let b_gpu = GpuBuffer::from_box(&stream, Box::new(b_array));

    // This will trigger the compile error E0308 because N: 16384 does not match GpuBuffer<i32, 1024>
    descend_launch!(gpu_vec_add_bad_size, <N: 16384>, &mut a_gpu, &b_gpu);

    let a_cpu_back = a_gpu.to_cpu(&stream);
}
