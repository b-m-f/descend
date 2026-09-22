use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

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

    descend_fn!(gpu_vec_add_bad_size as vec_add_16k, <N: 16384>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let a_array = [20i32; 1024];
    let b_array = [10i32; 1024];

    let mut a_gpu = GpuBuffer::from_host(&Box::new(a_array), &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream).unwrap();

    // The instantiation fixed N at 16384, so its parameters are
    // `GpuBuffer<[i32; 16384]>` and these 1024-element buffers do not fit.
    let _ = vec_add_16k(&stream, &mut a_gpu, &b_gpu);
}
