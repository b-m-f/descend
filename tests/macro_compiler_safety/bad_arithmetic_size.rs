use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_launch};

fn main() {
    descend! {
        fn gpu_bad_arithmetic<N: nat, M: nat, r: prv>(
            a_array: &r uniq gpu.global [i32; (N * M)],
            b_array: &r shrd gpu.global [i32; (N * M)]
        ) -[grid: gpu.grid<X<64>, X<256>>]-> () {
            sched block in grid {
                sched thread in block {
                    let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                    *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
                }
            }
        }
    }

    let ctx = Context::new(0);
    let stream = ctx.new_stream();
    let a_array = [20i32; 100];
    let mut a_gpu = GpuBuffer::to_gpu(&Box::new(a_array), &stream);
    let b_array = [20i32; 100];
    let b_gpu = GpuBuffer::to_gpu(&Box::new(b_array), &stream);

    // Bad Size! 10 * 20 = 200, but our GpuBuffer is 100.
    descend_launch!(gpu_bad_arithmetic, &stream, <N: 10, M: 20>, &mut a_gpu, &b_gpu);
}
