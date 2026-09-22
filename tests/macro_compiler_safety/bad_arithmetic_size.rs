use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

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

    descend_fn!(gpu_bad_arithmetic as arithmetic_200, <N: 10, M: 20>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let a_array = [20i32; 100];
    let mut a_gpu = GpuBuffer::from_host(&Box::new(a_array), &stream).unwrap();
    let b_array = [20i32; 100];
    let b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream).unwrap();

    // Bad Size! The instantiation gives N * M = 200 elements, our buffers 100.
    let _ = arithmetic_200(&stream, &mut a_gpu, &b_gpu);
}
