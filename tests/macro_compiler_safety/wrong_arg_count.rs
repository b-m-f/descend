use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_launch};

fn main() {
    descend! {
        fn gpu_arg_count<r: prv>(
            a_array: &r uniq gpu.global [i32; 100],
            b_array: &r shrd gpu.global [i32; 100]
        ) -[grid: gpu.grid<X<1>, X<100>>]-> () {
            sched block in grid {
                sched thread in block {
                    let a_ref = &uniq (*a_array).to_view.grp::<100>[[block]][[thread]];
                    *a_ref = *a_ref + (*b_array).to_view.grp::<100>[[block]][[thread]]
                }
            }
        }
    }

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut a_gpu = GpuBuffer::from_host(&[1i32; 100], &stream).unwrap();

    // The kernel has two parameters but only one buffer is passed.
    let _ = unsafe { descend_launch!(gpu_arg_count, &stream, , &mut a_gpu) };
}
