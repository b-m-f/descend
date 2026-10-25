use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

fn main() {
    descend! {
        fn gpu_small_grid<r: prv>(
            a_array: &r uniq gpu.global [i32; 1024],
            b_array: &r shrd gpu.global [i32; 1024]
        ) -[grid: gpu.grid<X<4>, X<256>>]-> () {
            sched block in grid {
                sched thread in block {
                    let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                    *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
                }
            }
        }
    }

    let gpu_small_grid = descend_fn!(gpu_small_grid<<<8, 256>>>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut a_gpu = GpuBuffer::from_host(&Box::new([20i32; 1024]), &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&Box::new([10i32; 1024]), &stream).unwrap();

    gpu_small_grid(&stream, &mut a_gpu, &b_gpu).unwrap();
}
