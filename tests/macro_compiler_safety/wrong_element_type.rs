use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

fn main() {
    descend! {
        fn gpu_elem_type<r: prv>(
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

    let gpu_elem_type = descend_fn!(gpu_elem_type);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut a_gpu = GpuBuffer::from_host(&[1i32; 100], &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&[1.0f32; 100], &stream).unwrap();

    // The second kernel parameter holds i32, not f32.
    let _ = gpu_elem_type(&stream, &mut a_gpu, &b_gpu);
}
