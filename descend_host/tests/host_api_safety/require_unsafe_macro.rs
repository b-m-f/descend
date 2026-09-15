use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_launch};

fn main() {
    descend! {
        fn gpu_vec_add<r: prv>(
            a_array: &r uniq gpu.global [i32; 10],
            b_array: &r shrd gpu.global [i32; 10]
        ) -[grid: gpu.grid<X<1>, X<1>>]-> () {
            sched block in grid {
                sched thread in block {
                    let a_ref = &uniq (*a_array).to_view.grp::<1>[[block]][[thread]];
                    *a_ref = *a_ref + (*b_array).to_view.grp::<1>[[block]][[thread]]
                }
            }
        }
    };

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    
    let mut a_gpu = GpuBuffer::from_host(&[1i32; 10], &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&[1i32; 10], &stream).unwrap();

    // Should fail without an unsafe block
    descend_launch!(gpu_vec_add, &stream, , &mut a_gpu, &b_gpu).unwrap();
}
