use descend_host::{DescendContext, GpuBuffer};
use descend_macros::{descend, descend_launch};

fn main() {
    descend! {
        fn gpu_bad_mut<r: prv>(
            a_array: &r uniq gpu.global [i32; 100],
            b_array: &r shrd gpu.global [i32; 100]
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
    let a_array = [20i32; 100];
    let a_gpu = GpuBuffer::from_slice(&ctx, &a_array); // Not mut!

    let b_array = [20i32; 100];
    let mut b_gpu = GpuBuffer::from_slice(&ctx, &b_array); // Mut!

    // Bad Mutability! Expects `uniq` (mut) for the first argument, but we give it an immutable `&a_gpu`.
    // And expects `shrd` for the second, but we give a mutable `&mut b_gpu`.
    descend_launch!(gpu_bad_mut, , &a_gpu, &mut b_gpu);
}
