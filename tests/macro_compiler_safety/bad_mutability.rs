use descend_host::{Context, GpuBuffer};
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

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let a_array = [20i32; 100];
    let a_gpu = GpuBuffer::from_host(&Box::new(a_array), &stream).unwrap(); // Not mut!

    let b_array = [20i32; 100];
    let mut b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream).unwrap(); // Mut!

    // Bad Mutability! Expects `uniq` (mut) for the first argument, but we give it an immutable `&a_gpu`.
    // Passing `&mut b_gpu` to the `shrd` parameter is accepted: it coerces to a shared reference.
    let _ = unsafe { descend_launch!(gpu_bad_mut, &stream, , &a_gpu, &mut b_gpu) };
}
