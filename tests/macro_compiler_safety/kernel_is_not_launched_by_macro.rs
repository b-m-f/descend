use descend_host::{Context, GpuBuffer};
use descend_macros::descend;

fn main() {
    descend! {
        fn gpu_generic_add<N: nat, r: prv>(
            a_array: &r uniq gpu.global [i32; N],
            b_array: &r shrd gpu.global [i32; N]
        ) -[grid: gpu.grid<X<4>, X<256>>]-> () {
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
    let mut a_gpu = GpuBuffer::from_host(&Box::new([20i32; 1024]), &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&Box::new([10i32; 1024]), &stream).unwrap();

    // A kernel is launched through a generated function. Nothing named after the
    // kernel exists in the macro namespace -- the carrier that holds its source is
    // named so that neither this nor a bare `gpu_generic_add(..)` resolves to the
    // macro's own plumbing.
    let _ = gpu_generic_add!(&stream, <N: 1024>, &mut a_gpu, &b_gpu);
}
