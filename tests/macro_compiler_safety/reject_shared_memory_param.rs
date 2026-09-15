use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_launch};

fn main() {
    descend! {
        fn gpu_shared_tmp<r1: prv, r2: prv>(
            vec: &r1 uniq gpu.global [i32; 64],
            [grid.forall(X)] tmp: &r2 uniq gpu.shared [i32; 64]
        ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
            let view = &uniq (*vec).to_view.grp::<64>;
            sched block in grid {
                let ib = &uniq (*view)[[block]];
                let tmp_view = &uniq (*tmp).to_view;
                sched thread in block {
                    let v = &uniq (*ib)[[thread]];
                    let t = &uniq (*tmp_view)[[thread]];
                    *t = *v;
                    *v = *t
                }
            }
        }
    }

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut vec_gpu = GpuBuffer::from_host(&[1i32; 64], &stream).unwrap();
    let mut tmp_gpu = GpuBuffer::from_host(&[0i32; 64], &stream).unwrap();

    // gpu.shared parameters have no host-side layout contract yet.
    let _ = unsafe { descend_launch!(gpu_shared_tmp, &stream, , &mut vec_gpu, &mut tmp_gpu) };
}
