use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

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

    let gpu_shared_tmp = descend_fn!(gpu_shared_tmp);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut vec_gpu = GpuBuffer::from_host(&[1i32; 64], &stream).unwrap();
    let mut tmp_gpu = GpuBuffer::from_host(&[0i32; 64], &stream).unwrap();

    // A gpu.shared parameter is allocated on the device, not passed from the
    // host, so the kernel takes one argument here and not two.
    let _ = gpu_shared_tmp(&stream, &mut vec_gpu, &mut tmp_gpu);
}
