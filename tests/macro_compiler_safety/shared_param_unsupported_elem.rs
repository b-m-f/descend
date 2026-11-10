use descend_macros::{descend, descend_fn};

fn main() {
    descend! {
        fn gpu_shared_bytes<r1: prv, r2: prv>(
            vec: &r1 uniq gpu.global [u8; 64],
            [grid.forall(X)] tmp: &r2 uniq gpu.shared [u8; 64]
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

    // The shared-memory parameters are checked where the kernel is
    // instantiated, so the instantiation is what reports the error.
    let _ = descend_fn!(gpu_shared_bytes);
}
