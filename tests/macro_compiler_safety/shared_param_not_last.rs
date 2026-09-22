use descend_macros::descend;

fn main() {
    descend! {
        fn gpu_shared_first<r1: prv, r2: prv>(
            [grid.forall(X)] tmp: &r2 uniq gpu.shared [i32; 64],
            vec: &r1 uniq gpu.global [i32; 64]
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

}
