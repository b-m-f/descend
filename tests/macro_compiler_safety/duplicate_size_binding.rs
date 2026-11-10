use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_dup_size<N: nat, r: prv>(
        a_array: &r uniq gpu.global [i32; N]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + 1
            }
        }
    }
}

fn main() {
    // `N` is bound twice. The bindings become a map, so one value silently wins and
    // the instantiation is not the one that was written.
    let _ = descend_fn!(gpu_dup_size, <N: 256, N: 512>);
}
