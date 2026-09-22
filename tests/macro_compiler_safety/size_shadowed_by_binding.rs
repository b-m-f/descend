use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_shadowed<N: nat, r: prv>(
        a_array: &r uniq gpu.global [i32; N]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                // `N` is a size of this kernel, so binding it here would be
                // rewritten to the instantiated value.
                let N = 3;
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + N
            }
        }
    }
}

descend_fn!(gpu_shadowed, <N: 256>);

fn main() {}
