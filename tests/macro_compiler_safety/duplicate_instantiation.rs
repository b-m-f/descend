use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_twice<N: nat, r: prv>(
        a_array: &r uniq gpu.global [i32; N],
        b_array: &r shrd gpu.global [i32; N]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
            }
        }
    }
}

// Two instantiations without an alias both claim the kernel's own name. The
// macro cannot see the other invocation, so rustc reports the collision.
descend_fn!(gpu_twice, <N: 256>);
descend_fn!(gpu_twice, <N: 512>);

fn main() {}
