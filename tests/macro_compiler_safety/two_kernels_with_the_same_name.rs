use descend_macros::descend;

// Each launchable kernel emits a carrier named after it, so two kernels of the same
// name emit the same macro twice. The error should name the redeclared kernel.
descend! {
    fn gpu_twice<N: nat, r: prv>(
        a_array: &r uniq gpu.global [i32; N]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + 1
            }
        }
    }

    fn gpu_twice<N: nat, r: prv>(
        a_array: &r uniq gpu.global [i32; N]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + 2
            }
        }
    }
}

fn main() {}
