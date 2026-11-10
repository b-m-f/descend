use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_declared<N: nat, r: prv>(
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
    // No kernel of this name was declared, so no carrier macro exists and no arm can
    // fire: a custom message is unreachable here, and none ever existed. Naming the
    // carrier after the kernel would only change which name rustc quotes --
    // `gpu_never_declared` rather than `__descend_kernel_gpu_never_declared` -- and it
    // would cost the clean `cannot find function` error that
    // `kernel_called_without_an_instantiation` pins. Commit 2f7faf4 made that trade
    // deliberately; this file pins the message so it stays visible.
    let _ = descend_fn!(gpu_never_declared, <N: 256>);
}
