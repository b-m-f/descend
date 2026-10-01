use descend_macros::{descend, descend_fn};

fn main() {
    descend! {
        fn gpu_tuple_param<r: prv>(
            values: &r uniq gpu.global [(i32, f32); 64]
        ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
            sched block in grid {
                sched thread in block {
                    let p = &uniq (*values).to_view.grp::<64>[[block]][[thread]];
                    ()
                }
            }
        }
    }

    // A tuple has no guaranteed layout on either side, so the macro cannot give this
    // parameter a host type. Without the check the generated Rust would take
    // `GpuBuffer<[i32; 64]>` while the kernel takes
    // `descend::tuple<descend::i32, descend::f32>*`.
    let _ = descend_fn!(gpu_tuple_param);
}
