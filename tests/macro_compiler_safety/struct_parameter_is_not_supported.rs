use descend_macros::{descend, descend_fn};

fn main() {
    descend! {
        struct Point { x: i32, y: i32 }

        fn gpu_struct_param<r: prv>(
            points: &r uniq gpu.global [Point; 64]
        ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
            sched block in grid {
                sched thread in block {
                    ()
                }
            }
        }
    }

    let _ = descend_fn!(gpu_struct_param);
}
