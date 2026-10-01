use descend_macros::descend;

fn main() {
    descend! {
        struct Particle {
            pos: (f32, f32),
            id: i32
        }

        fn gpu_touch<r: prv>(
            values: &r uniq gpu.global [Particle; 64]
        ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
            sched block in grid {
                sched thread in block {
                    let p = &uniq (*values).to_view.grp::<64>[[block]][[thread]];
                    (*p).id = (*p).id + 1
                }
            }
        }
    }
}
