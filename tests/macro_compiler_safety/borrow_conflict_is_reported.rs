use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_conflict<r: prv>(
        a_array: &r uniq gpu.global [i32; 256]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            let first = &uniq (*a_array).to_view;
            let second = &uniq (*a_array).to_view;
            sched thread in block {
                let a = &uniq (*first)[[thread]];
                let b = &shrd (*second)[[thread]];
                *a = *b
            }
        }
    }
}

fn main() {
    let _ = descend_fn!(gpu_conflict);
}
