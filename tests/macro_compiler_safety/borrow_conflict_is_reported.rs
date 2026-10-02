use descend_macros::{descend, descend_fn};

// Two overlapping `uniq` borrows of the same array. Most `BorrowingError` variants reach
// `TyError::emit` as a bare `{:?}` with no `error: ` prefix, so before the capture fell
// back to unprefixed output this was reported only as "the Descend compiler reported the
// reason above" -- losing the one diagnostic Descend exists to produce. The message is
// cut short because the unprefixed form is an AST dump; the full text is printed above
// it by the macro.
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
