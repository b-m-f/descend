use descend_macros::descend;

// A `descend!` block whose items contain no launchable function declares nothing:
// every arm of the expansion is skipped and the macro emits no items at all. The
// block reads like it did something, so the silence is the bug.
descend! {
    fn cpu_only(a: i32) -[t: cpu.thread]-> i32 {
        a + 1
    }
}

fn main() {}
