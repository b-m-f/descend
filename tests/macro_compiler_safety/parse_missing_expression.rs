use descend_macros::descend;

descend! {
    fn bad() -[grid: gpu.grid<X<1>, X<1>>]-> () { let x = ; }
}

fn main() {}
