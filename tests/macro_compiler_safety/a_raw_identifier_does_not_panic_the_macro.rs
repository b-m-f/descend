use descend_macros::descend_fn;

fn main() {
    // A raw identifier is a legal way to spell a name, so it reaches the macro. The
    // carrier's name is built by interpolating it into `__descend_kernel_{}`, which is
    // not a valid identifier once the `r#` is carried along. The name is refused before
    // it gets there, so this is a diagnostic rather than a panic.
    let _ = descend_fn!(r#match, <N: 256>);
}
