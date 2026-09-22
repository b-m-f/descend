/// Internal utility for formatting stdout log statements.
///
/// What it does: Suppresses internal debugging blocks unless explicitly configured.
/// Why it is needed: Limits standard console noise escaping during compilation unless explicitly troubleshooting.
macro_rules! debug {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG").is_ok() {
            eprintln!($($arg)*);
        }
    }
}

/// Emits `compile_error!("...")` in place of the items a macro would generate.
///
/// Every failure of the pipeline is reported this way rather than by panicking:
/// a panic inside a procedural macro surfaces as "proc macro panicked", which
/// buries the message and points at no particular part of the user's code.
pub fn compile_error_item(message: &str) -> proc_macro::TokenStream {
    use std::str::FromStr;
    let escaped = format!("compile_error!({message:?});");
    proc_macro::TokenStream::from_str(&escaped)
        .expect("compile_error token generation must succeed")
}
