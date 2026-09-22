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
pub fn compile_error_item(message: &str) -> proc_macro2::TokenStream {
    use std::str::FromStr;
    let escaped = format!("compile_error!({message:?});");
    proc_macro2::TokenStream::from_str(&escaped)
        .expect("compile_error token generation must succeed")
}

/// The same, underlining a particular token.
///
/// A failure about one kernel or one parameter should point at that kernel or
/// parameter, not at the whole `descend!` block the user wrote.
pub fn error_at(span: proc_macro2::Span, message: &str) -> proc_macro2::TokenStream {
    quote::quote_spanned! {span=> compile_error!(#message); }
}
