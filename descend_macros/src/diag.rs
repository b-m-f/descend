macro_rules! debug {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG").is_ok() {
            eprintln!($($arg)*);
        }
    }
}

pub fn compile_error_item(message: &str) -> proc_macro2::TokenStream {
    use std::str::FromStr;
    let escaped = format!("compile_error!({message:?});");
    proc_macro2::TokenStream::from_str(&escaped)
        .expect("compile_error token generation must succeed")
}

pub fn error_at(span: proc_macro2::Span, message: &str) -> proc_macro2::TokenStream {
    quote::quote_spanned! {span=> compile_error!(#message); }
}
