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

/// `compile_error!` without the trailing semicolon, for a macro expanded in expression
/// position. Its type is `!`, so it poisons the expression it replaces and nothing
/// downstream reports a second, derived error.
pub fn compile_error_expr(message: &str) -> proc_macro2::TokenStream {
    use std::str::FromStr;
    let escaped = format!("compile_error!({message:?})");
    proc_macro2::TokenStream::from_str(&escaped)
        .expect("compile_error token generation must succeed")
}

pub fn error_at(span: proc_macro2::Span, message: &str) -> proc_macro2::TokenStream {
    quote::quote_spanned! {span=> compile_error!(#message); }
}

pub fn error_at_expr(span: proc_macro2::Span, message: &str) -> proc_macro2::TokenStream {
    quote::quote_spanned! {span=> compile_error!(#message) }
}

#[derive(Debug)]
pub struct MacroError {
    pub message: String,
    pub span: Option<proc_macro2::Span>,
}

impl MacroError {
    pub fn at(message: String, span: Option<proc_macro2::Span>) -> Self {
        Self { message, span }
    }

    pub fn or_span(mut self, fallback: Option<proc_macro2::Span>) -> Self {
        self.span = self.span.or(fallback);
        self
    }

    pub fn to_tokens(&self) -> proc_macro2::TokenStream {
        match self.span {
            Some(span) => error_at(span, &self.message),
            None => compile_error_item(&self.message),
        }
    }

    /// For `descend_fn!` and the carrier, which expand in expression position.
    pub fn to_expr_tokens(&self) -> proc_macro2::TokenStream {
        match self.span {
            Some(span) => error_at_expr(span, &self.message),
            None => compile_error_expr(&self.message),
        }
    }
}

impl From<String> for MacroError {
    fn from(message: String) -> Self {
        Self {
            message,
            span: None,
        }
    }
}
