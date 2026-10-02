macro_rules! debug {
    ($($argument:tt)*) => {
        if $crate::diag::debugging_enabled() {
            eprintln!($($argument)*);
        }
    }
}

/// Whether `DESCEND_DEBUG` is set to anything.
///
/// Read once rather than per call. `debug!` is invoked once per rendered source and once
/// per generated CUDA module, and one of its call sites is inside `specialize`'s
/// per-identifier substitution loop, so looking the variable up each time would take the
/// environment lock and allocate a `String` for every token a kernel has.
pub fn debugging_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("DESCEND_DEBUG").is_some_and(|value| !value.is_empty())
    });
    *ENABLED
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
        let message = &self.message;
        match self.span {
            Some(span) => quote::quote_spanned! {span=> compile_error!{ #message } },
            None => quote::quote! { compile_error!{ #message } },
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

impl From<&str> for MacroError {
    fn from(message: &str) -> Self {
        Self::from(message.to_string())
    }
}

/// The message alone. A span is for pointing the compiler at a place in the user's
/// source, and has no meaning once the message is being read as text.
impl std::fmt::Display for MacroError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}
