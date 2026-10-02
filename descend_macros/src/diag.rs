macro_rules! debug {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG").is_ok() {
            eprintln!($($arg)*);
        }
    }
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
