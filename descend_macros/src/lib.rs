extern crate proc_macro;

#[macro_use]
mod diag;

mod capture;
mod compiler;
mod const_expr;
mod emit;
mod kernel;
mod macros;
mod main_fn;
mod nat;
mod nvcc;
mod parsing;
mod pipeline;
mod shared_mem;
mod specialize;
mod types;

use proc_macro2::TokenStream;

fn expand_or_error(
    input: proc_macro::TokenStream,
    expand: fn(TokenStream) -> Result<TokenStream, crate::diag::MacroError>,
) -> proc_macro::TokenStream {
    match expand(input.into()) {
        Ok(tokens) => tokens,
        Err(error) => error.to_tokens(),
    }
    .into()
}

#[proc_macro]
pub fn descend(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, macros::expand_descend)
}

#[proc_macro]
pub fn descend_fn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, macros::expand_descend_fn)
}

#[proc_macro]
pub fn _descend_fn_internal(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, macros::expand_fn_internal)
}
