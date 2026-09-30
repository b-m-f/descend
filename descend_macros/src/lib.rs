extern crate proc_macro;

#[macro_use]
mod diag;

mod ast;
mod compile;
mod descend_src;
mod emit;
mod ir;
mod nat;
mod shared_mem;
mod specialize;

use compile::compile_to_ptx;
use ir::Kernel;
use proc_macro2::{Delimiter, Group, Span, TokenStream, TokenTree};
use quote::{format_ident, quote};
use specialize::synthesize_main_program;

fn expand_or_error(
    input: proc_macro::TokenStream,
    expand: fn(TokenStream) -> Result<TokenStream, String>,
) -> proc_macro::TokenStream {
    match expand(input.into()) {
        Ok(tokens) => tokens,
        Err(message) => crate::diag::compile_error_item(&message),
    }
    .into()
}
use std::str::FromStr;

#[proc_macro]
pub fn descend(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, expand_descend)
}

fn expand_descend(input: TokenStream) -> Result<TokenStream, String> {
    let source_ts = input.clone();

    let source_str = crate::descend_src::to_descend_string(source_ts.clone());
    debug!("MACRO INPUT: {}", source_str);
    let source = descend::parser::SourceCode::new(source_str);
    let compilation_unit = crate::ast::parse_descend_source(&source)
        .map_err(|err| format!("failed to parse the Descend source: {err}"))?;

    let mut output = TokenStream::new();

    for item in &compilation_unit.items {
        match item {
            descend::ast::Item::FunDef(f) if crate::ir::is_launchable(f) => {
                let kernel_name = f.ident.name.to_string();
                let has_nat_generics = f
                    .generic_params
                    .iter()
                    .any(|param| matches!(param.kind, descend::ast::Kind::Nat));

                output.extend(carrier_macro_for(&kernel_name, &source_ts));

                if !has_nat_generics {
                    match compile_kernel_fn(&kernel_name, &source_ts, &compilation_unit) {
                        Ok(tokens) => output.extend(tokens),
                        Err(message) => {
                            return Ok(match kernel_name_span(&source_ts, &kernel_name) {
                                Some(span) => crate::diag::error_at(span, &message),
                                None => crate::diag::compile_error_item(&message),
                            })
                        }
                    }
                }
            }
            descend::ast::Item::StructDecl(s) => {
                let struct_code = crate::ast::format_rust_struct(s);
                output.extend(
                    struct_code
                        .parse::<TokenStream>()
                        .map_err(|err| format!("generated struct is malformed: {err}"))?,
                );
            }
            _ => {}
        }
    }

    Ok(output)
}

fn kernel_name_span(source_ts: &TokenStream, name: &str) -> Option<Span> {
    let mut previous_was_fn = false;
    for tt in source_ts.clone() {
        match &tt {
            TokenTree::Ident(ident) if previous_was_fn && ident == name => {
                return Some(ident.span());
            }
            TokenTree::Ident(ident) => previous_was_fn = ident == "fn",
            _ => previous_was_fn = false,
        }
    }
    None
}

fn compile_kernel_fn(
    kernel_name: &str,
    source_ts: &TokenStream,
    compilation_unit: &descend::ast::CompilUnit,
) -> Result<TokenStream, String> {
    let kernel = Kernel::extract(compilation_unit, kernel_name)?;
    crate::shared_mem::check_shared_params(&kernel)?;

    let full_descend = synthesize_main_program(
        source_ts.clone(),
        &kernel,
        &std::collections::HashMap::new(),
    )?;
    let compiled = compile_to_ptx(&full_descend, &kernel.function_name)?;

    crate::emit::generate_launch_fn(
        &kernel,
        &compiled,
        &std::collections::HashMap::new(),
        kernel_name,
    )
}

fn carrier_macro_for(kernel_name: &str, source_ts: &TokenStream) -> TokenStream {
    let kernel = format_ident!("{}", kernel_name);
    let refusal = format!(
        "kernel `{kernel_name}` is launched through a generated function, not through a macro. \
         Write `descend_fn!({kernel_name}, <...sizes>);` once and call \
         `{kernel_name}(&stream, ..)`; use `descend_fn!({kernel_name} as <name>, <...sizes>);` \
         for a second set of sizes."
    );

    quote! {
        macro_rules! #kernel {
            (@fn $name:ident, $($args:tt)*) => {
                ::descend_macros::_descend_fn_internal! {
                    { #source_ts }, #kernel, $name, $($args)*
                }
            };
            ($($args:tt)*) => {
                compile_error!(#refusal)
            };
        }
    }
}

#[proc_macro]
pub fn descend_fn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, expand_descend_fn)
}

fn expand_descend_fn(input: TokenStream) -> Result<TokenStream, String> {
    let mut iter = input.into_iter();

    let Some(TokenTree::Ident(kernel)) = iter.next() else {
        return Err(
            "expected a kernel name, as in `descend_fn!(gpu_vec_add, <N: 16384>)`".to_string(),
        );
    };

    let item_name = match iter.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == ',' => kernel.clone(),
        Some(TokenTree::Ident(keyword)) if keyword == "as" => {
            let Some(TokenTree::Ident(alias)) = iter.next() else {
                return Err(format!(
                    "expected a name for the instantiation of `{kernel}`"
                ));
            };
            match iter.next() {
                Some(TokenTree::Punct(p)) if p.as_char() == ',' => {}
                _ => return Err(format!("expected `,` after the name `{alias}`")),
            }
            alias
        }
        _ => {
            return Err(format!(
                "expected `,` or `as <name>` after the kernel `{kernel}`"
            ))
        }
    };

    let bindings: TokenStream = iter.collect();

    let mut inner = TokenStream::from_str(&format!("@fn {item_name},"))
        .map_err(|error| format!("malformed instantiation: {error}"))?;
    inner.extend(bindings);

    let mut output = format!("{kernel}!").parse::<TokenStream>().unwrap();
    output.extend(std::iter::once(TokenTree::Group(Group::new(
        Delimiter::Brace,
        inner,
    ))));
    Ok(output)
}

#[proc_macro]
pub fn _descend_fn_internal(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, fn_internal)
}

fn fn_internal(input: TokenStream) -> Result<TokenStream, String> {
    let mut iter = input.into_iter();

    let Some(TokenTree::Group(source_group)) = iter.next() else {
        return Err("expected the Descend source block as the first argument".to_string());
    };
    let source_ts = source_group.stream();

    let _comma = iter.next();
    let kernel_name = iter
        .next()
        .ok_or_else(|| "expected the kernel name".to_string())?
        .to_string();
    let _comma = iter.next();
    let item_name_tt = iter
        .next()
        .ok_or_else(|| "expected the name of the instantiation".to_string())?;
    let item_span = item_name_tt.span();
    let item_name = item_name_tt.to_string();
    let _comma = iter.next();

    let bindings = match crate::ast::parse_generic_bindings(iter.collect()) {
        Ok(bindings) => bindings,
        Err(message) => return Ok(crate::diag::error_at(item_span, &message)),
    };

    let source_str = crate::descend_src::to_descend_string(source_ts.clone());
    let source = descend::parser::SourceCode::new(source_str);
    let compilation_unit = crate::ast::parse_descend_source(&source)
        .map_err(|err| format!("failed to parse the Descend source: {err}"))?;

    match instantiate(
        &compilation_unit,
        source_ts,
        &kernel_name,
        &item_name,
        &bindings,
    ) {
        Ok(tokens) => Ok(tokens),
        Err(message) => Ok(crate::diag::error_at(item_span, &message)),
    }
}

fn instantiate(
    compilation_unit: &descend::ast::CompilUnit,
    source_ts: TokenStream,
    kernel_name: &str,
    item_name: &str,
    bindings: &std::collections::HashMap<String, String>,
) -> Result<TokenStream, String> {
    let kernel = Kernel::extract(compilation_unit, kernel_name)?;
    crate::shared_mem::check_shared_params(&kernel)?;

    for generic in &kernel.nat_generics {
        if !bindings.contains_key(generic) {
            return Err(format!(
                "instantiating kernel `{kernel_name}` needs a value for its size `{generic}`"
            ));
        }
    }
    for bound in bindings.keys() {
        if !kernel.nat_generics.contains(bound) {
            return Err(format!(
                "kernel `{kernel_name}` has no size `{bound}` to instantiate"
            ));
        }
    }

    let full_descend = synthesize_main_program(source_ts, &kernel, bindings)?;
    let compiled = compile_to_ptx(&full_descend, &kernel.function_name)?;

    crate::emit::generate_launch_fn(&kernel, &compiled, bindings, item_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_carrier_macro_instantiates_but_refuses_to_launch() {
        let source = TokenStream::from_str("fn gpu_k() {}").unwrap();
        let carrier = carrier_macro_for("gpu_k", &source).to_string();

        let compact: String = carrier.split_whitespace().collect();
        assert!(
            compact.starts_with("macro_rules!gpu_k{"),
            "the carrier must be named after the kernel: {carrier}"
        );
        assert!(
            carrier.contains("_descend_fn_internal"),
            "the instantiation arm is missing: {carrier}"
        );
        assert!(
            carrier.contains("compile_error"),
            "a direct launch must be refused: {carrier}"
        );
        assert!(
            carrier.contains("descend_fn"),
            "the refusal should name the instantiation to write instead: {carrier}"
        );
    }

    #[test]
    fn the_carrier_macro_embeds_the_kernel_source() {
        let source = TokenStream::from_str("fn gpu_k(a: &r uniq gpu.global [i32; 4]) {}").unwrap();
        let carrier = carrier_macro_for("gpu_k", &source).to_string();
        assert!(
            carrier.contains("gpu.global") || carrier.contains("gpu . global"),
            "the source must travel with the macro: {carrier}"
        );
    }
}
