use crate::parsing::{
    kernel_name_span, parse_descend_source, parse_fn_internal_args, parse_fn_invocation,
    parse_generic_bindings, to_descend_string, FnInternalArgs, FnInvocation,
};
use crate::pipeline::{compile_kernel_fn, instantiate};
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{format_ident, quote};
use std::str::FromStr;

pub fn expand_descend(input: TokenStream) -> Result<TokenStream, String> {
    let source_ts = input.clone();

    let source_str = to_descend_string(source_ts.clone());
    debug!("MACRO INPUT: {}", source_str);
    let source = descend::parser::SourceCode::new(source_str);
    let compilation_unit = parse_descend_source(&source)
        .map_err(|err| format!("failed to parse the Descend source: {err}"))?;

    let mut output = TokenStream::new();

    for item in &compilation_unit.items {
        match item {
            descend::ast::Item::FunDef(f) if crate::kernel::is_launchable(f) => {
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
                let struct_code = crate::emit::format_rust_struct(s);
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

pub fn expand_descend_fn(input: TokenStream) -> Result<TokenStream, String> {
    let FnInvocation {
        kernel,
        item_name,
        bindings,
    } = parse_fn_invocation(input)?;

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

pub fn expand_fn_internal(input: TokenStream) -> Result<TokenStream, String> {
    let FnInternalArgs {
        source_ts,
        kernel_name,
        item_name,
        item_span,
        bindings,
    } = parse_fn_internal_args(input)?;

    let bindings = match parse_generic_bindings(bindings) {
        Ok(bindings) => bindings,
        Err(message) => return Ok(crate::diag::error_at(item_span, &message)),
    };

    let source_str = to_descend_string(source_ts.clone());
    let source = descend::parser::SourceCode::new(source_str);
    let compilation_unit = parse_descend_source(&source)
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
