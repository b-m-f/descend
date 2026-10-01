use crate::diag::MacroError;
use crate::parsing::{
    flatten_source, parse_descend_source, parse_fn_internal_args, parse_fn_invocation,
    parse_generic_bindings, FnInternalArgs, FnInvocation,
};
use crate::pipeline::{compile_kernel_fn, instantiate};
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{format_ident, quote};
use std::str::FromStr;

pub fn expand_descend(input: TokenStream) -> Result<TokenStream, MacroError> {
    let source_ts = input.clone();

    let source = flatten_source(source_ts.clone());
    debug!("MACRO INPUT: {}", source.text);
    let source_code = descend::parser::SourceCode::new(source.text.clone());
    let compilation_unit = parse_descend_source(&source_code)?;

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
                    match compile_kernel_fn(&kernel_name, &source, &compilation_unit) {
                        Ok(tokens) => output.extend(tokens),
                        Err(error) => {
                            let name_span = f.ident.span.and_then(|s| source.span_at(s.begin));
                            return Ok(error.or_span(name_span).to_tokens());
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
         Write `let {kernel_name} = descend_fn!({kernel_name}, <...sizes>);` and call \
         `{kernel_name}(&stream, ..)`; a second set of sizes is a second binding."
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

/// `descend_fn!` is an expression: it expands to a block that defines the
/// instantiated function and then evaluates to it, so the caller names it with an
/// ordinary `let`. A procedural macro cannot tell item position from expression
/// position, so this is the only position the macro supports; an instantiation that
/// has to be reachable elsewhere goes in a hand-written wrapper function.
pub fn expand_descend_fn(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FnInvocation { kernel, bindings } = parse_fn_invocation(input)?;

    // The generated item keeps the kernel's name. It lives in the block below, so it
    // cannot collide with the binding the caller writes, nor with another
    // instantiation of the same kernel in the same scope.
    let mut carrier_args = TokenStream::from_str(&format!("@fn {kernel},"))
        .map_err(|error| format!("malformed instantiation: {error}"))?;
    carrier_args.extend(bindings);

    let mut block = format!("{kernel}!").parse::<TokenStream>().unwrap();
    block.extend(std::iter::once(TokenTree::Group(Group::new(
        Delimiter::Brace,
        carrier_args,
    ))));
    block.extend(format!("{kernel}").parse::<TokenStream>().unwrap());

    Ok(TokenStream::from(TokenTree::Group(Group::new(
        Delimiter::Brace,
        block,
    ))))
}

pub fn expand_fn_internal(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FnInternalArgs {
        source_ts,
        kernel_name,
        item_name,
        item_span,
        bindings,
    } = parse_fn_internal_args(input)?;

    let bindings = match parse_generic_bindings(bindings) {
        Ok(bindings) => bindings,
        Err(error) => return Ok(error.or_span(Some(item_span)).to_tokens()),
    };

    match instantiate(source_ts, &kernel_name, &item_name, &bindings) {
        Ok(tokens) => Ok(tokens),
        Err(error) => Ok(error.or_span(Some(item_span)).to_tokens()),
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
