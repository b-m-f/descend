use crate::diag::MacroError;
use crate::parsing::{
    flatten_source, parse_descend_source, parse_fn_internal_args, parse_fn_invocation,
    parse_generic_bindings, FnInternalArgs, FnInvocation,
};
use crate::pipeline::instantiate;
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
            // `descend!` declares a kernel; it does not compile one. Whatever its sizes
            // are, the kernel becomes a function only where a `descend_fn!` instantiates
            // it, so the carrier that carries this source to an instantiation is all
            // that is emitted here.
            descend::ast::Item::FunDef(f) if crate::kernel::is_launchable(f) => {
                output.extend(carrier_macro_for(&f.ident.name.to_string(), &source_ts));
            }
            // A declared type becomes its Rust counterpart here. Unlike a kernel this
            // costs nothing to emit -- no nvcc, just the type -- and a kernel that uses
            // it needs the Rust type in scope to be callable at all.
            descend::ast::Item::StructDecl(s) => {
                let struct_code = crate::emit::format_rust_struct(s)?;
                output.extend(
                    struct_code
                        .parse::<TokenStream>()
                        .map_err(|err| format!("generated type is malformed: {err}"))?,
                );
            }
            _ => {}
        }
    }

    Ok(output)
}

/// The name of the macro that carries a kernel's source from its `descend!` to an
/// instantiation.
///
/// Deliberately *not* the kernel's own name. A macro named after the kernel would make
/// `gpu_vec_add(&stream, ..)` — the call someone writes when they forget the
/// instantiation — resolve in the macro namespace and report `expected function, found
/// macro`, which describes the macro's own plumbing rather than the mistake. Under a
/// name nobody writes, the kernel's name is simply not in scope, and the call fails as
/// `cannot find function` like any other undefined call.
fn carrier_name(kernel_name: &str) -> proc_macro2::Ident {
    format_ident!("__descend_kernel_{}", kernel_name)
}

fn carrier_macro_for(kernel_name: &str, source_ts: &TokenStream) -> TokenStream {
    let carrier = carrier_name(kernel_name);
    let kernel = format_ident!("{}", kernel_name);

    quote! {
        // A kernel may be declared and never instantiated; that is not a mistake, and
        // the warning would name this macro.
        #[allow(unused_macros)]
        macro_rules! #carrier {
            (@fn $name:ident, $($args:tt)*) => {
                ::descend_macros::__descend_fn_internal! {
                    { #source_ts }, #kernel, $name, $($args)*
                }
            };
        }
    }
}

/// `descend_fn!` is an expression: it evaluates to the instantiated function, so the
/// caller names it with an ordinary `let`. A procedural macro cannot tell item position
/// from expression position, so this is the only position the macro supports; an
/// instantiation that has to be reachable elsewhere goes in a hand-written wrapper
/// function.
///
/// This hands the sizes to the kernel's carrier, which routes them back through
/// `expand_fn_internal` — the only place that knows whether the instantiation
/// succeeded, and therefore the only place that can decide the shape of the
/// expansion.
pub fn expand_descend_fn(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FnInvocation { kernel, bindings } = parse_fn_invocation(input)?;

    let mut carrier_args = TokenStream::from_str(&format!("@fn {kernel},"))
        .map_err(|error| format!("malformed instantiation: {error}"))?;
    carrier_args.extend(bindings);

    let mut invocation = format!("{}!", carrier_name(&kernel.to_string()))
        .parse::<TokenStream>()
        .unwrap();
    invocation.extend(std::iter::once(TokenTree::Group(Group::new(
        Delimiter::Brace,
        carrier_args,
    ))));
    Ok(invocation)
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
        Err(error) => return Ok(error.or_span(Some(item_span)).to_expr_tokens()),
    };

    match instantiate(source_ts, &kernel_name, &item_name, &bindings) {
        // The generated item keeps the kernel's name. It lives in the block below, so
        // it cannot collide with the binding the caller writes, nor with another
        // instantiation of the same kernel in the same scope.
        Ok(items) => Ok(block_yielding(items, &item_name)),
        // A bare `compile_error!` rather than a block: a block would have to end in the
        // name of a function that was never generated, and that dangling name is a
        // second, derived error on top of the real one.
        Err(error) => Ok(error.or_span(Some(item_span)).to_expr_tokens()),
    }
}

/// `{ <items> <name> }` — the items the instantiation generated, followed by the
/// function's name, so the block evaluates to that function.
fn block_yielding(items: TokenStream, name: &str) -> TokenStream {
    let mut block = items;
    block.extend(name.parse::<TokenStream>().expect("item name must be an ident"));
    TokenStream::from(TokenTree::Group(Group::new(Delimiter::Brace, block)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_carrier_macro_is_not_named_after_the_kernel() {
        let source = TokenStream::from_str("fn gpu_k() {}").unwrap();
        let carrier = carrier_macro_for("gpu_k", &source).to_string();

        let compact: String = carrier.split_whitespace().collect();
        assert!(
            !compact.contains("macro_rules!gpu_k{"),
            "a macro named after the kernel makes a forgotten instantiation report \
             `expected function, found macro`: {carrier}"
        );
        assert!(
            compact.contains("macro_rules!__descend_kernel_gpu_k{"),
            "the carrier should be under a name nobody writes: {carrier}"
        );
        assert!(
            carrier.contains("__descend_fn_internal"),
            "the instantiation arm is missing: {carrier}"
        );
    }

    #[test]
    fn an_instantiation_invokes_the_carrier_and_not_the_kernel_name() {
        let expansion = expand_descend_fn(TokenStream::from_str("gpu_k, <N: 8>").unwrap())
            .unwrap()
            .to_string();
        let compact: String = expansion.split_whitespace().collect();
        assert!(
            compact.starts_with("__descend_kernel_gpu_k!{"),
            "{expansion}"
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
