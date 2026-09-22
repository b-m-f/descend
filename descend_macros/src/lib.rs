extern crate proc_macro;

#[macro_use]
mod diag;

/// Descend AST helpers: reading a kernel's shape out of the parsed source.
mod ast;
/// Type checking, code generation, `nvcc`, and the PTX cache.
mod compile;
/// Tokens to Descend source text.
mod descend_src;
/// The Rust items a kernel becomes.
mod emit;
/// The kernel IR the rest of the crate works from.
mod ir;
/// `Nat` evaluation and the one remaining text substitution.
mod nat;
/// The dynamic shared-memory byte count and the checks around it.
mod shared_mem;
/// Specializing a kernel's sizes and synthesizing a compilable program.
mod specialize;

use compile::compile_to_ptx;
use ir::Kernel;
use proc_macro2::{Delimiter, Group, Span, TokenStream, TokenTree};
use quote::{format_ident, quote};
use specialize::synthesize_main_program;

/// Runs an expansion over `proc_macro2` tokens and turns its failure into a
/// `compile_error!`.
///
/// The three entry points are the crate's only `proc_macro` types: everything
/// below them works on `proc_macro2`, which — unlike `proc_macro` — can be
/// constructed outside a macro invocation and so can be unit-tested.
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

/// The primary entry point for embedding Descend code in Rust.
///
/// What it does: compiles every launchable kernel whose sizes are already fixed
/// into a callable Rust function of the same name, and leaves a kernel with
/// `nat` generics to be instantiated by [`descend_fn`], which is the only way to
/// name its sizes. Descend structs become `#[repr(C)]` Rust structs.
///
/// Why it is needed: users invoke `descend! { ... }` with arbitrary Descend code
/// (structs, functions, kernels), and every kernel needs to be callable from
/// Rust as a function. Whether that function exists immediately or after an
/// instantiation is decided by the kernel's own signature, not by the user.
#[proc_macro]
pub fn descend(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, expand_descend)
}

fn expand_descend(input: TokenStream) -> Result<TokenStream, String> {
    let source_ts = input.clone();

    // We format the raw tokens tightly (removing syntax-invalid spaces)
    // to strictly satisfy the Descend PEG parser requirements.
    let source_str = crate::descend_src::to_descend_string(source_ts.clone());
    debug!("MACRO INPUT: {}", source_str);
    let source = descend::parser::SourceCode::new(source_str);
    let compilation_unit = crate::ast::parse_descend_source(&source)
        .map_err(|err| format!("failed to parse the Descend source: {err}"))?;

    let mut output = TokenStream::new();

    // Support arbitrary numbers of kernels within a single block
    for item in &compilation_unit.items {
        match item {
            descend::ast::Item::FunDef(f) if crate::ir::is_launchable(f) => {
                let kernel_name = f.ident.name.to_string();
                let has_nat_generics = f
                    .generic_params
                    .iter()
                    .any(|param| matches!(param.kind, descend::ast::Kind::Nat));

                // Every kernel gets the carrier, so that any of them can be
                // instantiated under another name later.
                output.extend(carrier_macro_for(&kernel_name, &source_ts));

                // A kernel whose sizes are already fixed is compiled here and
                // becomes a function straight away. A generic one waits for an
                // instantiation to name its sizes.
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
            // A function that does not execute over a grid is callable from
            // Descend code but is not a launchable entry point.
            _ => {}
        }
    }

    Ok(output)
}

/// The span of `name` where the block declares it, so a failure about this
/// kernel underlines its name.
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

/// Compiles one kernel and emits the function that launches it.
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

/// Emits the `macro_rules!` that carries a kernel's source to a later
/// instantiation site.
///
/// A kernel with `nat` generics cannot be compiled where it is written: its
/// sizes are substituted into the Descend source as literals before code
/// generation, so they have to be named at a compile-time site. This macro is
/// how the source reaches that site. It is not a way to launch the kernel —
/// launching is what the generated function is for — so every invocation other
/// than an instantiation is rejected with a message saying so.
fn carrier_macro_for(kernel_name: &str, source_ts: &TokenStream) -> TokenStream {
    let kernel = format_ident!("{}", kernel_name);
    let refusal = format!(
        "kernel `{kernel_name}` is launched through a generated function, not through a macro. \
         Write `descend_fn!({kernel_name}, <...sizes>);` once and call \
         `{kernel_name}(&stream, ..)`; use `descend_fn!({kernel_name} as <name>, <...sizes>);` \
         for a second set of sizes."
    );

    // `quote!` leaves `$` alone, so the `macro_rules!` metavariables pass
    // through, and the kernel's source travels inside the braced group.
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

/// Instantiates a generic kernel as a named Rust function.
///
/// What it does: rewrites `descend_fn!(gpu_vec_add, <N: 16384>)` — or
/// `descend_fn!(gpu_matmul as tile64, <bs: 16, m: 64>)` when one kernel needs
/// several sizes — into an invocation of the `macro_rules!` that `descend!`
/// generated for the kernel, which carries the kernel's source to this site.
/// Why it is needed: a kernel with `nat` generics cannot be compiled where it is
/// defined, because its sizes are baked into the generated CUDA and only become
/// literals here. Naming an instantiation is what turns such a kernel into an
/// ordinary function item, which — unlike a macro — can be exported from its
/// module and called from anywhere.
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

    // `descend_fn!(kernel, <sizes>)` names the function after the kernel;
    // `descend_fn!(kernel as name, <sizes>)` names it explicitly, which is what
    // distinguishes two instantiations of one kernel.
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

/// Internal counterpart of [`descend_fn`], reached through the kernel's own
/// generated macro so that it has the kernel's source.
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
    // This token comes from the user's `descend_fn!`, so anything that goes
    // wrong with the instantiation can be reported against it.
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

/// Compiles one instantiation of a kernel and emits its function.
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
