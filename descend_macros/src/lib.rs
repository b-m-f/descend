extern crate proc_macro;

#[macro_use]
mod helper;

mod compiler;
mod format;
mod host_api;
mod kernel_ir;
mod main_workaround;
mod parser;
mod shared_mem;

use compiler::compile_to_ptx;
use kernel_ir::Kernel;
use main_workaround::synthesize_main_program;
use proc_macro::{Delimiter, Group, TokenStream, TokenTree};
use std::str::FromStr;

/// Recursively traverses a `TokenStream` to replace a specific placeholder identifier with the user's pristine code.
///
/// What it does:
/// - `TokenTree::Group` match: Recursively descends into nested brackets/braces (like `{ ... }` or `( ... )`). `TokenStream`s are tree structures, not flat arrays. Without this recursion, we would fail to find placeholders hidden inside the macro's inner blocks.
/// - `TokenTree::Ident` match: Looks for the exact literal word `_SOURCE_PLACEHOLDER_` and swaps it out for the user's untouched `TokenStream` (wrapped in braces).
///
/// Why it is needed: This technique allows us to use highly readable string templates (e.g. `format!("macro_rules! ...")`) for boilerplate macro generation, while cleanly injecting the user's original `TokenStream` without stringifying it. This perfectly preserves compiler span hygiene so `rustc` errors point directly to the user's specific faulty syntax rather than highlighting the entire macro block.
fn substitute_source_placeholder(ts: TokenStream, source_ts: &TokenStream) -> TokenStream {
    let mut out = TokenStream::new();
    for tt in ts {
        match tt {
            TokenTree::Group(g) => {
                let new_stream = substitute_source_placeholder(g.stream(), source_ts);
                out.extend(std::iter::once(TokenTree::Group(Group::new(
                    g.delimiter(),
                    new_stream,
                ))));
            }
            TokenTree::Ident(id) if id.to_string() == "_SOURCE_PLACEHOLDER_" => {
                out.extend(std::iter::once(TokenTree::Group(Group::new(
                    Delimiter::Brace,
                    source_ts.clone(),
                ))));
            }
            other => {
                out.extend(std::iter::once(other));
            }
        }
    }
    out
}

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
pub fn descend(input: TokenStream) -> TokenStream {
    match expand_descend(input) {
        Ok(tokens) => tokens,
        Err(message) => crate::helper::compile_error_item(&message),
    }
}

fn expand_descend(input: TokenStream) -> Result<TokenStream, String> {
    let source_ts = input.clone();

    // We format the raw tokens tightly (removing syntax-invalid spaces)
    // to strictly satisfy the Descend PEG parser requirements.
    let source_str = crate::format::to_descend_string(source_ts.clone());
    debug!("MACRO INPUT: {}", source_str);
    let source = descend::parser::SourceCode::new(source_str);
    let compilation_unit = crate::parser::parse_descend_source(&source)
        .map_err(|err| format!("failed to parse the Descend source: {err}"))?;

    let mut output = TokenStream::new();

    // Support arbitrary numbers of kernels within a single block
    for item in &compilation_unit.items {
        match item {
            descend::ast::Item::FunDef(f) if crate::kernel_ir::is_launchable(f) => {
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
                    output.extend(compile_kernel_fn(
                        &kernel_name,
                        &source_ts,
                        &compilation_unit,
                    )?);
                }
            }
            descend::ast::Item::StructDecl(s) => {
                let struct_code = crate::parser::format_rust_struct(s);
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

/// Compiles one kernel and emits the function that launches it.
fn compile_kernel_fn(
    kernel_name: &str,
    source_ts: &TokenStream,
    compilation_unit: &descend::ast::CompilUnit,
) -> Result<TokenStream, String> {
    let kernel = Kernel::extract(compilation_unit, kernel_name)?;
    crate::host_api::check_shared_params(&kernel)?;

    let full_descend = synthesize_main_program(
        source_ts.clone(),
        &kernel,
        &std::collections::HashMap::new(),
    );
    let compiled = compile_to_ptx(&full_descend, &kernel.function_name)?;

    crate::host_api::generate_launch_fn(
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
    let template = format!(
        "macro_rules! {0} {{ \
           (@fn $name:ident, $($args:tt)*) => {{ \
             ::descend_macros::_descend_fn_internal!{{ _SOURCE_PLACEHOLDER_, {0}, $name, $($args)* }} \
           }}; \
           ($($args:tt)*) => {{ \
             compile_error!(concat!( \
               \"kernel `{0}` is launched through a generated function, not through a macro. \
                 Name an instantiation with `descend_fn!({0} as <name>, <...sizes>);` and call \
                 `<name>(&stream, ..)`.\" \
             )) \
           }} \
         }}",
        kernel_name
    );
    let template_ts = template
        .parse::<TokenStream>()
        .expect("carrier macro template must parse");
    substitute_source_placeholder(template_ts, source_ts)
}

/// Instantiates a generic kernel as a named Rust function.
///
/// What it does: rewrites `descend_fn!(gpu_matmul as tile64, <bs: 16, m: 64>)`
/// into an invocation of the `macro_rules!` that `descend!` generated for the
/// kernel, which carries the kernel's source to this site.
/// Why it is needed: a kernel with `nat` generics cannot be compiled where it is
/// defined, because its sizes are baked into the generated CUDA and only become
/// literals here. Naming an instantiation is what turns such a kernel into an
/// ordinary function item, which — unlike a macro — can be exported from its
/// module and called from anywhere.
#[proc_macro]
pub fn descend_fn(input: TokenStream) -> TokenStream {
    match expand_descend_fn(input) {
        Ok(tokens) => tokens,
        Err(message) => crate::helper::compile_error_item(&message),
    }
}

fn expand_descend_fn(input: TokenStream) -> Result<TokenStream, String> {
    let mut iter = input.into_iter();

    let Some(TokenTree::Ident(kernel)) = iter.next() else {
        return Err(
            "expected a kernel name, as in `descend_fn!(gpu_matmul as tile64, <m: 64>)`"
                .to_string(),
        );
    };
    match iter.next() {
        Some(TokenTree::Ident(keyword)) if keyword.to_string() == "as" => {}
        _ => return Err(format!("expected `as <name>` after the kernel `{kernel}`")),
    }
    let Some(TokenTree::Ident(item_name)) = iter.next() else {
        return Err(format!(
            "expected a name for the instantiation of `{kernel}`"
        ));
    };
    match iter.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == ',' => {}
        _ => return Err(format!("expected `,` after the name `{item_name}`")),
    }

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
pub fn _descend_fn_internal(input: TokenStream) -> TokenStream {
    match fn_internal(input) {
        Ok(tokens) => tokens,
        Err(message) => crate::helper::compile_error_item(&message),
    }
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
    let item_name = iter
        .next()
        .ok_or_else(|| "expected the name of the instantiation".to_string())?
        .to_string();
    let _comma = iter.next();

    let bindings = crate::parser::parse_generic_bindings(iter.collect())?;

    let source_str = crate::format::to_descend_string(source_ts.clone());
    let source = descend::parser::SourceCode::new(source_str);
    let compilation_unit = crate::parser::parse_descend_source(&source)
        .map_err(|err| format!("failed to parse the Descend source: {err}"))?;

    let kernel = Kernel::extract(&compilation_unit, &kernel_name)?;
    crate::host_api::check_shared_params(&kernel)?;

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

    let full_descend = synthesize_main_program(source_ts, &kernel, &bindings);
    let compiled = compile_to_ptx(&full_descend, &kernel.function_name)?;

    crate::host_api::generate_launch_fn(&kernel, &compiled, &bindings, &item_name)
}
