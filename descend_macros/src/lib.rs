//! The procedural macros that compile Descend into a callable Rust function.
//!
//! Two macros are meant to be written. [`descend!`] declares kernels and types; it
//! compiles nothing. [`descend_fn!`] instantiates one declared kernel at one set of
//! sizes, and that is where nvcc runs. A third, [`__descend_fn_internal`], is plumbing:
//! the carrier macro a `descend!` emits calls it, and nobody writes it by hand.
//!
//! Everything else in this crate is private, so the stages below are named as code
//! rather than linked: rustdoc for a proc-macro crate shows only its macros.
//!
//! # The instantiation pipeline
//!
//! `instantiate` is the whole of it, and reads as an ordered list of stages:
//!
//! 1. `invocation::parse_sizes` — read the `<N: 16384>` bindings.
//! 2. `specialize::specialize_kernel` — substitute them into the kernel's tokens.
//! 3. `source::DescendSource` — render the token stream back to Descend source, keeping
//!    a map from each byte offset to the span it came from.
//! 4. `source::parse_descend_source` — parse that into Descend's AST.
//! 5. `kernel::Kernel::extract` — pull the named kernel out, with its parameters as
//!    `host_ty::HostTy` so the host and device spellings cannot drift apart.
//! 6. `check::check_kernel` — refuse what cannot be launched, before anything is
//!    compiled. The bounds are in `limits`.
//! 7. `synthesize::synthesize_main_function` — write the Descend `main` the kernel is
//!    type-checked against.
//! 8. `translate::translate` — type-check, then generate CUDA and strip it down to a
//!    device-only translation unit.
//! 9. `assemble::assemble` — run nvcc, and find the kernel's entry symbol in the PTX.
//! 10. `emit::generate_launch_function` — the Rust function the caller binds.
//!
//! # Errors
//!
//! A *stage* returns `diag::MacroError`, which carries the span to point the compiler
//! at. The sub-computations a stage calls — `nat`, `spell`, `host_ty::Unrepresentable` —
//! return a *reason fragment* instead: a clause that completes a sentence the stage
//! writes around it, as in ``kernel `k`: parameter `p` is a tuple. …``. Fragments
//! deliberately do not know about spans, because only the stage knows which token the
//! reader has to change.
//!
//! # Environment
//!
//! | Variable | Effect |
//! |---|---|
//! | `DESCEND_DEBUG` | Print the rendered Descend source, the generated CUDA, and the PTX cache decisions to stderr. |
//! | `DESCEND_NVCC_FLAGS` | Extra arguments for nvcc. Without an `-arch=`, one is detected through `nvidia-smi`. |
//! | `DESCEND_NO_CACHE` | Compile every kernel, ignoring and not writing the PTX cache. |
//! | `DESCEND_PTX_CACHE` | Where to keep the PTX cache. See `nvcc::cache_directory` for the search order; in practice it lands in `~/.cache/descend/ptx`, because Cargo does not set `CARGO_TARGET_DIR` for a proc macro. |

extern crate proc_macro;

#[macro_use]
mod diag;

mod artifact;
mod assemble;
mod capture;
mod check;
#[cfg(test)]
mod const_expr;
mod emit;
mod host_ty;
mod invocation;
mod kernel;
#[cfg(test)]
mod launch_scrape;
mod limits;
mod nat;
mod nvcc;
mod source;
mod specialize;
mod spell;
mod synthesize;
mod translate;

use crate::diag::MacroError;
use crate::invocation::{FunctionInternalArguments, FunctionInvocation};
use crate::kernel::Kernel;
use crate::source::DescendSource;
use descend::ast::CompilUnit;
use descend::parser::SourceCode;
use proc_macro2::{Ident, TokenStream};

/// Declare Descend kernels and types.
///
/// Each kernel becomes a carrier macro holding its source; nothing is compiled here.
/// Each `struct` becomes its `#[repr(C)]` Rust counterpart immediately, because a kernel
/// taking it needs that type in scope to be callable at all.
///
/// A kernel is declared by an execution annotation `-[grid: gpu.grid<.., ..>]->`. Use
/// [`descend_fn!`] to instantiate one.
#[proc_macro]
pub fn descend(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, declare_kernels)
}

/// Instantiate a kernel declared by [`descend!`], evaluating to the function.
///
/// ```ignore
/// let gpu_vec_add = descend_fn!(gpu_vec_add, <N: 16384>);
/// ```
///
/// This is an expression, so the caller names the result with an ordinary `let`. A
/// procedural macro cannot tell item position from expression position, so this is the
/// only position the macro supports; an instantiation that has to be reachable elsewhere
/// goes in a hand-written wrapper function.
///
/// Instantiating is what compiles: this runs Descend's type checker and nvcc, and embeds
/// the resulting PTX in the generated function.
#[proc_macro]
pub fn descend_fn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, route_to_carrier)
}

/// Internal: invoked only by the carrier a `descend!` emits, never written by hand.
///
/// The carrier exists so that a kernel's name is *not* what carries its source. Under a
/// name nobody writes, calling a declared-but-uninstantiated kernel fails as `cannot
/// find function` rather than as `expected function, found macro`. See `emit::carrier_name`.
#[proc_macro]
pub fn __descend_fn_internal(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, instantiate_kernel)
}

// `descend!` declares; it does not compile. Whatever a kernel's sizes are, it becomes a
// function only where a `descend_fn!` instantiates it.
fn declare_kernels(input: TokenStream) -> Result<TokenStream, MacroError> {
    let source_token_stream = input.clone();
    let source = render_source(input);
    let code = SourceCode::new(source.text.clone());
    let unit = source::parse_descend_source(&code)?;
    declare(&unit, &source, &source_token_stream)
}

fn route_to_carrier(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FunctionInvocation { kernel, bindings } = invocation::parse_function_invocation(input)?;
    emit::carrier_invocation(&kernel, bindings)
}

fn instantiate_kernel(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FunctionInternalArguments {
        source_token_stream,
        kernel,
        bindings,
    } = invocation::parse_function_internal_arguments(input)?;
    let asked_at = kernel.span();

    instantiate(source_token_stream, &kernel, bindings)
        // The generated function keeps the kernel's name. It lives in this block, so it
        // cannot collide with the binding the caller writes, nor with another
        // instantiation of the same kernel in the same scope.
        .map(|items| emit::block_yielding(items, &kernel))
        .map_err(|error| error.or_span(Some(asked_at)))
}

fn instantiate(
    source_token_stream: TokenStream,
    kernel: &Ident,
    bindings: TokenStream,
) -> Result<TokenStream, MacroError> {
    let kernel_name = kernel.to_string();

    let sizes = invocation::parse_sizes(bindings)?;
    let specialized = specialize::specialize_kernel(source_token_stream, &kernel_name, &sizes)?;
    let source = render_source(specialized);
    let code = SourceCode::new(source.text.clone());
    let unit = source::parse_descend_source(&code)?;
    let kernel = Kernel::extract(&unit, &kernel_name, &source)?;

    // From here the stages are working from Descend's AST rather than from the tokens
    // the reader wrote, so not all of them can name the token that offended them. The
    // kernel's declaration is the fallback span for the rest of the pipeline.
    let declared_at = kernel.name_span;
    let launch = check::check_kernel(&kernel)?;
    let main_function = synthesize::synthesize_main_function(&kernel)
        .map_err(|error| error.or_span(declared_at))?;
    let program = format!("{}\n{}", main_function, source.text);
    let cpp_source =
        translate::translate(&program, &kernel_name).map_err(|error| error.or_span(declared_at))?;
    let ptx = assemble::assemble(&cpp_source.device_code, &kernel_name)
        .map_err(|error| error.or_span(declared_at))?;

    emit::generate_launch_function(&kernel, &launch, &ptx)
        .map_err(|error| error.or_span(declared_at))
}

fn render_source(input: TokenStream) -> DescendSource {
    let source = source::DescendSource::from_token_stream(input);
    debug!("MACRO INPUT: {}", source.text);
    source
}

fn declare(
    unit: &CompilUnit,
    source: &DescendSource,
    source_token_stream: &TokenStream,
) -> Result<TokenStream, MacroError> {
    let mut output = TokenStream::new();
    let mut declared_names: Vec<String> = Vec::new();

    for item in &unit.items {
        match item {
            // The carrier that carries this source to an instantiation is all that is
            // emitted for a kernel here.
            descend::ast::Item::FunDef(fun_def) if kernel::is_launchable(fun_def) => {
                let kernel_name = fun_def.ident.name.to_string();
                if declared_names.contains(&kernel_name) {
                    return Err(MacroError::at(
                        format!(
                            "kernel `{kernel_name}` is declared twice in this `descend!` block. Each \
                             kernel carries its source under a macro named after it, so the \
                             second declaration would silently shadow the first"
                        ),
                        fun_def
                            .ident
                            .span
                            .and_then(|span| source.span_at(span.begin)),
                    ));
                }
                declared_names.push(kernel_name.clone());
                output.extend(emit::carrier_macro(&kernel_name, source_token_stream));
            }
            // A declared type becomes its Rust counterpart here. Unlike a kernel this
            // costs nothing to emit -- no nvcc, just the type -- and a kernel that uses
            // it needs the Rust type in scope to be callable at all.
            descend::ast::Item::StructDecl(struct_decl) => {
                let struct_code = emit::format_rust_struct(struct_decl, source)?;
                output.extend(struct_code.parse::<TokenStream>().map_err(|error| {
                    MacroError::from(format!(
                        "the definition generated for struct `{}` is malformed: {error}",
                        struct_decl.ident.name
                    ))
                })?);
            }
            _ => {}
        }
    }

    if output.is_empty() {
        return Err(
            "this `descend!` block declares nothing. A kernel is declared by an \
                    execution annotation `-[grid: gpu.grid<.., ..>]->`, and a type by a \
                    `struct`; without either, the macro emits no items at all"
                .into(),
        );
    }

    Ok(output)
}

fn expand_or_error(
    input: proc_macro::TokenStream,
    expand: fn(TokenStream) -> Result<TokenStream, MacroError>,
) -> proc_macro::TokenStream {
    match expand(input.into()) {
        Ok(tokens) => tokens,
        Err(error) => error.to_tokens(),
    }
    .into()
}
