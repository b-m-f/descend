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
mod nat;
mod nvcc;
mod source;
mod specialize;
mod spell;
mod synthesize;
mod translate;

use crate::artifact::{LaunchConfig, Ptx};
use crate::diag::MacroError;
use crate::invocation::{FunctionInternalArguments, FunctionInvocation};
use crate::kernel::Kernel;
use crate::source::DescendSource;
use crate::translate::DeviceCpp;
use descend::ast::CompilUnit;
use descend::parser::SourceCode;
use proc_macro2::{Ident, TokenStream};
use std::collections::HashMap;

#[proc_macro]
pub fn descend(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, declare_kernels)
}

#[proc_macro]
pub fn descend_fn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, route_to_carrier)
}

/// Internal: invoked only by the carrier a `descend!` emits, never written by hand.
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
    let unit = parse(&code)?;
    declare(&unit, &source, &source_token_stream)
}

// `descend_fn!` is an expression: it evaluates to the instantiated function, so the
// caller names it with an ordinary `let`. A procedural macro cannot tell item position
// from expression position, so this is the only position the macro supports; an
// instantiation that has to be reachable elsewhere goes in a hand-written wrapper
// function.
fn route_to_carrier(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FunctionInvocation { kernel, bindings } = parse_invocation(input)?;
    emit::carrier_invocation(&kernel, bindings)
}

fn instantiate_kernel(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FunctionInternalArguments {
        source_token_stream,
        kernel,
        bindings,
    } = parse_internal_args(input)?;
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

    let sizes = parse_sizes(bindings)?;
    let specialized = specialize(source_token_stream, &kernel_name, &sizes)?;
    let source = render_source(specialized);
    let code = SourceCode::new(source.text.clone());
    let unit = parse(&code)?;
    let kernel = extract(&unit, &kernel_name, &source)?;

    let declared_at = kernel.name_span;
    let launch = check(&kernel)?;
    let program = synthesize(&kernel, &source).map_err(|error| error.or_span(declared_at))?;
    let cpp = translate(&program, &kernel_name).map_err(|error| error.or_span(declared_at))?;
    let ptx = assemble(&cpp, &kernel_name).map_err(|error| error.or_span(declared_at))?;

    emit_launch_function(&kernel, &launch, &ptx).map_err(|error| error.or_span(declared_at))
}

fn render_source(input: TokenStream) -> DescendSource {
    let source = source::DescendSource::from_token_stream(input);
    debug!("MACRO INPUT: {}", source.text);
    source
}

fn parse<'a>(code: &'a SourceCode<'a>) -> Result<CompilUnit<'a>, MacroError> {
    source::parse_descend_source(code).map_err(MacroError::from)
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
                let struct_code = emit::format_rust_struct(struct_decl)?;
                output.extend(
                    struct_code
                        .parse::<TokenStream>()
                        .map_err(|error| format!("generated type is malformed: {error}"))?,
                );
            }
            _ => {}
        }
    }

    if output.is_empty() {
        return Err(
            "this `descend!` block declares nothing. A kernel is declared by an \
                    execution annotation `-[grid: gpu.grid<.., ..>]->`, and a type by a \
                    `struct`; without either, the macro emits no items at all"
                .to_string()
                .into(),
        );
    }

    Ok(output)
}

fn parse_invocation(input: TokenStream) -> Result<FunctionInvocation, MacroError> {
    invocation::parse_function_invocation(input).map_err(MacroError::from)
}

fn parse_internal_args(input: TokenStream) -> Result<FunctionInternalArguments, MacroError> {
    invocation::parse_function_internal_arguments(input).map_err(MacroError::from)
}

fn parse_sizes(bindings: TokenStream) -> Result<HashMap<String, String>, MacroError> {
    invocation::parse_sizes(bindings)
}

fn specialize(
    source_token_stream: TokenStream,
    kernel_name: &str,
    sizes: &HashMap<String, String>,
) -> Result<TokenStream, MacroError> {
    specialize::specialize_kernel(source_token_stream, kernel_name, sizes)
}

fn extract(
    unit: &CompilUnit,
    kernel_name: &str,
    source: &DescendSource,
) -> Result<Kernel, MacroError> {
    Kernel::extract(unit, kernel_name, source).map_err(MacroError::from)
}

fn check(kernel: &Kernel) -> Result<LaunchConfig, MacroError> {
    check::check_kernel(kernel)
}

fn synthesize(kernel: &Kernel, source: &DescendSource) -> Result<String, MacroError> {
    let main_function = synthesize::synthesize_main_function(kernel)?;
    Ok(format!("{}\n{}", main_function, source.text))
}

fn translate(program: &str, kernel_name: &str) -> Result<DeviceCpp, MacroError> {
    translate::translate(program, kernel_name).map_err(MacroError::from)
}

fn assemble(cpp: &DeviceCpp, kernel_name: &str) -> Result<Ptx, MacroError> {
    assemble::assemble(&cpp.code, kernel_name).map_err(MacroError::from)
}

fn emit_launch_function(
    kernel: &Kernel,
    launch: &LaunchConfig,
    ptx: &Ptx,
) -> Result<TokenStream, MacroError> {
    emit::generate_launch_function(kernel, launch, ptx)
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
