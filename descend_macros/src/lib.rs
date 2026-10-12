extern crate proc_macro;

#[macro_use]
mod diag;

mod artifact;
mod assemble;
#[cfg(test)]
mod capture;
mod check;
mod emit;
mod host_ty;
mod invocation;
mod kernel;
mod limits;
mod nat;
mod nvrtc;
mod shared_memory;
mod source;
mod specialize;
mod spell;
mod translate;

use crate::diag::MacroError;
use crate::invocation::{FunctionInternalArguments, FunctionInvocation};
use crate::kernel::Kernel;
use crate::source::DescendSource;
use descend::ast::CompilUnit;
use descend::parser::SourceCode;
use proc_macro2::{Ident, TokenStream};

#[proc_macro]
pub fn descend(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, declare_kernels)
}

#[proc_macro]
pub fn descend_fn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, route_to_carrier)
}

#[proc_macro]
pub fn __descend_fn_internal(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_or_error(input, instantiate_kernel)
}

fn declare_kernels(input: TokenStream) -> Result<TokenStream, MacroError> {
    let source_token_stream = input.clone();
    let source = render_source(input);
    let code = SourceCode::new(source.text.clone());
    let unit = source::parse_descend_source(&code, &source)?;
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
    let unit = source::parse_descend_source(&code, &source)?;
    let kernel = Kernel::extract(&unit, &kernel_name, &source)?;

    let declared_at = kernel.name_span;
    let launch = check::check_kernel(&kernel)?;
    let device_code =
        translate::translate(&source.text, &kernel_name, |span| source.span_for(span))
            .map_err(|error| error.or_span(declared_at))?;
    let ptx = assemble::assemble(&device_code, &kernel_name)
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
