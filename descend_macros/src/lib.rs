extern crate proc_macro;

#[macro_use]
mod diag;

mod artifact;
mod assemble;
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
    let unit = source::parse_descend_source(&code)?;
    declare(&unit, &source, &source_token_stream)
}

fn route_to_carrier(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FunctionInvocation {
        kernel,
        dimension_annotation,
        bindings,
    } = invocation::parse_function_invocation(input)?;
    emit::carrier_invocation(&kernel, dimension_annotation.as_ref(), bindings)
}

fn instantiate_kernel(input: TokenStream) -> Result<TokenStream, MacroError> {
    let FunctionInternalArguments {
        source_token_stream,
        kernel,
        dimension_annotation,
        bindings,
    } = invocation::parse_function_internal_arguments(input)?;
    let asked_at = kernel.span();

    instantiate(source_token_stream, &kernel, dimension_annotation, bindings)
        .map(|items| emit::block_yielding(items, &kernel))
        .map_err(|error| error.or_span(Some(asked_at)))
}

fn instantiate(
    source_token_stream: TokenStream,
    kernel: &Ident,
    dimension_annotation: Option<crate::invocation::DimensionAnnotation>,
    bindings: TokenStream,
) -> Result<TokenStream, MacroError> {
    let kernel_name = kernel.to_string();

    let sizes = invocation::parse_sizes(bindings)?;
    let specialized = specialize::specialize_kernel(source_token_stream, &kernel_name, &sizes)?;
    let source = render_source(specialized);
    let code = SourceCode::new(source.text.clone());
    let mut unit = source::parse_descend_source(&code)?;
    let kernel = Kernel::extract(&unit, &kernel_name, &source)?;

    let declared_at = kernel.name_span;
    let launch_config = check::check_kernel(&kernel)?;
    if let Some(dimension_annotation) = &dimension_annotation {
        check::confirm_dimensions(&kernel, &launch_config, dimension_annotation)?;
    }
    let device_code =
        translate::translate(&code, &mut unit, &kernel_name, kernel.shared_memory_dtys())
            .map_err(|error| error.or_span(declared_at))?;
    let ptx = assemble::assemble(&device_code, &kernel_name)
        .map_err(|error| error.or_span(declared_at))?;

    emit::generate_launch_function(&kernel, &launch_config, &ptx)
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
                debug!(
                    "custom types are not supported yet: struct `{}` is not generated",
                    struct_decl.ident.name
                );
            }
            _ => {}
        }
    }

    if output.is_empty() {
        return Err(
            "this `descend!` block declares nothing. A kernel is declared by an \
                    execution annotation `-[grid: gpu.grid<.., ..>]->`; a `struct` emits \
                    nothing, because custom types are not supported yet"
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    const KERNEL_A: &str = "fn gpu_a<r: prv>(a: &r uniq gpu.global [i32; 256]) \
                            -[grid: gpu.grid<X<1>, X<256>>]-> () { \
                            sched block in grid { sched thread in block { () } } }";
    const KERNEL_B: &str = "fn gpu_b<r: prv>(b: &r uniq gpu.global [i32; 256]) \
                            -[grid: gpu.grid<X<1>, X<256>>]-> () { \
                            sched block in grid { sched thread in block { () } } }";

    fn declared(source_text: &str) -> Result<String, MacroError> {
        let source_token_stream =
            TokenStream::from_str(source_text).expect("the fixture must tokenize");
        let source = render_source(source_token_stream.clone());
        let code = SourceCode::new(source.text.clone());
        let unit = source::parse_descend_source(&code)?;
        declare(&unit, &source, &source_token_stream).map(|output| output.to_string())
    }

    #[test]
    fn every_kernel_of_a_block_gets_a_carrier_named_after_it() {
        let output = declared(&format!("{KERNEL_A} {KERNEL_B}")).expect("both kernels declare");

        assert!(output.contains("__descend_kernel_gpu_a"), "{output}");
        assert!(output.contains("__descend_kernel_gpu_b"), "{output}");
    }

    #[test]
    fn every_carrier_carries_the_whole_block() {
        let output = declared(&format!("{KERNEL_A} {KERNEL_B}")).expect("both kernels declare");

        assert_eq!(
            output.matches("fn gpu_a").count(),
            2,
            "one copy per carrier, since a carrier carries its block: {output}"
        );
        assert_eq!(output.matches("fn gpu_b").count(), 2, "{output}");
    }

    #[test]
    fn a_declared_type_is_carried_by_every_kernel_but_not_generated() {
        let output = declared(&format!(
            "struct Point {{ x: i32, y: i32 }} {KERNEL_A} {KERNEL_B}"
        ))
        .expect("the block declares");

        assert_eq!(
            output.matches("struct Point").count(),
            2,
            "one carried copy per kernel, and no Rust definition: {output}"
        );
    }

    #[test]
    fn a_block_with_one_kernel_carries_its_helpers_and_declared_types() {
        let output = declared(&format!(
            "struct Point {{ x: i32, y: i32 }} \
             fn helper() -[t: cpu.thread]-> () {{ () }} {KERNEL_A}"
        ))
        .expect("the block declares");

        assert!(output.contains("__descend_kernel_gpu_a"), "{output}");
        assert!(output.contains("fn helper"), "{output}");
        assert_eq!(
            output.matches("struct Point").count(),
            1,
            "the copy the single kernel carries, and no Rust definition: {output}"
        );
    }

    #[test]
    fn a_kernel_declared_twice_in_one_block_is_refused() {
        let error = declared(&format!("{KERNEL_A} {KERNEL_A}")).unwrap_err();

        assert!(error.message.contains("declared twice"), "{error}");
    }
}
