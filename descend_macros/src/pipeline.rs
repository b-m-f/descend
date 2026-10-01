use crate::compiler::compile_to_ptx;
use crate::diag::MacroError;
use crate::kernel::Kernel;
use crate::main_fn::synthesize_main_fn;
use crate::parsing::{flatten_source, nat_generics, parse_descend_source, DescendSource};
use crate::specialize::specialize_kernel;
use proc_macro2::TokenStream;
use std::collections::HashMap;

pub fn compile_kernel_fn(
    kernel_name: &str,
    source: &DescendSource,
    compilation_unit: &descend::ast::CompilUnit,
) -> Result<TokenStream, MacroError> {
    let kernel = extract_kernel(compilation_unit, kernel_name, source)?;

    compile_and_emit(&kernel, source, kernel_name)
}

pub fn instantiate(
    source_ts: TokenStream,
    kernel_name: &str,
    item_name: &str,
    bindings: &HashMap<String, String>,
) -> Result<TokenStream, MacroError> {
    check_size_bindings(
        &nat_generics(&source_ts, kernel_name)?,
        kernel_name,
        bindings,
    )?;

    let source = flatten_source(specialize_kernel(source_ts, kernel_name, bindings)?);
    let source_code = descend::parser::SourceCode::new(source.text.clone());
    let compilation_unit = parse_descend_source(&source_code)?;
    let kernel = extract_kernel(&compilation_unit, kernel_name, &source)?;

    compile_and_emit(&kernel, &source, item_name)
}

fn extract_kernel(
    compilation_unit: &descend::ast::CompilUnit,
    kernel_name: &str,
    source: &DescendSource,
) -> Result<Kernel, MacroError> {
    let kernel = Kernel::extract(compilation_unit, kernel_name, source)?;
    crate::shared_mem::check_shared_params(&kernel)?;
    Ok(kernel)
}

fn check_size_bindings(
    sizes: &[String],
    kernel_name: &str,
    bindings: &HashMap<String, String>,
) -> Result<(), MacroError> {
    for size in sizes {
        if !bindings.contains_key(size) {
            return Err(format!(
                "instantiating kernel `{kernel_name}` needs a value for its size `{size}`"
            )
            .into());
        }
    }
    for bound in bindings.keys() {
        if !sizes.contains(bound) {
            return Err(
                format!("kernel `{kernel_name}` has no size `{bound}` to instantiate").into(),
            );
        }
    }

    Ok(())
}

fn compile_and_emit(
    kernel: &Kernel,
    source: &DescendSource,
    item_name: &str,
) -> Result<TokenStream, MacroError> {
    let program = format!("{}\n{}", synthesize_main_fn(kernel), source.text);
    let compiled = compile_to_ptx(&program, &kernel.function_name)?;

    crate::emit::generate_launch_fn(kernel, &compiled, item_name)
}
