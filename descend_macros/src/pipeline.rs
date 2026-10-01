use crate::compiler::compile_to_ptx;
use crate::kernel::Kernel;
use crate::specialize::synthesize_main_program;
use proc_macro2::TokenStream;
use std::collections::HashMap;

pub fn compile_kernel_fn(
    kernel_name: &str,
    source_ts: &TokenStream,
    compilation_unit: &descend::ast::CompilUnit,
) -> Result<TokenStream, String> {
    let kernel = extract_kernel(compilation_unit, kernel_name)?;

    compile_and_emit(
        &kernel,
        source_ts.clone(),
        &std::collections::HashMap::new(),
        kernel_name,
    )
}

pub fn instantiate(
    compilation_unit: &descend::ast::CompilUnit,
    source_ts: TokenStream,
    kernel_name: &str,
    item_name: &str,
    bindings: &HashMap<String, String>,
) -> Result<TokenStream, String> {
    let kernel = extract_kernel(compilation_unit, kernel_name)?;
    check_size_bindings(&kernel, kernel_name, bindings)?;

    compile_and_emit(&kernel, source_ts, bindings, item_name)
}

fn extract_kernel(
    compilation_unit: &descend::ast::CompilUnit,
    kernel_name: &str,
) -> Result<Kernel, String> {
    let kernel = Kernel::extract(compilation_unit, kernel_name)?;
    crate::shared_mem::check_shared_params(&kernel)?;
    Ok(kernel)
}

fn check_size_bindings(
    kernel: &Kernel,
    kernel_name: &str,
    bindings: &HashMap<String, String>,
) -> Result<(), String> {
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

    Ok(())
}

fn compile_and_emit(
    kernel: &Kernel,
    source_ts: TokenStream,
    bindings: &HashMap<String, String>,
    item_name: &str,
) -> Result<TokenStream, String> {
    let full_descend = synthesize_main_program(source_ts, kernel, bindings)?;
    let compiled = compile_to_ptx(&full_descend, &kernel.function_name)?;

    crate::emit::generate_launch_fn(kernel, &compiled, bindings, item_name)
}
