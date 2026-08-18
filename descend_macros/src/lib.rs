#![feature(proc_macro_span)]
extern crate proc_macro;

mod compiler;
#[macro_use]
mod helper;
mod host_api;
mod kernel_ir;
mod main_workaround;
mod parser;
mod sanitizer;

use compiler::compile_to_ptx;
use host_api::generate_host_wrapper;
use kernel_ir::Kernel;
use main_workaround::synthesize_main_program;
use proc_macro::TokenStream;
use sanitizer::extract_source_str;

#[proc_macro]
pub fn descend_kernel(input: TokenStream) -> TokenStream {
    eprintln!(">>> descend_kernel START");
    let mut source_str = extract_source_str(&input);
    let call_args = crate::parser::extract_call_args(&mut source_str);

    let source = descend::parser::SourceCode::new(source_str.clone());
    let compilation_unit = match crate::parser::parse_descend_source(&source) {
        Ok(c) => c,
        Err(err) => {
            return format!("compile_error!({:?});", err).parse().unwrap();
        }
    };

    let mut kernel = match Kernel::extract(&compilation_unit) {
        Ok(i) => i,
        Err(err) => {
            return format!("compile_error!({:?});", err).parse().unwrap();
        }
    };
    kernel.call_args = call_args;

    let kernel_name = kernel.function_name.clone();

    // Generate a declarative macro named after the kernel
    // It captures the source code and forwards it to descend_launch_internal
    let output = format!(
        // TODO: eliminate potential delimiter confusion if source contains four times #
        r####"
        macro_rules! _descend_internal_{} {{
            ($($args:tt)*) => {{
                ::descend_macros::_descend_launch_internal!(
                    r###"{}"###,
                    {} $($args)*
                )
            }}
        }}
        "####,
        kernel_name, source_str, kernel_name
    );

    eprintln!(">>> descend_kernel END");
    output.parse().unwrap()
}

#[proc_macro]
pub fn descend_launch(input: TokenStream) -> TokenStream {
    eprintln!(">>> descend_launch START proxy");
    let mut iter = input.into_iter();

    // First token is the kernel name identifier
    let kernel_ident = match iter.next() {
        Some(proc_macro::TokenTree::Ident(id)) => id,
        _ => {
            return "compile_error!(\"Expected kernel name identifier in descend_launch!\");"
                .parse()
                .unwrap()
        }
    };

    // The rest of the tokens are the launch arguments
    let rest: TokenStream = iter.collect();

    // Expand to: `kernel_name!(rest...)`
    let output = format!(
        "_descend_internal_{}!({});",
        kernel_ident.to_string(),
        rest.to_string()
    );

    eprintln!(">>> descend_launch END proxy");
    output.parse().unwrap()
}

#[proc_macro]
pub fn _descend_launch_internal(input: TokenStream) -> TokenStream {
    eprintln!(">>> descend_launch_internal START");

    // 1. Extract the source string literal
    let mut iter = input.into_iter();
    let source_tt = iter.next().expect("Expected source string literal");
    let comma_tt = iter.next().expect("Expected comma");

    let source_str = match source_tt {
        proc_macro::TokenTree::Literal(lit) => {
            let s = lit.to_string();
            // Trim the r###" and "###
            s.strip_prefix("r###\"")
                .and_then(|s| s.strip_suffix("\"###"))
                .unwrap_or(&s)
                .to_string()
        }
        _ => panic!("Expected string literal for source"),
    };

    let rest: TokenStream = iter.collect();

    // 2. Parse LaunchConfig from the rest of the stream
    // (This is exactly what the old descend_launch did)
    // To use syn on `rest`, we can convert it to proc_macro2::TokenStream
    // Wait, syn::parse takes proc_macro::TokenStream natively
    let launch_config = match syn::parse::<crate::parser::LaunchConfig>(rest) {
        Ok(c) => c,
        Err(err) => {
            return err.to_compile_error().into();
        }
    };

    let kernel_name = &launch_config.kernel_name;
    let generic_to_size = launch_config.generic_to_size;
    let arrays = launch_config.array_args;

    // 3. Re-parse the source code to get the AST
    // Since we no longer use a global map, we re-parse the stashed source
    let source_code = descend::parser::SourceCode::new(source_str.clone());
    let compilation_unit = crate::parser::parse_descend_source(&source_code).unwrap();
    let mut kernel = Kernel::extract(&compilation_unit).unwrap();

    // Extract call args from source_str (which mutates it)
    let mut mutable_source_str = source_str.clone();
    let call_args = crate::parser::extract_call_args(&mut mutable_source_str);
    kernel.call_args = call_args;

    let full_descend = synthesize_main_program(&mutable_source_str, &kernel, &generic_to_size);

    eprintln!(">>> descend_launch compiling to ptx");
    let (ptx_lib_path_str, mangled_name) = match compile_to_ptx(&full_descend, kernel_name) {
        Ok(res) => res,
        Err(e) => {
            return format!("compile_error!({:?});", e).parse().unwrap();
        }
    };

    // Generate the launcher wrapper with the embedded PTX
    let output = generate_host_wrapper(
        &kernel,
        &ptx_lib_path_str,
        arrays,
        &mangled_name,
        &generic_to_size,
    );

    eprintln!(
        ">>> descend_launch END
OUTPUT:
{}",
        output
    );
    TokenStream::from(output)
}
