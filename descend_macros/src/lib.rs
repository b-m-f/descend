#![feature(proc_macro_span)]
extern crate proc_macro;

mod compiler;
mod host_api;
mod kernel_info;
mod main_workaround;
mod sanitizer;

use compiler::compile_to_ptx;
use host_api::generate_host_wrapper;
use kernel_info::KernelInfo;
use main_workaround::synthesize_main_program;
use proc_macro::TokenStream;
use sanitizer::extract_source_str;

#[proc_macro]
pub fn descend(input: TokenStream) -> TokenStream {
    // TODO: describe nightly extraction and fallback
    let source_str = extract_source_str(&input);

    // Parse the input string using descend's parser
    let source = descend::parser::SourceCode::new(source_str.clone());
    let compilation_unit = match descend::parser::parse(&source) {
        Ok(c) => c,
        Err(_) => {
            let msg = format!(
                "Failed to parse descend macro input. Ensure syntax is valid.\nSource string:\n{}",
                source_str
            );
            return TokenStream::from(
                syn::Error::new(proc_macro2::Span::call_site(), msg).to_compile_error(),
            );
        }
    };

    // TODO: describe the KernelInfo object
    // why is it needed? Kind of an IR
    let info = match KernelInfo::extract(&compilation_unit) {
        Ok(i) => i,
        Err(err) => {
            return TokenStream::from(
                syn::Error::new(proc_macro2::Span::call_site(), err).to_compile_error(),
            );
        }
    };

    // TODO: find a better way to achieve this
    // or describe why this workaround is necessary
    //
    // Synthesize full valid Descend program (workaround for ty_check requiring fn main)
    let full_descend = synthesize_main_program(&source_str, &info);
    //
    let code = format!("\"{}\"", full_descend);
    code.parse().expect("invalid rust syntax")

    // // Compile to PTX bytecode via compiler module
    // let ptx_lib_path_str = compile_to_ptx(&full_descend, &info.fn_name);
    //
    // // Generate the launcher wrapper with the embedded PTX
    // let output = generate_host_wrapper(&info, &ptx_lib_path_str);
    // TokenStream::from_str(full_descend).expect("invalid rust syntax")
}
