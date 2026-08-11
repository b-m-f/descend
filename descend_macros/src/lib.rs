#![feature(proc_macro_span)]
extern crate proc_macro;

mod compiler;
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
pub fn descend(input: TokenStream) -> TokenStream {
    // TODO: describe nightly extraction and fallback
    let source_str = extract_source_str(&input);

    // Parse the input string using the parser module
    let source = descend::parser::SourceCode::new(source_str.clone());
    let compilation_unit = match crate::parser::parse_descend_source(&source) {
        Ok(c) => c,
        Err(err) => {
            // TODO: no unwrap
            return format!("compile_error!({:?});", err).parse().unwrap();
        }
    };

    // TODO: describe the KernelInfo object
    // why is it needed? Kind of an IR
    let kernel = match Kernel::extract(&compilation_unit) {
        Ok(i) => i,
        Err(err) => {
            // TODO: no unwrap
            return format!("compile_error!({:?});", err).parse().unwrap();
        }
    };
    // TODO: turn into debug
    println!("{}", kernel);

    // TODO: find a better way to achieve this
    // or describe why this workaround is necessary
    //
    // Synthesize full valid Descend program (workaround for ty_check requiring fn main)
    let full_descend = synthesize_main_program(&source_str, &kernel);
    //
    let code = format!("\"{}\"", full_descend);
    return code.parse().expect("invalid rust syntax");

    // // Compile to PTX bytecode via compiler module
    // let ptx_lib_path_str = compile_to_ptx(&full_descend, &info.fn_name);
    //
    // // Generate the launcher wrapper with the embedded PTX
    // let output = generate_host_wrapper(&info, &ptx_lib_path_str);
    // TokenStream::from_str(full_descend).expect("invalid rust syntax");
}
