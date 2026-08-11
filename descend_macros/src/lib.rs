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

use std::sync::Mutex;
use std::collections::HashMap;
use lazy_static::lazy_static;

lazy_static! {
    static ref KERNELS: Mutex<HashMap<String, Kernel>> = Mutex::new(HashMap::new());
    static ref KERNEL_SOURCES: Mutex<HashMap<String, String>> = Mutex::new(HashMap::new());
}

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

    KERNELS.lock().unwrap_or_else(|e| e.into_inner()).insert(kernel.function_name.clone(), kernel.clone());
    KERNEL_SOURCES.lock().unwrap_or_else(|e| e.into_inner()).insert(kernel.function_name.clone(), source_str.clone());

    eprintln!(">>> descend_kernel END");
    // Return a dummy string literal for the kernel definition since it's just meant for registration
    return "\"kernel registered\"".parse().unwrap();
}

#[proc_macro]
pub fn descend_launch(input: TokenStream) -> TokenStream {
    eprintln!(">>> descend_launch START");
    let input_str = input.to_string();

    let parts: Vec<&str> = input_str.split(',').collect();
    if parts.is_empty() {
        return "compile_error!(\"descend_launch! expects kernel name and arguments\");"
            .parse()
            .unwrap();
    }

    let kernel_part = parts[0].trim();
    let mut kernel_name = kernel_part;
    let mut generic_to_size = HashMap::new();

    if let Some(start_idx) = kernel_part.find('<') {
        if let Some(end_idx) = kernel_part.rfind('>') {
            kernel_name = &kernel_part[0..start_idx].trim();
            let generics_str = &kernel_part[start_idx + 1..end_idx];
            let gen_parts: Vec<&str> = generics_str.split(',').collect();
            for gen_part in gen_parts {
                let kv: Vec<&str> = gen_part.split(':').collect();
                if kv.len() == 2 {
                    let k = kv[0].trim().to_string();
                    let v = kv[1].trim().to_string();
                    generic_to_size.insert(k, v);
                } else {
                    let kv: Vec<&str> = gen_part.split('=').collect();
                    if kv.len() == 2 {
                        let k = kv[0].trim().to_string();
                        let v = kv[1].trim().to_string();
                        generic_to_size.insert(k, v);
                    }
                }
            }
        }
    }

    let kernel_lock = KERNELS.lock().unwrap_or_else(|e| e.into_inner());
    let kernel = match kernel_lock.get(kernel_name) {
        Some(k) => k,
        None => {
            if kernel_lock.is_empty() {
                // If no kernels are registered at all, this is likely rust-analyzer evaluating 
                // macros out of order. We return a runtime panic to avoid a false-positive red squiggly.
                return format!("{{ panic!(\"Kernel '{}' not found. Ensure it was defined with descend_kernel! earlier in the same file.\"); }}", kernel_name)
                    .parse()
                    .unwrap();
            } else {
                // If some kernels are registered but this one isn't, the user made a typo!
                let available: Vec<_> = kernel_lock.keys().collect();
                let err_msg = format!("Kernel '{}' not found. Available kernels: {:?}", kernel_name, available);
                return format!("compile_error!({:?});", err_msg)
                    .parse()
                    .unwrap();
            }
        }
    };

    let source_lock = KERNEL_SOURCES.lock().unwrap_or_else(|e| e.into_inner());
    let source_str = source_lock.get(kernel_name).expect("KERNEL_SOURCES missing");
    
    let mut args = Vec::new();

    for i in 1..parts.len() {
        let arg_str = parts[i].trim();
        args.push(arg_str.to_string());
    }

    let full_descend = synthesize_main_program(source_str, kernel, &generic_to_size);

    eprintln!(">>> descend_launch compiling to ptx");
    let (ptx_lib_path_str, mangled_name) = match compile_to_ptx(&full_descend, kernel_name) {
        Ok(res) => res,
        Err(e) => {
            return format!("compile_error!({:?});", e).parse().unwrap();
        }
    };
    
    // Generate the launcher wrapper with the embedded PTX
    let output = generate_host_wrapper(kernel, &ptx_lib_path_str, args, &mangled_name, &generic_to_size);
    
    eprintln!(">>> descend_launch END
OUTPUT:
{}", output);
    TokenStream::from(output)
}
