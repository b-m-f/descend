extern crate proc_macro;

mod compiler;
#[macro_use]
mod helper;
mod host_api;
mod kernel_ir;
mod main_workaround;
mod parser;
mod format;

use compiler::compile_to_ptx;
use host_api::generate_host_wrapper;
use kernel_ir::Kernel;
use main_workaround::synthesize_main_program;
use proc_macro::{TokenStream, TokenTree, Ident, Punct, Spacing, Group, Delimiter};
use std::str::FromStr;

/// The primary entry point for embedding Descend code in Rust.
/// 
/// What it does: Extracts all kernel function definitions and generates corresponding `macro_rules!` wrappers 
/// to allow them to be executed seamlessly via `descend_launch!`.
/// Why it is needed: Users invoke `descend! { ... }` with arbitrary Descend code (structs, functions, kernels).
/// We must bind macro entries for every single function present in the supplied code block.
#[proc_macro]
pub fn descend(input: TokenStream) -> TokenStream {
    let source_ts = input.clone();
    
    // We format the raw tokens tightly (removing syntax-invalid spaces) 
    // to strictly satisfy the Descend PEG parser requirements.
    let source_str = crate::format::to_descend_string(source_ts.clone());
    let source = descend::parser::SourceCode::new(source_str.clone());
    let compilation_unit = match crate::parser::parse_descend_source(&source) {
        Ok(c) => c,
        Err(err) => {
            return std::format!("compile_error!({:?});", err).parse().unwrap();
        }
    };

    let mut output = TokenStream::new();
    
    // Support arbitrary numbers of kernels within a single block
    for item in &compilation_unit.items {
        if let descend::ast::Item::FunDef(f) = item {
            let kernel_name = f.ident.name.to_string();
            
            // macro_rules! kernel_name
            output.extend(vec![
                TokenTree::Ident(Ident::new("macro_rules", proc_macro::Span::call_site())),
                TokenTree::Punct(Punct::new('!', Spacing::Alone)),
                TokenTree::Ident(Ident::new(&kernel_name, proc_macro::Span::call_site())),
            ]);
            
            let mut rule_match = TokenStream::new();
            rule_match.extend(vec![
                TokenTree::Punct(Punct::new('$', Spacing::Alone)),
                TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::from_str("$args:tt").unwrap())),
                TokenTree::Punct(Punct::new('*', Spacing::Alone)),
            ]);
            
            let mut macro_body = TokenStream::new();
            macro_body.extend(TokenStream::from_str("::descend_macros::_descend_launch_internal!").unwrap());
            
            let mut launch_args = TokenStream::new();
            // Block containing entire original source tokens
            launch_args.extend(vec![
                TokenTree::Group(Group::new(Delimiter::Brace, source_ts.clone())),
                TokenTree::Punct(Punct::new(',', Spacing::Alone)),
                TokenTree::Ident(Ident::new(&kernel_name, proc_macro::Span::call_site())),
                TokenTree::Punct(Punct::new(',', Spacing::Alone)),
            ]);
            launch_args.extend(TokenStream::from_str("$($args)*").unwrap());
            
            macro_body.extend(vec![
                TokenTree::Group(Group::new(Delimiter::Parenthesis, launch_args)),
            ]);
            
            let mut rule_body = TokenStream::new();
            rule_body.extend(vec![
                TokenTree::Group(Group::new(Delimiter::Parenthesis, rule_match)),
                TokenTree::Punct(Punct::new('=', Spacing::Joint)),
                TokenTree::Punct(Punct::new('>', Spacing::Alone)),
                TokenTree::Group(Group::new(Delimiter::Brace, macro_body)),
            ]);
            
            output.extend(vec![
                TokenTree::Group(Group::new(Delimiter::Brace, rule_body)),
            ]);
        }
    }
    
    output
}

/// The invocation layer for triggering a GPU kernel.
/// 
/// What it does: Converts a call like `descend_launch!(my_kernel, ...)` into `my_kernel!(...)`.
/// Why it is needed: Resolves to the generated `macro_rules!` wrapper that ultimately triggers NVCC compilation.
#[proc_macro]
pub fn descend_launch(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();

    let kernel_ident = match iter.next() {
        Some(TokenTree::Ident(id)) => id,
        _ => return "compile_error!(\"Expected kernel name identifier in descend_launch!\");".parse().unwrap(),
    };
    
    match iter.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == ',' => {}
        _ => return "compile_error!(\"Expected comma after kernel name in descend_launch!\");".parse().unwrap(),
    }

    let rest: TokenStream = iter.collect();
    let mut output = TokenStream::new();
    output.extend(vec![
        TokenTree::Ident(kernel_ident),
        TokenTree::Punct(Punct::new('!', Spacing::Alone)),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, rest)),
    ]);
    
    output
}

/// Internal pipeline connecting the caller's arguments to the JIT NVCC compiler and rust wrapper.
/// 
/// What it does: Orchestrates the sequential 5 phases of compilation requested:
/// Parse, Evaluate to Static Values, Type Check, Compile, Extract & Embed.
/// Why it is needed: This is the internal macro dispatched by the individual `macro_rules!` wrappers
/// that has access to both the target function's name and the entire Descend block token stream.
#[proc_macro]
pub fn _descend_launch_internal(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    
    let source_tt = iter.next().expect("Expected source group");
    let mut source_ts = match source_tt {
        TokenTree::Group(g) => g.stream(),
        _ => panic!("Expected block for source"),
    };
    
    let _comma_tt = iter.next().expect("Expected comma");
    let target_name_tt = iter.next().expect("Expected kernel name string");
    let target_name = target_name_tt.to_string();
    let _comma_tt = iter.next().expect("Expected comma");

    let rest: TokenStream = iter.collect();

    // Step 1: Parse
    let (kernel, generic_to_size, array_args) = step_1_parse_and_extract_ir(rest, &mut source_ts, &target_name);
    
    // Step 2: Evaluate to Static Values
    let full_descend = step_2_evaluate_to_static_values(source_ts, &kernel, &generic_to_size);

    // Steps 3 & 4: Type Check and Compile
    let (ptx_content, mangled_name) = step_3_and_4_type_check_and_compile(&full_descend, &kernel.function_name);

    // Step 5: Extract and Embed
    step_5_extract_and_embed(&kernel, &ptx_content, array_args, &mangled_name, &generic_to_size)
}

/// Step 1: Parse and Extract Kernel IR
/// 
/// What it does: Parses host-provided generic arguments and transforms the raw token stream into a 
/// strongly-typed Kernel IR mapped to exactly the function being launched.
/// Why it is needed: It prepares the structured definitions and types necessary to bridge the Descend 
/// compilation environment with the runtime host parameters.
fn step_1_parse_and_extract_ir(
    rest_args: TokenStream,
    source_ts: &mut TokenStream,
    target_name: &str
) -> (Kernel, std::collections::HashMap<String, String>, Vec<TokenStream>) {
    let launch_config = crate::parser::LaunchConfig::parse(rest_args)
        .unwrap_or_else(|err| panic!("Launch config error: {:?}", err));

    let call_args = crate::parser::extract_call_args_tokens(source_ts);
    let source_str = crate::format::to_descend_string(source_ts.clone());
    
    let source_code = descend::parser::SourceCode::new(source_str);
    let compilation_unit = crate::parser::parse_descend_source(&source_code).expect("Failed to parse descend source");
    
    let mut kernel = Kernel::extract(&compilation_unit, target_name).expect("Kernel extraction failed");
    kernel.call_args = call_args;

    (kernel, launch_config.generic_to_size, launch_config.array_args)
}

/// Step 2: Evaluate to Static Values
/// 
/// What it does: Replaces procedural macro parameter bounds in the raw stream with user-supplied integers.
/// Why it is needed: Required because Descend's GPU target constraints rely on static allocations and 
/// fully evaluated lengths for both compile-time correctness and CUDA runtime boundaries.
fn step_2_evaluate_to_static_values(
    source_ts: TokenStream,
    kernel: &Kernel,
    generic_to_size: &std::collections::HashMap<String, String>
) -> String {
    synthesize_main_program(source_ts, kernel, generic_to_size)
}

/// Step 3 & 4: Type Check and Compile
/// 
/// What it does: Runs the embedded Descend code natively through its AST type checker, then outsources code generation to NVCC.
/// Why it is needed: Ensures borrow/lifetime safety constraints hold for GPU code prior to letting the standard C++ compilation step 
/// build the raw PTX byte array.
fn step_3_and_4_type_check_and_compile(full_descend: &str, function_name: &str) -> (String, String) {
    compile_to_ptx(full_descend, function_name)
        .unwrap_or_else(|e| panic!("Compile to PTX error: {:?}", e))
}

/// Step 5: Extract and Embed
/// 
/// What it does: Generates the Rust bindings necessary to allocate host pointers into CUDA slices.
/// Why it is needed: It dynamically compiles the bytecode transparently and creates an interface function enabling 
/// standard rust code to cleanly call the complex GPU kernels under the hood.
fn step_5_extract_and_embed(
    kernel: &Kernel,
    ptx_content: &str,
    array_args: Vec<TokenStream>,
    mangled_name: &str,
    generic_to_size: &std::collections::HashMap<String, String>
) -> TokenStream {
    generate_host_wrapper(
        kernel,
        ptx_content,
        array_args,
        mangled_name,
        generic_to_size,
    )
}
