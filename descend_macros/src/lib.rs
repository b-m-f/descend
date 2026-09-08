extern crate proc_macro;

#[macro_use]
mod helper;

mod compiler;
mod format;
mod host_api;
mod kernel_ir;
mod main_workaround;
mod parser;

use compiler::compile_to_ptx;
use host_api::generate_host_wrapper;
use kernel_ir::Kernel;
use main_workaround::synthesize_main_program;
use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

/// Recursively traverses a `TokenStream` to replace a specific placeholder identifier with the user's pristine code.
///
/// What it does:
/// - `TokenTree::Group` match: Recursively descends into nested brackets/braces (like `{ ... }` or `( ... )`). `TokenStream`s are tree structures, not flat arrays. Without this recursion, we would fail to find placeholders hidden inside the macro's inner blocks.
/// - `TokenTree::Ident` match: Looks for the exact literal word `_SOURCE_PLACEHOLDER_` and swaps it out for the user's untouched `TokenStream` (wrapped in braces).
///
/// Why it is needed: This technique allows us to use highly readable string templates (e.g. `format!("macro_rules! ...")`) for boilerplate macro generation, while cleanly injecting the user's original `TokenStream` without stringifying it. This perfectly preserves compiler span hygiene so `rustc` errors point directly to the user's specific faulty syntax rather than highlighting the entire macro block.
fn substitute_source_placeholder(ts: TokenStream, source_ts: &TokenStream) -> TokenStream {
    let mut out = TokenStream::new();
    for tt in ts {
        match tt {
            TokenTree::Group(g) => {
                let new_stream = substitute_source_placeholder(g.stream(), source_ts);
                out.extend(std::iter::once(TokenTree::Group(Group::new(
                    g.delimiter(),
                    new_stream,
                ))));
            }
            TokenTree::Ident(id) if id.to_string() == "_SOURCE_PLACEHOLDER_" => {
                out.extend(std::iter::once(TokenTree::Group(Group::new(
                    Delimiter::Brace,
                    source_ts.clone(),
                ))));
            }
            other => {
                out.extend(std::iter::once(other));
            }
        }
    }
    out
}

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
    debug!("MACRO INPUT: {}", source_str);
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

            let template = format!(
                "macro_rules! {0} {{ ($($args:tt)*) => {{ ::descend_macros::_descend_launch_internal!( _SOURCE_PLACEHOLDER_, {0}, $($args)* ) }} }}", 
                kernel_name
            );

            let template_ts = template.parse::<TokenStream>().unwrap();
            let rule_ts = substitute_source_placeholder(template_ts, &source_ts);

            output.extend(rule_ts);
        } else if let descend::ast::Item::StructDecl(s) = item {
            let struct_code = crate::parser::format_rust_struct(s);
            output.extend(struct_code.parse::<TokenStream>().unwrap());
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
        _ => {
            return "compile_error!(\"Expected kernel name identifier in descend_launch!\");"
                .parse()
                .unwrap()
        }
    };

    match iter.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == ',' => {}
        _ => {
            return "compile_error!(\"Expected comma after kernel name in descend_launch!\");"
                .parse()
                .unwrap()
        }
    }

    let rest: TokenStream = iter.collect();

    let mut output = format!("{}!", kernel_ident).parse::<TokenStream>().unwrap();
    output.extend(std::iter::once(TokenTree::Group(Group::new(
        Delimiter::Parenthesis,
        rest,
    ))));

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
    let (kernel, stream_expr, generic_to_size, array_args) =
        step_1_parse_and_extract_ir(rest, &mut source_ts, &target_name);

    // Step 2: Evaluate to Static Values
    let full_descend = step_2_evaluate_to_static_values(source_ts, &kernel, &generic_to_size);

    // Steps 3 & 4: Type Check and Compile
    let (ptx_content, mangled_name) =
        step_3_and_4_type_check_and_compile(&full_descend, &kernel.function_name);

    // Step 5: Extract and Embed
    step_5_extract_and_embed(
        &kernel,
        &ptx_content,
        stream_expr,
        array_args,
        &mangled_name,
        &generic_to_size,
    )
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
    target_name: &str,
) -> (
    Kernel,
    TokenStream,
    std::collections::HashMap<String, String>,
    Vec<TokenStream>,
) {
    let launch_config = crate::parser::LaunchConfig::parse(rest_args)
        .unwrap_or_else(|err| panic!("Launch config error: {:?}", err));

    let call_args = crate::parser::extract_call_args_tokens(source_ts);
    let source_str = crate::format::to_descend_string(source_ts.clone());
    debug!("MACRO INPUT: {}", source_str);

    let source_code = descend::parser::SourceCode::new(source_str);
    let compilation_unit =
        crate::parser::parse_descend_source(&source_code).expect("Failed to parse descend source");

    let mut kernel =
        Kernel::extract(&compilation_unit, target_name).expect("Kernel extraction failed");
    kernel.call_args = call_args;

    (
        kernel,
        launch_config.stream_expr,
        launch_config.generic_to_size,
        launch_config.array_args,
    )
}

/// Step 2: Evaluate to Static Values
///
/// What it does: Replaces procedural macro parameter bounds in the raw stream with user-supplied integers.
/// Why it is needed: Required because Descend's GPU target constraints rely on static allocations and
/// fully evaluated lengths for both compile-time correctness and CUDA runtime boundaries.
fn step_2_evaluate_to_static_values(
    source_ts: TokenStream,
    kernel: &Kernel,
    generic_to_size: &std::collections::HashMap<String, String>,
) -> String {
    synthesize_main_program(source_ts, kernel, generic_to_size)
}

/// Step 3 & 4: Type Check and Compile
///
/// What it does: Runs the embedded Descend code natively through its AST type checker, then outsources code generation to NVCC.
/// Why it is needed: Ensures borrow/lifetime safety constraints hold for GPU code prior to letting the standard C++ compilation step
/// build the raw PTX byte array.
fn step_3_and_4_type_check_and_compile(
    full_descend: &str,
    function_name: &str,
) -> (String, String) {
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
    stream_expr: TokenStream,
    array_args: Vec<TokenStream>,
    mangled_name: &str,
    generic_to_size: &std::collections::HashMap<String, String>,
) -> TokenStream {
    generate_host_wrapper(
        kernel,
        ptx_content,
        stream_expr,
        array_args,
        mangled_name,
        generic_to_size,
    )
}
