
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

#[proc_macro]
pub fn descend_kernel(input: TokenStream) -> TokenStream {
    let mut source_ts = input.clone();
    let call_args = crate::parser::extract_call_args_tokens(&mut source_ts);
    let source_str = crate::format::to_descend_string(source_ts.clone());

    let source = descend::parser::SourceCode::new(source_str.clone());
    let compilation_unit = match crate::parser::parse_descend_source(&source) {
        Ok(c) => c,
        Err(err) => {
            return std::format!("compile_error!({:?});", err).parse().unwrap();
        }
    };

    let mut kernel = match Kernel::extract(&compilation_unit) {
        Ok(i) => i,
        Err(err) => {
            return std::format!("compile_error!({:?});", err).parse().unwrap();
        }
    };
    kernel.call_args = call_args;

    let kernel_name = kernel.function_name.clone();

    let mut output = TokenStream::new();
    
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
    launch_args.extend(vec![
        TokenTree::Group(Group::new(Delimiter::Brace, source_ts)),
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
    
    output
}

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
    eprintln!("REST TOKENS: {}", rest.to_string());
    let mut output = TokenStream::new();
    output.extend(vec![
        TokenTree::Ident(kernel_ident),
        TokenTree::Punct(Punct::new('!', Spacing::Alone)),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, rest)),
    ]);
    
    output
}

#[proc_macro]
pub fn _descend_launch_internal(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    
    let source_tt = iter.next().expect("Expected source group");
    let source_ts = match source_tt {
        TokenTree::Group(g) => g.stream(),
        _ => panic!("Expected block for source"),
    };
    
    let _comma_tt = iter.next().expect("Expected comma");
    let rest: TokenStream = iter.collect();
    eprintln!("REST TOKENS: {}", rest.to_string());

    let launch_config = match crate::parser::LaunchConfig::parse(rest) {
        Ok(c) => c,
        Err(err) => {
            return std::format!("compile_error!({:?});", err).parse().unwrap();
        }
    };

    let generic_to_size = launch_config.generic_to_size;
    eprintln!("generic_to_size: {:?}", generic_to_size);
    let arrays = launch_config.array_args;

    let mut mutable_source_ts = source_ts.clone();
    let call_args = crate::parser::extract_call_args_tokens(&mut mutable_source_ts);
    
    let source_str = crate::format::to_descend_string(mutable_source_ts.clone());
    let source_code = descend::parser::SourceCode::new(source_str.clone());
    let compilation_unit = crate::parser::parse_descend_source(&source_code).unwrap();
    let mut kernel = Kernel::extract(&compilation_unit).unwrap();
    kernel.call_args = call_args;

    let full_descend = synthesize_main_program(mutable_source_ts, &kernel, &generic_to_size);

    let kernel_name = kernel.function_name.clone();
    let (ptx_content, mangled_name) = match compile_to_ptx(&full_descend, &kernel_name) {
        Ok(res) => res,
        Err(e) => {
            return std::format!("compile_error!({:?});", e).parse().unwrap();
        }
    };

    generate_host_wrapper(
        &kernel,
        &ptx_content,
        arrays,
        &mangled_name,
        &generic_to_size,
    )
}
