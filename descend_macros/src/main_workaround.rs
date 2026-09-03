use crate::kernel_ir::Kernel;
use proc_macro::{TokenStream, TokenTree, Group};
use std::collections::HashMap;

/// Synthesizes a valid full Descend program by appending a dummy `main` function wrapper.
/// Produces a synthesized main execution unit resolving missing elements for compiler compatibility.
/// 
/// What it does: Crawls through the AST token streams swapping abstract bound markers for statically-resolved concrete values passed during macro expansion, emitting a pseudo-main function.
/// Why it is needed: Binds runtime host inputs and constraints into the strict type-driven context required to generate functional Descend PTX C++ sources.
pub fn synthesize_main_program(
    source_ts: TokenStream,
    kernel: &Kernel,
    generic_to_size: &HashMap<String, String>,
) -> String {
    debug!("[DEBUG] Starting generic substitution for kernel '{}'", kernel.function_name);
    for (gen, size) in generic_to_size {
        debug!("  - Registered substitution: {} = {}", gen, size);
    }

    let mut main_params = Vec::new();
    let mut main_vars = Vec::new();
    let mut main_args = Vec::new();
    let mut shared_mem_configs = Vec::new();

    for (i, param) in kernel.params.iter().enumerate() {
        let mem_qual = if param.is_uniq { "uniq" } else { "shrd" };
        let mut actual_ty_str_cpu = param.cpu_ty_str.clone();
        let mut actual_shared_ty = param.shared_ty_str.clone();
        for (gen, size) in generic_to_size {
            actual_ty_str_cpu = crate::host_api::substitute_word(&actual_ty_str_cpu, gen, size);
            actual_shared_ty = crate::host_api::substitute_word(&actual_shared_ty, gen, size);
        }
        if param.is_shared_mem {
            shared_mem_configs.push(actual_shared_ty);
            continue;
        }
        let param_str = format!("arg{}: {}", i, actual_ty_str_cpu);
        main_params.push(param_str);

        let var_str = format!(
            "let mut arg{}_gpu = gpu_alloc_copy(&uniq gpu, &shrd *arg{});",
            i, i
        );
        main_vars.push(var_str);

        let arg_str = format!("&{} arg{}_gpu", mem_qual, i);
        main_args.push(arg_str);
    }

    let mut actual_grid_dim_str = kernel.grid_dim_str.clone();
    for (gen, size) in generic_to_size {
        actual_grid_dim_str = crate::host_api::substitute_word(&actual_grid_dim_str, gen, size);
    }

    if !shared_mem_configs.is_empty() {
        actual_grid_dim_str.push_str("; ");
        actual_grid_dim_str.push_str(&shared_mem_configs.join(", "));
    }

    let main_fn = format!(
        "
fn main({}) -[t: cpu.thread]-> () <'a> {{
    let mut gpu = gpu_device(0);
    {}
    {}::<<<{}>>>({});
    ()
}}
",
        main_params.join(", "),
        main_vars.join("\n    "),
        kernel.function_name,
        actual_grid_dim_str,
        main_args.join(", ")
    );

    // Remove the generic `<N: nat, ...>` declarations from the function signature FIRST
    let stripped_ts = remove_generic_declarations(source_ts, generic_to_size.keys().map(|k| k.as_str()).collect());
    // Apply generic substitutions to the rest of the token stream
    let substituted_ts = substitute_generics_in_tokens(stripped_ts, generic_to_size);

    // Convert back to string for Descend compiler
    let full_descend = crate::format::to_descend_string(substituted_ts);

    format!("{}\n{}", main_fn, full_descend)
}

/// Resolves generic dimension tags deeply nested inside source tokens.
/// 
/// What it does: Crawls structural `Group` variants directly mutating matched identifiers to the literal bounds mapped within `generic_to_size`.
/// Why it is needed: Replaces procedural variable bounds securely without corrupting spacing metadata or breaking string limits enforced by parsing frameworks.
fn substitute_generics_in_tokens(stream: TokenStream, generic_to_size: &HashMap<String, String>) -> TokenStream {
    let mut out = TokenStream::new();
    for tt in stream {
        match tt {
            TokenTree::Group(g) => {
                let mut new_g = Group::new(g.delimiter(), substitute_generics_in_tokens(g.stream(), generic_to_size));
                new_g.set_span(g.span());
                out.extend(std::iter::once(TokenTree::Group(new_g)));
            }
            TokenTree::Ident(i) => {
                let name = i.to_string();
                if let Some(size) = generic_to_size.get(&name) {
                    debug!("[DEBUG] Applying generic substitution: '{}' -> '{}'", name, size);
                    if let Ok(ts) = size.parse::<TokenStream>() {
                        out.extend(ts);
                        continue;
                    }
                }
                out.extend(std::iter::once(TokenTree::Ident(i)));
            }
            _ => out.extend(std::iter::once(tt)),
        }
    }
    out
}

/// Isolates and filters the `<gs: nat>` declaration lists from function headers.
/// 
/// What it does: Drops generic brackets globally over matching function signatures within raw token lists.
/// Why it is needed: After generic substitution, residual declarations produce invalid code inside Descend's grammar. They must be removed prior to `parse_descend_source`.
fn remove_generic_declarations(stream: TokenStream, generic_keys: Vec<&str>) -> TokenStream {
    let mut out = TokenStream::new();
    let mut iter = stream.into_iter().peekable();
    
    while let Some(tt) = iter.next() {
        if let TokenTree::Group(g) = &tt {
            let mut new_g = Group::new(g.delimiter(), remove_generic_declarations(g.stream(), generic_keys.clone()));
            new_g.set_span(g.span());
            out.extend(std::iter::once(TokenTree::Group(new_g)));
            continue;
        }
        
        out.extend(std::iter::once(tt.clone()));

        if let TokenTree::Ident(ident) = &tt {
            if ident.to_string() == "fn" {
                // Try to safely consume the function name
                if let Some(name_tt) = iter.next() {
                    out.extend(std::iter::once(name_tt.clone()));
                    
                    // Check if the next token is '<'
                    if let Some(TokenTree::Punct(p)) = iter.peek() {
                        if p.as_char() == '<' {
                            let left_angle = iter.next().unwrap();
                            let (inner_tokens, right_angle) = extract_generic_block(&mut iter);
                            
                            let filtered = filter_generics(inner_tokens, &generic_keys);
                            let filtered_vec: Vec<_> = filtered.into_iter().collect();
                            if !filtered_vec.is_empty() {
                                out.extend(std::iter::once(left_angle));
                                for t in filtered_vec {
                                    out.extend(std::iter::once(t));
                                }
                                out.extend(std::iter::once(right_angle));
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn extract_generic_block(iter: &mut std::iter::Peekable<proc_macro::token_stream::IntoIter>) -> (Vec<TokenTree>, TokenTree) {
    let mut inner_tokens = Vec::new();
    let mut depth = 1;
    let mut right_angle = None;
    
    while let Some(tt) = iter.next() {
        if let TokenTree::Punct(ref inner_p) = tt {
            if inner_p.as_char() == '<' {
                depth += 1;
            } else if inner_p.as_char() == '>' {
                depth -= 1;
                if depth == 0 {
                    right_angle = Some(tt);
                    break;
                }
            }
        }
        inner_tokens.push(tt);
    }
    
    (inner_tokens, right_angle.unwrap_or_else(|| TokenTree::Punct(proc_macro::Punct::new('>', proc_macro::Spacing::Alone))))
}

fn filter_generics(tokens: Vec<TokenTree>, keys: &[&str]) -> TokenStream {
    let mut out = TokenStream::new();
    let mut parts = Vec::new();
    let mut current_part = Vec::new();
    
    for tt in tokens {
        if let TokenTree::Punct(ref p) = tt {
            if p.as_char() == ',' {
                parts.push(current_part);
                current_part = Vec::new();
                continue;
            }
        }
        current_part.push(tt);
    }
    if !current_part.is_empty() {
        parts.push(current_part);
    }
    
    let mut keep_parts = Vec::new();
    for part in parts {
        let mut keep = true;
        for tt in &part {
            if let TokenTree::Ident(ref i) = tt {
                debug!("[DEBUG] CHECKING KEY: {}, in keys: {:?}", i.to_string(), keys);
                if keys.contains(&i.to_string().as_str()) {
                    keep = false;
                }
                break; // only check the first identifier in the decl (e.g., `N` in `N: nat`)
            }
        }
        if keep {
            keep_parts.push(part);
        }
    }
    
    for (i, part) in keep_parts.into_iter().enumerate() {
        if i > 0 {
            out.extend(std::iter::once(TokenTree::Punct(proc_macro::Punct::new(',', proc_macro::Spacing::Alone))));
        }
        for tt in part {
            out.extend(std::iter::once(tt));
        }
    }
    
    out
}
