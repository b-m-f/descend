use crate::kernel_ir::Kernel;
use std::collections::HashMap;

/// Synthesizes a valid full Descend program by appending a dummy `main` function wrapper.
/// This satisfies `descend::ty_check`, which currently hardcodes a requirement for a `main` entry point.
pub fn synthesize_main_program(source_str: &str, kernel: &Kernel, generic_to_size: &HashMap<String, String>) -> String {
    let mut main_params = Vec::new();
    let mut main_vars = Vec::new();
    let mut main_args = Vec::new();

    for (i, param) in kernel.params.iter().enumerate() {
        let mem_qual = if param.is_uniq { "uniq" } else { "shrd" };
        
        let mut actual_size_str = param.size_str.clone();
        for (gen, _) in &kernel.generic_size_bindings {
            if let Some(size) = generic_to_size.get(gen) {
                actual_size_str = actual_size_str.replace(gen, size);
            }
        }
        
        let param_str = format!("arg{}: &'a {} cpu.mem [i32; {}]", i, mem_qual, actual_size_str);
        main_params.push(param_str);

        let var_str = format!("let mut arg{}_gpu = gpu_alloc_copy(&uniq gpu, &shrd *arg{});", i, i);
        main_vars.push(var_str);

        let arg_str = format!("&{} arg{}_gpu", mem_qual, i);
        main_args.push(arg_str);
    }
    
    let mut actual_generic_params_str = kernel.generic_params_str.clone();
    for (gen, _) in &kernel.generic_size_bindings {
        if let Some(size) = generic_to_size.get(gen) {
            actual_generic_params_str = actual_generic_params_str.replace(gen, size);
        }
    }

    let mut full_descend = source_str.to_string();

    for (gen, _) in &kernel.generic_size_bindings {
        if let Some(size) = generic_to_size.get(gen) {
            // Remove the generic declaration robustly handling spaces e.g. "N : nat ,"
            full_descend = remove_generic_decl(&full_descend, gen);
            
            // Note: Descend might have empty angle brackets if we remove all, e.g. `< r: prv>`.
            // Now, replace the generic variable everywhere else as a word boundary
            full_descend = replace_word(&full_descend, gen, size);
        }
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
        kernel.grid_dim_str,
        main_args.join(", ")
    );

    format!("{}\n{}", main_fn, full_descend)
}

pub fn replace_word(s: &str, word: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut last_end = 0;
    
    let bytes = s.as_bytes();
    let word_bytes = word.as_bytes();
    let mut i = 0;
    while i <= bytes.len().saturating_sub(word.len()) {
        if &bytes[i..i+word.len()] == word_bytes {
            let prev_ok = if i == 0 { true } else { !bytes[i-1].is_ascii_alphanumeric() && bytes[i-1] != b'_' };
            let next_idx = i + word.len();
            let next_ok = if next_idx == bytes.len() { true } else { !bytes[next_idx].is_ascii_alphanumeric() && bytes[next_idx] != b'_' };
            
            if prev_ok && next_ok {
                result.push_str(&s[last_end..i]);
                result.push_str(replacement);
                i += word.len();
                last_end = i;
                continue;
            }
        }
        i += 1;
    }
    result.push_str(&s[last_end..]);
    result
}

fn remove_generic_decl(s: &str, gen: &str) -> String {
    let mut chars = s.char_indices().peekable();
    let mut result = String::new();
    let gen_len = gen.len();
    
    while let Some((i, c)) = chars.next() {
        if s[i..].starts_with(gen) {
            let prev_char = if i == 0 { ' ' } else { s[..i].chars().last().unwrap() };
            if !prev_char.is_ascii_alphanumeric() && prev_char != '_' {
                let next_idx = i + gen_len;
                let next_char = if next_idx < s.len() { s[next_idx..].chars().next().unwrap() } else { ' ' };
                if !next_char.is_ascii_alphanumeric() && next_char != '_' {
                    let mut lookahead = chars.clone();
                    for _ in 1..gen.chars().count() { lookahead.next(); }
                    
                    let mut is_decl = false;
                    while let Some(&(_, lc)) = lookahead.peek() {
                        if lc.is_whitespace() { lookahead.next(); } else { break; }
                    }
                    if let Some((_, ':')) = lookahead.next() {
                        while let Some(&(_, lc)) = lookahead.peek() {
                            if lc.is_whitespace() { lookahead.next(); } else { break; }
                        }
                        if let Some((_, 'n')) = lookahead.next() {
                            if let Some((_, 'a')) = lookahead.next() {
                                if let Some((_, 't')) = lookahead.next() {
                                    is_decl = true;
                                    while let Some(&(_, lc)) = lookahead.peek() {
                                        if lc.is_whitespace() { lookahead.next(); } else { break; }
                                    }
                                    if let Some(&(_, ',')) = lookahead.peek() {
                                        lookahead.next();
                                    }
                                }
                            }
                        }
                    }
                    
                    if is_decl {
                        chars = lookahead;
                        continue;
                    }
                }
            }
        }
        result.push(c);
    }
    
    let mut final_res = String::new();
    let mut chars2 = result.char_indices().peekable();
    while let Some((_, c)) = chars2.next() {
        if c == ',' {
            let mut lookahead = chars2.clone();
            let mut is_bracket = false;
            while let Some(&(_, lc)) = lookahead.peek() {
                if lc.is_whitespace() { lookahead.next(); }
                else if lc == '>' { is_bracket = true; break; }
                else { break; }
            }
            if is_bracket { continue; }
        }
        if c == '<' {
            let mut lookahead = chars2.clone();
            let mut is_comma = false;
            while let Some(&(_, lc)) = lookahead.peek() {
                if lc.is_whitespace() { lookahead.next(); }
                else if lc == ',' { is_comma = true; lookahead.next(); break; }
                else { break; }
            }
            if is_comma {
                final_res.push('<');
                chars2 = lookahead;
                continue;
            }
        }
        final_res.push(c);
    }
    
    final_res
}
