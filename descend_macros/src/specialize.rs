use crate::ir::Kernel;
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
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
) -> Result<String, String> {
    debug!(
        "[DEBUG] Starting generic substitution for kernel '{}'",
        kernel.function_name
    );
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
            actual_ty_str_cpu = crate::nat::substitute_word(&actual_ty_str_cpu, gen, size);
            actual_shared_ty = crate::nat::substitute_word(&actual_shared_ty, gen, size);
        }
        if param.is_shared_mem {
            shared_mem_configs.push(actual_shared_ty);
            continue;
        }

        if param.is_by_value {
            // The kernel takes the value itself, so the synthesized host
            // function takes it too and hands it over unchanged: there is
            // nothing to allocate or copy to the device.
            let mut value_ty = param.value_ty_str.clone();
            for (gen, size) in generic_to_size {
                value_ty = crate::nat::substitute_word(&value_ty, gen, size);
            }
            main_params.push(format!("arg{}: {}", i, value_ty));
            main_args.push(format!("arg{}", i));
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
        actual_grid_dim_str = crate::nat::substitute_word(&actual_grid_dim_str, gen, size);
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

    // Specialize the target kernel only. The block may hold other kernels, and
    // rewriting a size name inside one of them would change code the caller
    // never asked about — and the whole block is type-checked as one program.
    let specialized = specialize_kernel(source_ts, &kernel.function_name, generic_to_size)?;

    // Convert back to string for Descend compiler
    let full_descend = crate::descend_src::to_descend_string(specialized);

    Ok(format!("{}\n{}", main_fn, full_descend))
}

/// Replaces a kernel's `nat` generics with the literals bound for it, touching
/// nothing outside that kernel.
///
/// Strips the kernel's own `<N: nat, ..>` declarations, then rewrites `N` within
/// its signature and body. Sibling items in the same `descend!` block keep their
/// identifiers, and a kernel that binds a local named like one of its sizes is
/// rejected rather than silently rewritten.
fn specialize_kernel(
    stream: TokenStream,
    target: &str,
    generic_to_size: &HashMap<String, String>,
) -> Result<TokenStream, String> {
    let mut out = TokenStream::new();
    let mut iter = stream.into_iter().peekable();

    while let Some(tt) = iter.next() {
        let is_fn = matches!(&tt, TokenTree::Ident(ident) if ident == "fn");
        out.extend(std::iter::once(tt));
        if !is_fn {
            continue;
        }

        let Some(name_tt) = iter.next() else { break };
        let is_target = matches!(&name_tt, TokenTree::Ident(ident) if ident == target);
        out.extend(std::iter::once(name_tt));
        if !is_target {
            continue;
        }

        // The kernel's own generic declarations go away; any that the launch did
        // not bind stay, so the Descend compiler still reports them.
        if let Some(TokenTree::Punct(p)) = iter.peek() {
            if p.as_char() == '<' {
                let left_angle = iter.next().expect("peeked");
                let (inner_tokens, right_angle) = extract_generic_block(&mut iter);
                let keys: Vec<&str> = generic_to_size.keys().map(|k| k.as_str()).collect();
                let kept: Vec<TokenTree> =
                    filter_generics(inner_tokens, &keys).into_iter().collect();
                if !kept.is_empty() {
                    out.extend(std::iter::once(left_angle));
                    out.extend(kept);
                    out.extend(std::iter::once(right_angle));
                }
            }
        }

        // Everything up to and including the body block belongs to this kernel.
        let mut kernel_tokens = Vec::new();
        for tt in iter.by_ref() {
            let is_body = matches!(&tt, TokenTree::Group(g) if g.delimiter() == Delimiter::Brace);
            kernel_tokens.push(tt);
            if is_body {
                break;
            }
        }
        let kernel_ts: TokenStream = kernel_tokens.into_iter().collect();
        reject_shadowed_sizes(kernel_ts.clone(), target, generic_to_size)?;
        out.extend(substitute_generics_in_tokens(kernel_ts, generic_to_size));
    }

    Ok(out)
}

/// Refuses a kernel that binds a name it also uses as a size.
///
/// Substitution rewrites the size name everywhere it appears in the kernel, so a
/// local with that name would silently become an integer literal. Catching it
/// here turns a baffling Descend type error into a sentence naming the binding.
fn reject_shadowed_sizes(
    stream: TokenStream,
    kernel: &str,
    generic_to_size: &HashMap<String, String>,
) -> Result<(), String> {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();

    for (i, tt) in tokens.iter().enumerate() {
        if let TokenTree::Group(group) = tt {
            reject_shadowed_sizes(group.stream(), kernel, generic_to_size)?;
            continue;
        }
        let TokenTree::Ident(ident) = tt else {
            continue;
        };
        let name = ident.to_string();
        if !generic_to_size.contains_key(&name) {
            continue;
        }

        let previous = match i.checked_sub(1).and_then(|prev| tokens.get(prev)) {
            Some(TokenTree::Ident(ident)) => ident.to_string(),
            _ => String::new(),
        };
        // `x =>` introduces the name of an `indep` arm.
        let introduces_arm = matches!(
            tokens.get(i + 1),
            Some(TokenTree::Punct(p)) if p.as_char() == '='
        ) && matches!(
            tokens.get(i + 2),
            Some(TokenTree::Punct(p)) if p.as_char() == '>'
        );

        if matches!(previous.as_str(), "let" | "mut" | "sched" | "for") || introduces_arm {
            return Err(format!(
                "kernel `{kernel}` binds `{name}`, which is also one of its sizes; rename the \
                 binding, or the size would be replaced by its value here"
            ));
        }
    }

    Ok(())
}

/// Resolves generic dimension tags deeply nested inside source tokens.
///
/// What it does: Crawls structural `Group` variants directly mutating matched identifiers to the literal bounds mapped within `generic_to_size`.
/// Why it is needed: Replaces procedural variable bounds securely without corrupting spacing metadata or breaking string limits enforced by parsing frameworks.
fn substitute_generics_in_tokens(
    stream: TokenStream,
    generic_to_size: &HashMap<String, String>,
) -> TokenStream {
    let mut out = TokenStream::new();
    // A name after a single `.` is a field or a method, never a size. Two dots
    // are a range — `0..n` does bind the size `n` — so only a lone dot shields
    // the name that follows it.
    let mut previous_is_dot = false;
    let mut penultimate_is_dot = false;
    for tt in stream {
        let is_dot = matches!(&tt, TokenTree::Punct(p) if p.as_char() == '.');
        let follows_dot = previous_is_dot && !penultimate_is_dot;
        penultimate_is_dot = previous_is_dot;
        previous_is_dot = is_dot;
        match tt {
            TokenTree::Group(g) => {
                let mut new_g = Group::new(
                    g.delimiter(),
                    substitute_generics_in_tokens(g.stream(), generic_to_size),
                );
                new_g.set_span(g.span());
                out.extend(std::iter::once(TokenTree::Group(new_g)));
            }
            TokenTree::Ident(i) => {
                let name = i.to_string();
                if let Some(size) = generic_to_size.get(&name).filter(|_| !follows_dot) {
                    debug!(
                        "[DEBUG] Applying generic substitution: '{}' -> '{}'",
                        name, size
                    );
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

fn extract_generic_block(
    iter: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
) -> (Vec<TokenTree>, TokenTree) {
    let mut inner_tokens = Vec::new();
    let mut depth = 1;
    let mut right_angle = None;

    for tt in iter.by_ref() {
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

    (
        inner_tokens,
        right_angle.unwrap_or_else(|| {
            TokenTree::Punct(proc_macro2::Punct::new('>', proc_macro2::Spacing::Alone))
        }),
    )
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
                debug!(
                    "[DEBUG] CHECKING KEY: {}, in keys: {:?}",
                    i.to_string(),
                    keys
                );
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
            out.extend(std::iter::once(TokenTree::Punct(proc_macro2::Punct::new(
                ',',
                proc_macro2::Spacing::Alone,
            ))));
        }
        for tt in part {
            out.extend(std::iter::once(tt));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn sizes(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    /// The packed source with all whitespace removed, since the formatter's
    /// spacing is not what these tests are about.
    fn specialize(source: &str, target: &str, pairs: &[(&str, &str)]) -> Result<String, String> {
        let stream = TokenStream::from_str(source).expect("test input must tokenize");
        let packed = crate::descend_src::to_descend_string(specialize_kernel(
            stream,
            target,
            &sizes(pairs),
        )?);
        Ok(packed.split_whitespace().collect())
    }

    #[test]
    fn the_targets_sizes_become_literals() {
        let out = specialize(
            "fn gpu_k<N: nat, r: prv>(a: &r uniq gpu.global [i32; N]) { let x = N; }",
            "gpu_k",
            &[("N", "16384")],
        )
        .unwrap();
        assert!(out.contains("[i32;16384]"), "{out}");
        assert!(out.contains("letx=16384"), "{out}");
        assert!(
            !out.contains("N:nat"),
            "the declaration should be gone: {out}"
        );
    }

    #[test]
    fn a_sibling_kernel_is_left_alone() {
        // The whole block is type-checked as one program, so rewriting `N` in a
        // kernel nobody asked about could break it.
        let out = specialize(
            "fn gpu_k<N: nat>(a: &r uniq gpu.global [i32; N]) { let x = N; }
             fn other<N: nat>(b: &r uniq gpu.global [i32; N]) { let N = 1; }",
            "gpu_k",
            &[("N", "16384")],
        )
        .unwrap();
        assert!(
            out.contains("[i32;16384]"),
            "the target was not specialized: {out}"
        );
        assert!(
            out.contains("fnother<N:nat>"),
            "the sibling's generics were touched: {out}"
        );
        assert!(
            out.contains("letN=1"),
            "the sibling's body was touched: {out}"
        );
    }

    #[test]
    fn a_local_shadowing_a_size_is_rejected() {
        let error = specialize(
            "fn gpu_k<N: nat>(a: &r uniq gpu.global [i32; N]) { let N = 3; }",
            "gpu_k",
            &[("N", "16384")],
        )
        .unwrap_err();
        assert!(
            error.contains("gpu_k") && error.contains("binds `N`"),
            "unhelpful error: {error}"
        );
    }

    #[test]
    fn a_scheduling_binder_shadowing_a_size_is_rejected() {
        let error = specialize(
            "fn gpu_k<block: nat>(a: &r uniq gpu.global [i32; block]) { sched block in grid { } }",
            "gpu_k",
            &[("block", "64")],
        )
        .unwrap_err();
        assert!(error.contains("binds `block`"), "unhelpful error: {error}");
    }

    #[test]
    fn a_field_access_is_not_a_size_but_a_range_bound_is() {
        let out = specialize(
            "fn gpu_k<n: nat>(a: &r uniq gpu.global [i32; n]) { let y = p.n; for i in 0..n { } }",
            "gpu_k",
            &[("n", "128")],
        )
        .unwrap();
        assert!(out.contains("p.n"), "a field name must survive: {out}");
        assert!(out.contains("0..128"), "a range bound is a size: {out}");
    }
}
