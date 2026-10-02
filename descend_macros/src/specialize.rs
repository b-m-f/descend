use crate::diag::MacroError;
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use std::collections::HashMap;

pub fn specialize_kernel(
    stream: TokenStream,
    target: &str,
    generic_to_size: &HashMap<String, String>,
) -> Result<TokenStream, MacroError> {
    let mut out = TokenStream::new();
    let mut iter = stream.into_iter().peekable();
    let mut found = false;

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
        found = true;

        let mut declared: Vec<String> = Vec::new();
        if matches!(iter.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '<') {
            let left_angle = iter.next().expect("peeked");
            let (inner_tokens, right_angle) = extract_generic_block(&mut iter);
            declared = nat_generic_names(&inner_tokens);
            let keys: Vec<&str> = generic_to_size.keys().map(|k| k.as_str()).collect();
            let kept: Vec<TokenTree> = filter_generics(inner_tokens, &keys).into_iter().collect();
            if !kept.is_empty() {
                out.extend(std::iter::once(left_angle));
                out.extend(kept);
                out.extend(std::iter::once(right_angle));
            }
        }
        check_size_bindings(&declared, target, generic_to_size)?;

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
        out.extend(substitute_generics_in_tokens(kernel_ts, generic_to_size)?);
    }

    if !found {
        return Err(format!("Kernel definition '{target}' not found in macro body").into());
    }

    Ok(out)
}

fn nat_generic_names(tokens: &[TokenTree]) -> Vec<String> {
    let mut names = Vec::new();

    for part in tokens.split(|tt| matches!(tt, TokenTree::Punct(p) if p.as_char() == ',')) {
        if let [TokenTree::Ident(name), TokenTree::Punct(colon), TokenTree::Ident(kind)] = part {
            if colon.as_char() == ':' && kind == "nat" {
                names.push(name.to_string());
            }
        }
    }

    names
}

fn check_size_bindings(
    sizes: &[String],
    kernel_name: &str,
    bindings: &HashMap<String, String>,
) -> Result<(), MacroError> {
    for size in sizes {
        if !bindings.contains_key(size) {
            return Err(format!(
                "instantiating kernel `{kernel_name}` needs a value for its size `{size}`"
            )
            .into());
        }
    }
    for bound in bindings.keys() {
        if !sizes.contains(bound) {
            return Err(
                format!("kernel `{kernel_name}` has no size `{bound}` to instantiate").into(),
            );
        }
    }

    Ok(())
}

fn reject_shadowed_sizes(
    stream: TokenStream,
    kernel: &str,
    generic_to_size: &HashMap<String, String>,
) -> Result<(), MacroError> {
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
        let introduces_arm = matches!(
            tokens.get(i + 1),
            Some(TokenTree::Punct(p)) if p.as_char() == '='
        ) && matches!(
            tokens.get(i + 2),
            Some(TokenTree::Punct(p)) if p.as_char() == '>'
        );

        if matches!(previous.as_str(), "let" | "mut" | "sched" | "for") || introduces_arm {
            return Err(MacroError::at(
                format!(
                    "kernel `{kernel}` binds `{name}`, which is also one of its sizes; rename the \
                     binding, or the size would be replaced by its value here"
                ),
                Some(ident.span()),
            ));
        }
    }

    Ok(())
}

fn substitute_generics_in_tokens(
    stream: TokenStream,
    generic_to_size: &HashMap<String, String>,
) -> Result<TokenStream, MacroError> {
    let mut out = TokenStream::new();
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
                    substitute_generics_in_tokens(g.stream(), generic_to_size)?,
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
                    let ts = size.parse::<TokenStream>().map_err(|error| {
                        MacroError::at(
                            format!(
                                "the value `{size}` bound to size `{name}` is not a valid size \
                                 expression: {error}"
                            ),
                            Some(i.span()),
                        )
                    })?;
                    out.extend(ts);
                    continue;
                }
                out.extend(std::iter::once(TokenTree::Ident(i)));
            }
            _ => out.extend(std::iter::once(tt)),
        }
    }
    Ok(out)
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
                break;
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

    fn specialize(
        source: &str,
        target: &str,
        pairs: &[(&str, &str)],
    ) -> Result<String, MacroError> {
        let stream = TokenStream::from_str(source).expect("test input must tokenize");
        let packed = crate::source::DescendSource::from_token_stream(specialize_kernel(
            stream,
            target,
            &sizes(pairs),
        )?)
        .text;
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
        .unwrap_err()
        .message;
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
        .unwrap_err()
        .message;
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
