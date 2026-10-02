use crate::diag::MacroError;
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use std::collections::HashMap;

pub fn specialize_kernel(
    token_stream: TokenStream,
    kernel_name: &str,
    sizes: &HashMap<String, String>,
) -> Result<TokenStream, MacroError> {
    let mut output = TokenStream::new();
    let mut iter = token_stream.into_iter().peekable();
    let mut found = false;

    while let Some(token_tree) = iter.next() {
        let is_fn = matches!(&token_tree, TokenTree::Ident(ident) if ident == "fn");
        output.extend([token_tree]);
        if !is_fn {
            continue;
        }

        let Some(name_token_tree) = iter.next() else {
            break;
        };
        let requested = matches!(&name_token_tree, TokenTree::Ident(ident) if ident == kernel_name);
        output.extend([name_token_tree]);
        if !requested {
            continue;
        }
        found = true;

        let mut declared: Vec<String> = Vec::new();
        if matches!(iter.peek(), Some(TokenTree::Punct(punct)) if punct.as_char() == '<') {
            let left_angle = iter.next().expect("peeked");
            let opened_at = left_angle.span();
            let (inner_tokens, right_angle) = extract_generic_block(&mut iter);

            // Reported here rather than left to be invented: a closing `>` supplied by
            // the macro produces a token stream that looks well formed, and the failure
            // resurfaces as a parse error in Descend pointing at source the reader never
            // wrote.
            let Some(right_angle) = right_angle else {
                return Err(MacroError::at(
                    format!(
                        "kernel `{kernel_name}` has a generic parameter list that `<` opens and \
                         nothing closes"
                    ),
                    Some(opened_at),
                ));
            };

            declared = nat_generic_names(&inner_tokens);
            let keys: Vec<&str> = sizes.keys().map(|key| key.as_str()).collect();
            let kept: Vec<TokenTree> = filter_generics(inner_tokens, &keys).into_iter().collect();
            if !kept.is_empty() {
                output.extend([left_angle]);
                output.extend(kept);
                output.extend([right_angle]);
            }
        }
        check_size_bindings(&declared, kernel_name, sizes)?;

        let mut kernel_tokens = Vec::new();
        for token_tree in iter.by_ref() {
            let is_body = matches!(&token_tree, TokenTree::Group(group) if group.delimiter() == Delimiter::Brace);
            kernel_tokens.push(token_tree);
            if is_body {
                break;
            }
        }
        let kernel_token_stream: TokenStream = kernel_tokens.into_iter().collect();
        reject_shadowed_sizes(kernel_token_stream.clone(), kernel_name, sizes)?;
        output.extend(substitute_generics_in_tokens(kernel_token_stream, sizes)?);
    }

    if !found {
        return Err(crate::kernel::not_defined(kernel_name));
    }

    Ok(output)
}

fn nat_generic_names(tokens: &[TokenTree]) -> Vec<String> {
    let mut names = Vec::new();

    for part in tokens
        .split(|token_tree| matches!(token_tree, TokenTree::Punct(punct) if punct.as_char() == ','))
    {
        if let [TokenTree::Ident(name), TokenTree::Punct(colon), TokenTree::Ident(kind)] = part {
            if colon.as_char() == ':' && kind == "nat" {
                names.push(name.to_string());
            }
        }
    }

    names
}

fn check_size_bindings(
    declared_sizes: &[String],
    kernel_name: &str,
    sizes: &HashMap<String, String>,
) -> Result<(), MacroError> {
    for size in declared_sizes {
        if !sizes.contains_key(size) {
            return Err(format!(
                "instantiating kernel `{kernel_name}` needs a value for its size `{size}`"
            )
            .into());
        }
    }
    for bound in sizes.keys() {
        if !declared_sizes.contains(bound) {
            return Err(
                format!("kernel `{kernel_name}` has no size `{bound}` to instantiate").into(),
            );
        }
    }

    Ok(())
}

fn reject_shadowed_sizes(
    token_stream: TokenStream,
    kernel_name: &str,
    sizes: &HashMap<String, String>,
) -> Result<(), MacroError> {
    let tokens: Vec<TokenTree> = token_stream.into_iter().collect();

    for (index, token_tree) in tokens.iter().enumerate() {
        if let TokenTree::Group(group) = token_tree {
            reject_shadowed_sizes(group.stream(), kernel_name, sizes)?;
            continue;
        }
        let TokenTree::Ident(ident) = token_tree else {
            continue;
        };
        let name = ident.to_string();
        if !sizes.contains_key(&name) {
            continue;
        }

        let previous = match index
            .checked_sub(1)
            .and_then(|previous_index| tokens.get(previous_index))
        {
            Some(TokenTree::Ident(ident)) => ident.to_string(),
            _ => String::new(),
        };
        let introduces_arm = matches!(
            tokens.get(index + 1),
            Some(TokenTree::Punct(punct)) if punct.as_char() == '='
        ) && matches!(
            tokens.get(index + 2),
            Some(TokenTree::Punct(punct)) if punct.as_char() == '>'
        );

        if matches!(previous.as_str(), "let" | "mut" | "sched" | "for") || introduces_arm {
            return Err(MacroError::at(
                format!(
                    "kernel `{kernel_name}` binds `{name}`, which is also one of its sizes; rename the \
                     binding, or the size would be replaced by its value here"
                ),
                Some(ident.span()),
            ));
        }
    }

    Ok(())
}

fn substitute_generics_in_tokens(
    token_stream: TokenStream,
    sizes: &HashMap<String, String>,
) -> Result<TokenStream, MacroError> {
    let mut output = TokenStream::new();
    let mut previous_is_dot = false;
    let mut penultimate_is_dot = false;
    for token_tree in token_stream {
        let is_dot = matches!(&token_tree, TokenTree::Punct(punct) if punct.as_char() == '.');
        let follows_dot = previous_is_dot && !penultimate_is_dot;
        penultimate_is_dot = previous_is_dot;
        previous_is_dot = is_dot;
        match token_tree {
            TokenTree::Group(group) => {
                let mut new_group = Group::new(
                    group.delimiter(),
                    substitute_generics_in_tokens(group.stream(), sizes)?,
                );
                new_group.set_span(group.span());
                output.extend([TokenTree::Group(new_group)]);
            }
            TokenTree::Ident(ident) => {
                let name = ident.to_string();
                if let Some(size) = sizes.get(&name).filter(|_| !follows_dot) {
                    debug!("substituting size `{name}` with `{size}`");
                    let size_tokens = size.parse::<TokenStream>().map_err(|error| {
                        MacroError::at(
                            format!(
                                "the value `{size}` bound to size `{name}` is not a valid size \
                                 expression: {error}"
                            ),
                            Some(ident.span()),
                        )
                    })?;
                    output.extend(size_tokens);
                    continue;
                }
                output.extend([TokenTree::Ident(ident)]);
            }
            _ => output.extend([token_tree]),
        }
    }
    Ok(output)
}

fn filter_generics(tokens: Vec<TokenTree>, keys: &[&str]) -> TokenStream {
    let mut output = TokenStream::new();
    let mut parts = Vec::new();
    let mut current_part = Vec::new();

    for token_tree in tokens {
        if let TokenTree::Punct(punct) = &token_tree {
            if punct.as_char() == ',' {
                parts.push(current_part);
                current_part = Vec::new();
                continue;
            }
        }
        current_part.push(token_tree);
    }
    if !current_part.is_empty() {
        parts.push(current_part);
    }

    let mut keep_parts = Vec::new();
    for part in parts {
        let mut keep = true;
        for token_tree in &part {
            if let TokenTree::Ident(ident) = token_tree {
                if keys.contains(&ident.to_string().as_str()) {
                    keep = false;
                }
                break;
            }
        }
        if keep {
            keep_parts.push(part);
        }
    }

    for (index, part) in keep_parts.into_iter().enumerate() {
        if index > 0 {
            output.extend([TokenTree::Punct(proc_macro2::Punct::new(
                ',',
                proc_macro2::Spacing::Alone,
            ))]);
        }
        for token_tree in part {
            output.extend([token_tree]);
        }
    }

    output
}

/// The tokens between a generic parameter list's `<` and its matching `>`, and the `>`
/// itself -- or `None` for the `>` if the stream ends before one.
fn extract_generic_block(
    iter: &mut std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
) -> (Vec<TokenTree>, Option<TokenTree>) {
    let mut inner_tokens = Vec::new();
    let mut depth = 1;

    for token_tree in iter.by_ref() {
        if let TokenTree::Punct(inner_punct) = &token_tree {
            if inner_punct.as_char() == '<' {
                depth += 1;
            } else if inner_punct.as_char() == '>' {
                depth -= 1;
                if depth == 0 {
                    return (inner_tokens, Some(token_tree));
                }
            }
        }
        inner_tokens.push(token_tree);
    }

    (inner_tokens, None)
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
        kernel_name: &str,
        pairs: &[(&str, &str)],
    ) -> Result<String, MacroError> {
        let token_stream = TokenStream::from_str(source).expect("test input must tokenize");
        let rendered = crate::source::DescendSource::from_token_stream(specialize_kernel(
            token_stream,
            kernel_name,
            &sizes(pairs),
        )?)
        .text;
        Ok(rendered.split_whitespace().collect())
    }

    #[test]
    fn the_targets_sizes_become_literals() {
        let rendered = specialize(
            "fn gpu_k<N: nat, r: prv>(a: &r uniq gpu.global [i32; N]) { let x = N; }",
            "gpu_k",
            &[("N", "16384")],
        )
        .unwrap();
        assert!(rendered.contains("[i32;16384]"), "{rendered}");
        assert!(rendered.contains("letx=16384"), "{rendered}");
        assert!(
            !rendered.contains("N:nat"),
            "the declaration should be gone: {rendered}"
        );
    }

    #[test]
    fn a_sibling_kernel_is_left_alone() {
        let rendered = specialize(
            "fn gpu_k<N: nat>(a: &r uniq gpu.global [i32; N]) { let x = N; }
             fn other<N: nat>(b: &r uniq gpu.global [i32; N]) { let N = 1; }",
            "gpu_k",
            &[("N", "16384")],
        )
        .unwrap();
        assert!(
            rendered.contains("[i32;16384]"),
            "the target was not specialized: {rendered}"
        );
        assert!(
            rendered.contains("fnother<N:nat>"),
            "the sibling's generics were touched: {rendered}"
        );
        assert!(
            rendered.contains("letN=1"),
            "the sibling's body was touched: {rendered}"
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
        let rendered = specialize(
            "fn gpu_k<n: nat>(a: &r uniq gpu.global [i32; n]) { let y = p.n; for i in 0..n { } }",
            "gpu_k",
            &[("n", "128")],
        )
        .unwrap();
        assert!(
            rendered.contains("p.n"),
            "a field name must survive: {rendered}"
        );
        assert!(
            rendered.contains("0..128"),
            "a range bound is a size: {rendered}"
        );
    }

    /// The `>` used to be invented here, which produced a token stream that looked well
    /// formed and then failed inside Descend's parser, pointing at source nobody wrote.
    #[test]
    fn a_generic_parameter_list_that_is_never_closed_is_reported() {
        let error = specialize(
            "fn gpu_k<N: nat(a: &r uniq gpu.global [i32; N]) { () }",
            "gpu_k",
            &[("N", "16384")],
        )
        .unwrap_err();
        assert!(
            error.message.contains("nothing closes"),
            "unhelpful error: {error}"
        );
        assert!(
            error.span.is_some(),
            "the report should point at the `<` that opened the list: {error}"
        );
    }

    #[test]
    fn a_kernel_the_source_does_not_define_is_said_to_be_missing() {
        let error = specialize(
            "fn gpu_k<N: nat>(a: &r uniq gpu.global [i32; N]) { () }",
            "gpu_other",
            &[("N", "16384")],
        )
        .unwrap_err();
        assert!(
            error.message.contains("gpu_other") && error.message.contains("not defined"),
            "{error}"
        );
    }
}
