use proc_macro2::{Delimiter, Ident, Span, TokenStream, TokenTree};
use std::collections::HashMap;

pub fn to_descend_string(stream: TokenStream) -> String {
    let mut out = String::new();
    let mut last_was_ident_or_lit = false;

    fn flatten(ts: TokenStream, out: &mut String, last_was_ident_or_lit: &mut bool) {
        for tt in ts {
            match tt {
                TokenTree::Group(g) => {
                    let delim = g.delimiter();
                    let (open, close) = match delim {
                        Delimiter::Parenthesis => ("(", ")"),
                        Delimiter::Brace => ("{", "}"),
                        Delimiter::Bracket => ("[", "]"),
                        Delimiter::None => ("", ""),
                    };
                    out.push(' ');
                    out.push_str(open);
                    *last_was_ident_or_lit = false;

                    flatten(g.stream(), out, last_was_ident_or_lit);

                    out.push_str(close);
                    out.push(' ');
                    *last_was_ident_or_lit = false;
                }
                TokenTree::Ident(i) => {
                    if *last_was_ident_or_lit {
                        out.push(' ');
                    }
                    out.push_str(&i.to_string());
                    *last_was_ident_or_lit = true;
                }
                TokenTree::Literal(l) => {
                    if *last_was_ident_or_lit {
                        out.push(' ');
                    }
                    out.push_str(&l.to_string());
                    *last_was_ident_or_lit = true;
                }
                TokenTree::Punct(p) => {
                    out.push(p.as_char());
                    *last_was_ident_or_lit = false;
                }
            }
        }
    }
    flatten(stream, &mut out, &mut last_was_ident_or_lit);
    out
}

pub fn parse_generic_bindings(input: TokenStream) -> Result<HashMap<String, String>, String> {
    let mut bindings = HashMap::new();
    let mut tokens = input.into_iter().peekable();

    match tokens.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == '<' => {}
        None => return Ok(bindings),
        Some(other) => {
            return Err(format!(
                "expected `<` to open the generic bindings, found `{other}`"
            ))
        }
    }

    loop {
        match tokens.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == '>' => break,
            Some(TokenTree::Punct(p)) if p.as_char() == ',' => continue,
            Some(TokenTree::Ident(name)) => {
                match tokens.next() {
                    Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
                    _ => return Err(format!("expected `:` after generic `{name}`")),
                }
                match tokens.next() {
                    Some(TokenTree::Literal(value)) => {
                        bindings.insert(name.to_string(), value.to_string());
                    }
                    _ => {
                        return Err(format!(
                            "generic `{name}` needs an integer literal: a kernel's sizes are \
                             compiled into its code, so they cannot come from a runtime value"
                        ))
                    }
                }
            }
            Some(other) => return Err(format!("unexpected `{other}` in the generic bindings")),
            None => return Err("the generic bindings are not terminated by `>`".to_string()),
        }
    }

    Ok(bindings)
}

pub fn parse_descend_source<'a>(
    source: &'a descend::parser::SourceCode,
) -> Result<descend::ast::CompilUnit<'a>, String> {
    descend::parser::parse(source)
        .map_err(|e| format!("Failed to parse descend macro input: {:?}", e))
}

pub fn kernel_name_span(source_ts: &TokenStream, name: &str) -> Option<Span> {
    let mut previous_was_fn = false;
    for tt in source_ts.clone() {
        match &tt {
            TokenTree::Ident(ident) if previous_was_fn && ident == name => {
                return Some(ident.span());
            }
            TokenTree::Ident(ident) => previous_was_fn = ident == "fn",
            _ => previous_was_fn = false,
        }
    }
    None
}

pub struct FnInvocation {
    pub kernel: Ident,
    pub item_name: Ident,
    pub bindings: TokenStream,
}

pub fn parse_fn_invocation(input: TokenStream) -> Result<FnInvocation, String> {
    let mut iter = input.into_iter();

    let Some(TokenTree::Ident(kernel)) = iter.next() else {
        return Err(
            "expected a kernel name, as in `descend_fn!(gpu_vec_add, <N: 16384>)`".to_string(),
        );
    };

    let item_name = match iter.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == ',' => kernel.clone(),
        Some(TokenTree::Ident(keyword)) if keyword == "as" => {
            let Some(TokenTree::Ident(alias)) = iter.next() else {
                return Err(format!(
                    "expected a name for the instantiation of `{kernel}`"
                ));
            };
            match iter.next() {
                Some(TokenTree::Punct(p)) if p.as_char() == ',' => {}
                _ => return Err(format!("expected `,` after the name `{alias}`")),
            }
            alias
        }
        _ => {
            return Err(format!(
                "expected `,` or `as <name>` after the kernel `{kernel}`"
            ))
        }
    };

    let bindings: TokenStream = iter.collect();

    Ok(FnInvocation {
        kernel,
        item_name,
        bindings,
    })
}

pub struct FnInternalArgs {
    pub source_ts: TokenStream,
    pub kernel_name: String,
    pub item_name: String,
    pub item_span: Span,
    pub bindings: TokenStream,
}

pub fn parse_fn_internal_args(input: TokenStream) -> Result<FnInternalArgs, String> {
    let mut iter = input.into_iter();

    let Some(TokenTree::Group(source_group)) = iter.next() else {
        return Err("expected the Descend source block as the first argument".to_string());
    };
    let source_ts = source_group.stream();

    let _comma = iter.next();
    let kernel_name = iter
        .next()
        .ok_or_else(|| "expected the kernel name".to_string())?
        .to_string();
    let _comma = iter.next();
    let item_name_tt = iter
        .next()
        .ok_or_else(|| "expected the name of the instantiation".to_string())?;
    let item_span = item_name_tt.span();
    let item_name = item_name_tt.to_string();
    let _comma = iter.next();

    Ok(FnInternalArgs {
        source_ts,
        kernel_name,
        item_name,
        item_span,
        bindings: iter.collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn packed(source: &str) -> String {
        to_descend_string(TokenStream::from_str(source).expect("test input must tokenize"))
    }

    #[test]
    fn address_spaces_keep_their_dot() {
        assert!(packed("gpu.global").contains("gpu.global"));
        assert!(packed("gpu.shared").contains("gpu.shared"));
    }

    #[test]
    fn ownership_and_provenance_stay_separate_words() {
        let out = packed("a_array: &r uniq gpu.global [i32; N]");
        assert!(out.contains("&r uniq gpu.global"), "{out}");
    }

    #[test]
    fn turbofish_stays_glued() {
        let out = packed("(*a).to_view.grp::<256>[[block]][[thread]]");
        assert!(out.contains("grp::<256>"), "{out}");
    }

    #[test]
    fn a_packed_kernel_parses_as_descend() {
        let source = packed(
            "fn gpu_vec_add<N: nat, r: prv>(
                a_array: &r uniq gpu.global [i32; N],
                b_array: &r shrd gpu.global [i32; N]
            ) -[grid: gpu.grid<X<64>, X<256>>]-> () {
                sched block in grid {
                    sched thread in block {
                        let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                        *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
                    }
                }
            }",
        );
        let source_code = descend::parser::SourceCode::new(source.clone());
        assert!(
            descend::parser::parse(&source_code).is_ok(),
            "the packed source did not parse:\n{source}"
        );
    }

    #[test]
    fn a_packed_shared_memory_kernel_parses_as_descend() {
        let source = packed(
            "fn gpu_copy<r1: prv, r2: prv>(
                vec: &r1 uniq gpu.global [i32; 64],
                [grid.forall(X)] tmp: &r2 uniq gpu.shared [i32; 64]
            ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
                let view = &uniq (*vec).to_view.grp::<64>;
                sched block in grid {
                    let ib = &uniq (*view)[[block]];
                    let tmp_view = &uniq (*tmp).to_view;
                    sched thread in block {
                        let v = &uniq (*ib)[[thread]];
                        let t = &uniq (*tmp_view)[[thread]];
                        *t = *v;
                        *v = *t
                    }
                }
            }",
        );
        let source_code = descend::parser::SourceCode::new(source.clone());
        assert!(
            descend::parser::parse(&source_code).is_ok(),
            "the packed source did not parse:\n{source}"
        );
    }

    #[test]
    fn the_execution_annotation_survives() {
        let out = packed("-[grid: gpu.grid<X<64>, X<256>>]-> ()");
        assert!(out.contains("gpu.grid"), "{out}");
        assert!(out.contains("X<64>"), "{out}");
    }

    #[test]
    fn adjacent_identifiers_do_not_merge() {
        let out = packed("let mut x");
        assert!(out.contains("let mut x"), "{out}");
    }

    fn bindings(source: &str) -> Result<HashMap<String, String>, String> {
        parse_generic_bindings(TokenStream::from_str(source).expect("test input must tokenize"))
    }

    #[test]
    fn reads_several_sizes() {
        let parsed = bindings("<bs: 16, m: 64, n: 128, k: 64>").unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed["bs"], "16");
        assert_eq!(parsed["k"], "64");
    }

    #[test]
    fn an_empty_binding_list_is_allowed() {
        assert!(bindings("<>").unwrap().is_empty());
        assert!(bindings("").unwrap().is_empty());
    }

    #[test]
    fn a_runtime_expression_is_rejected_with_the_reason() {
        let error = bindings("<N: n>").unwrap_err();
        assert!(
            error.contains("needs an integer literal") && error.contains("compiled into its code"),
            "unhelpful error: {error}"
        );
    }

    #[test]
    fn a_missing_colon_is_rejected() {
        assert!(bindings("<N 16384>").unwrap_err().contains("expected `:`"));
    }

    #[test]
    fn an_unterminated_list_is_rejected() {
        assert!(bindings("<N: 16384")
            .unwrap_err()
            .contains("not terminated"));
    }
}
