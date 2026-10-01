use crate::diag::MacroError;
use proc_macro2::{Delimiter, Group, Ident, Literal, Span, TokenStream, TokenTree};
use std::collections::HashMap;

pub struct DescendSource {
    pub text: String,
    spans: Vec<(u32, u32, Span)>,
}

impl DescendSource {
    pub fn span_at(&self, offset: u32) -> Option<Span> {
        let next = self.spans.partition_point(|(begin, _, _)| *begin <= offset);
        let (begin, end, span) = self.spans.get(next.checked_sub(1)?)?;
        (offset >= *begin && offset < *end).then_some(*span)
    }
}

pub fn flatten_source(stream: TokenStream) -> DescendSource {
    let mut source = DescendSource {
        text: String::new(),
        spans: Vec::new(),
    };
    let mut last_was_ident_or_lit = false;
    flatten(stream, &mut source, &mut last_was_ident_or_lit);
    source
}

fn flatten(ts: TokenStream, source: &mut DescendSource, last_was_ident_or_lit: &mut bool) {
    for tt in ts {
        match tt {
            TokenTree::Group(g) => {
                let (open, close) = match g.delimiter() {
                    Delimiter::Parenthesis => ("(", ")"),
                    Delimiter::Brace => ("{", "}"),
                    Delimiter::Bracket => ("[", "]"),
                    Delimiter::None => ("", ""),
                };
                source.text.push(' ');
                source.text.push_str(open);
                *last_was_ident_or_lit = false;

                flatten(g.stream(), source, last_was_ident_or_lit);

                source.text.push_str(close);
                source.text.push(' ');
                *last_was_ident_or_lit = false;
            }
            TokenTree::Ident(i) => {
                if *last_was_ident_or_lit {
                    source.text.push(' ');
                }
                source.write_spanned(&i.to_string(), i.span());
                *last_was_ident_or_lit = true;
            }
            TokenTree::Literal(l) => {
                if *last_was_ident_or_lit {
                    source.text.push(' ');
                }
                source.write_spanned(&l.to_string(), l.span());
                *last_was_ident_or_lit = true;
            }
            TokenTree::Punct(p) => {
                source.text.push(p.as_char());
                *last_was_ident_or_lit = false;
            }
        }
    }
}

impl DescendSource {
    fn write_spanned(&mut self, text: &str, span: Span) {
        let begin = self.text.len() as u32;
        self.text.push_str(text);
        self.spans.push((begin, self.text.len() as u32, span));
    }
}

struct Cursor {
    tokens: std::iter::Peekable<proc_macro2::token_stream::IntoIter>,
}

impl Cursor {
    fn new(stream: TokenStream) -> Self {
        Self {
            tokens: stream.into_iter().peekable(),
        }
    }

    fn next(&mut self) -> Option<TokenTree> {
        self.tokens.next()
    }

    fn next_ident(&mut self) -> Option<Ident> {
        match self
            .tokens
            .next_if(|tt| matches!(tt, TokenTree::Ident(_)))?
        {
            TokenTree::Ident(ident) => Some(ident),
            _ => None,
        }
    }

    fn next_literal(&mut self) -> Option<Literal> {
        match self
            .tokens
            .next_if(|tt| matches!(tt, TokenTree::Literal(_)))?
        {
            TokenTree::Literal(literal) => Some(literal),
            _ => None,
        }
    }

    fn next_group(&mut self) -> Option<Group> {
        match self
            .tokens
            .next_if(|tt| matches!(tt, TokenTree::Group(_)))?
        {
            TokenTree::Group(group) => Some(group),
            _ => None,
        }
    }

    fn eat_punct(&mut self, c: char) -> bool {
        self.tokens
            .next_if(|tt| matches!(tt, TokenTree::Punct(p) if p.as_char() == c))
            .is_some()
    }

    fn eat_keyword(&mut self, keyword: &str) -> bool {
        self.tokens
            .next_if(|tt| matches!(tt, TokenTree::Ident(i) if i == keyword))
            .is_some()
    }

    fn rest(self) -> TokenStream {
        self.tokens.collect()
    }
}

pub fn parse_generic_bindings(input: TokenStream) -> Result<HashMap<String, String>, MacroError> {
    let mut bindings = HashMap::new();
    let mut cursor = Cursor::new(input);

    match cursor.next() {
        Some(TokenTree::Punct(p)) if p.as_char() == '<' => {}
        None => return Ok(bindings),
        Some(other) => {
            return Err(MacroError::at(
                format!("expected `<` to open the generic bindings, found `{other}`"),
                Some(other.span()),
            ))
        }
    }

    loop {
        match cursor.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == '>' => break,
            Some(TokenTree::Punct(p)) if p.as_char() == ',' => continue,
            Some(TokenTree::Ident(name)) => {
                if !cursor.eat_punct(':') {
                    return Err(MacroError::at(
                        format!("expected `:` after generic `{name}`"),
                        Some(name.span()),
                    ));
                }
                let Some(value) = cursor.next_literal() else {
                    return Err(MacroError::at(
                        format!(
                            "generic `{name}` needs an integer literal: a kernel's sizes are \
                             compiled into its code, so they cannot come from a runtime value"
                        ),
                        Some(name.span()),
                    ));
                };
                bindings.insert(name.to_string(), value.to_string());
            }
            Some(other) => {
                return Err(MacroError::at(
                    format!("unexpected `{other}` in the generic bindings"),
                    Some(other.span()),
                ))
            }
            None => {
                return Err("the generic bindings are not terminated by `>`"
                    .to_string()
                    .into())
            }
        }
    }

    Ok(bindings)
}

pub fn parse_descend_source<'a>(
    source: &'a descend::parser::SourceCode,
) -> Result<descend::ast::CompilUnit<'a>, String> {
    let (parsed, captured) =
        crate::capture::with_captured_output(|| descend::parser::parse(source));

    parsed.map_err(|_| {
        crate::capture::descend_message(&captured).unwrap_or_else(|| {
            "the Descend source did not parse; the Descend compiler reported the reason above"
                .to_string()
        })
    })
}

pub struct FnInvocation {
    pub kernel: Ident,
    pub item_name: Ident,
    pub bindings: TokenStream,
}

pub fn parse_fn_invocation(input: TokenStream) -> Result<FnInvocation, String> {
    let mut cursor = Cursor::new(input);

    let Some(kernel) = cursor.next_ident() else {
        return Err(
            "expected a kernel name, as in `descend_fn!(gpu_vec_add, <N: 16384>)`".to_string(),
        );
    };

    let item_name = if cursor.eat_punct(',') {
        kernel.clone()
    } else if cursor.eat_keyword("as") {
        let Some(alias) = cursor.next_ident() else {
            return Err(format!(
                "expected a name for the instantiation of `{kernel}`"
            ));
        };
        if !cursor.eat_punct(',') {
            return Err(format!("expected `,` after the name `{alias}`"));
        }
        alias
    } else {
        return Err(format!(
            "expected `,` or `as <name>` after the kernel `{kernel}`"
        ));
    };

    Ok(FnInvocation {
        kernel,
        item_name,
        bindings: cursor.rest(),
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
    let mut cursor = Cursor::new(input);

    let Some(source_group) = cursor.next_group() else {
        return Err("expected the Descend source block as the first argument".to_string());
    };

    if !cursor.eat_punct(',') {
        return Err("expected `,` after the Descend source block".to_string());
    }

    let Some(kernel) = cursor.next_ident() else {
        return Err("expected the kernel name".to_string());
    };

    if !cursor.eat_punct(',') {
        return Err(format!("expected `,` after the kernel `{kernel}`"));
    }

    let Some(item) = cursor.next_ident() else {
        return Err("expected the name of the instantiation".to_string());
    };

    if !cursor.eat_punct(',') {
        return Err(format!("expected `,` after the name `{item}`"));
    }

    Ok(FnInternalArgs {
        source_ts: source_group.stream(),
        kernel_name: kernel.to_string(),
        item_name: item.to_string(),
        item_span: item.span(),
        bindings: cursor.rest(),
    })
}

pub fn extract_generic_block(
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

pub fn nat_generics(source_ts: &TokenStream, kernel: &str) -> Result<Vec<String>, String> {
    let mut iter = source_ts.clone().into_iter().peekable();

    while let Some(tt) = iter.next() {
        if !matches!(&tt, TokenTree::Ident(ident) if ident == "fn") {
            continue;
        }
        if !matches!(iter.peek(), Some(TokenTree::Ident(ident)) if ident == kernel) {
            continue;
        }
        iter.next();

        if !matches!(iter.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '<') {
            return Ok(Vec::new());
        }
        iter.next();

        let (inner_tokens, _) = extract_generic_block(&mut iter);
        return Ok(nat_generic_names(inner_tokens));
    }

    Err(format!(
        "Kernel definition '{}' not found in macro body",
        kernel
    ))
}

fn nat_generic_names(tokens: Vec<TokenTree>) -> Vec<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn packed(source: &str) -> String {
        flatten_source(TokenStream::from_str(source).expect("test input must tokenize")).text
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
            .map_err(|error| error.message)
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
