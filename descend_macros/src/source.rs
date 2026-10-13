use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};

pub struct DescendSource {
    pub text: String,
    spans: Vec<(u32, u32, Span)>,
    last_was_ident_or_literal: bool,
}

impl DescendSource {
    pub fn span_at(&self, offset: u32) -> Option<Span> {
        let next = self.spans.partition_point(|(begin, _, _)| *begin <= offset);
        let (begin, end, span) = self.spans.get(next.checked_sub(1)?)?;
        (offset >= *begin && offset < *end).then_some(*span)
    }

    pub fn from_token_stream(token_stream: TokenStream) -> DescendSource {
        let mut source = DescendSource {
            text: String::new(),
            spans: Vec::new(),
            last_was_ident_or_literal: false,
        };
        source.push_token_stream(token_stream);
        source
    }

    fn push_token_stream(&mut self, token_stream: TokenStream) {
        for token_tree in token_stream {
            match token_tree {
                TokenTree::Group(group) => {
                    let (open, close) = match group.delimiter() {
                        Delimiter::Parenthesis => ("(", ")"),
                        Delimiter::Brace => ("{", "}"),
                        Delimiter::Bracket => ("[", "]"),
                        Delimiter::None => ("", ""),
                    };
                    if self.last_was_ident_or_literal {
                        self.text.push(' ');
                    }
                    if !open.is_empty() {
                        self.text.push_str(open);
                    }
                    self.last_was_ident_or_literal = false;

                    self.push_token_stream(group.stream());

                    if !close.is_empty() {
                        self.text.push_str(close);
                    }
                    self.text.push(' ');
                    self.last_was_ident_or_literal = false;
                }
                TokenTree::Ident(ident) => {
                    if self.last_was_ident_or_literal {
                        self.text.push(' ');
                    }
                    self.write_spanned(&ident.to_string(), ident.span());
                    self.last_was_ident_or_literal = true;
                }
                TokenTree::Literal(literal) => {
                    if self.last_was_ident_or_literal {
                        self.text.push(' ');
                    }
                    self.write_spanned(&literal.to_string(), literal.span());
                    self.last_was_ident_or_literal = true;
                }
                TokenTree::Punct(punct) => {
                    if self.last_was_ident_or_literal && punct.as_char() != '.' {
                        self.text.push(' ');
                    }
                    self.text.push(punct.as_char());
                    self.last_was_ident_or_literal = false;
                }
            }
        }
    }

    fn write_spanned(&mut self, text: &str, span: Span) {
        let begin = self.text.len() as u32;
        self.text.push_str(text);
        self.spans.push((begin, self.text.len() as u32, span));
    }
}

pub fn parse_descend_source<'a>(
    source: &'a descend::parser::SourceCode,
) -> Result<descend::ast::CompilUnit<'a>, crate::diag::MacroError> {
    let (parsed, captured) =
        crate::capture::with_captured_output(|| descend::parser::parse(source));

    parsed.map_err(|_| {
        crate::capture::descend_message(&captured)
            .unwrap_or_else(|| {
                "the Descend source did not parse; the Descend compiler reported the reason above"
                    .to_string()
            })
            .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn rendered_source(source: &str) -> String {
        DescendSource::from_token_stream(
            TokenStream::from_str(source).expect("test input must tokenize"),
        )
        .text
    }

    #[test]
    fn address_spaces_keep_their_dot() {
        assert!(rendered_source("gpu.global").contains("gpu.global"));
        assert!(rendered_source("gpu.shared").contains("gpu.shared"));
    }

    #[test]
    fn ownership_and_provenance_stay_separate_words() {
        let rendered = rendered_source("a_array: &r uniq gpu.global [i32; N]");
        assert!(rendered.contains("&r uniq gpu.global"), "{rendered}");
    }

    #[test]
    fn only_identifiers_and_literals_carry_a_span() {
        let source = DescendSource::from_token_stream(
            TokenStream::from_str("(*b_array)").expect("the expression must tokenize"),
        );
        let dereference = source.text.find("*b_array").unwrap() as u32;
        let identifier = source.text.find("b_array").unwrap() as u32;

        assert!(
            source.span_at(dereference).is_none(),
            "punctuation has no span to point a diagnostic at"
        );
        assert!(source.span_at(identifier).is_some());
    }

    #[test]
    fn a_parse_failure_carries_the_compilers_message() {
        let rendered = DescendSource::from_token_stream(
            TokenStream::from_str("fn bad() -[grid: gpu.grid<X<1>, X<1>>]-> () { @ }")
                .expect("the source must tokenize"),
        );
        let source = descend::parser::SourceCode::new(rendered.text.clone());

        let error = parse_descend_source(&source).expect_err("@ must be rejected");

        assert!(!error.message.is_empty(), "{error:?}");
        assert!(
            error.span.is_none(),
            "the Descend compiler reports the position itself, so the macro adds none: {error:?}"
        );
    }

    #[test]
    fn turbofish_stays_glued() {
        let rendered = rendered_source("(*a).to_view.grp::<256>[[block]][[thread]]");
        assert!(rendered.contains("::<256 >"), "{rendered}");
        assert!(
            rendered_source("g::<64>(x)").contains("::<64 >(x)"),
            "{}",
            rendered_source("g::<64>(x)")
        );
    }

    #[test]
    fn a_rendered_kernel_parses_as_descend() {
        let source = rendered_source(
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
            "the rendered source did not parse:\n{source}"
        );
    }

    #[test]
    fn a_rendered_shared_memory_kernel_parses_as_descend() {
        let source = rendered_source(
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
            "the rendered source did not parse:\n{source}"
        );
    }

    #[test]
    fn a_keyword_is_separated_from_following_punctuation() {
        assert!(rendered_source("g(&shrd *x)").contains("&shrd *x"));
        assert!(rendered_source("g(&uniq *x)").contains("&uniq *x"));
        assert!(rendered_source("for c in &uniq *chunk { () }").contains("in &uniq"));
    }

    const MANDATORY_WHITESPACE_SITES: &[(u32, &str, &str)] = &[
        (363, "struct __ ident", "struct Point { x: i32, y: i32 }"),
        (376, "fun_decl: fn __ ident", "fn g() -[t: cpu.thread]-> i32;"),
        (394, "global_fun_def: fn __ ident", "fn f() -[t: cpu.thread]-> () { () }"),
        (424, "param_decl: mutability __", "fn f(mut x: i32) -[t: cpu.thread]-> () { () }"),
        (424, "param_decl: `]` __ ident", "fn f([grid.forall(X)] tmp: &r uniq gpu.shared [i32; 8]) -[grid: gpu.grid<X<1>, X<8>>]-> () { () }"),
        (443, "pattern: mutability __ ident", "fn f() -[t: cpu.thread]-> () { let mut x = 1; () }"),
        (455, "let __ pattern", "fn f() -[t: cpu.thread]-> () { let x = 1; () }"),
        (464, "for __ ident __ in __ expression", "fn f() -[t: cpu.thread]-> () { for c in &uniq *chunk { () } }"),
        (469, "while __ cond __ block", "fn f() -[t: cpu.thread]-> () { while x { () } }"),
        (507, "expr __ as __ dty", "fn f() -[t: cpu.thread]-> () { let y = x as i32; () }"),
        (507, "expr __ as __ reference dty", "fn f() -[t: cpu.thread]-> () { let y = x as &r shrd gpu.global i32; () }"),
        (580, "& provenance __ ownership __ place", "fn f() -[t: cpu.thread]-> () { let a = &'a uniq *x; () }"),
        (580, "& ownership __ place", "fn f() -[t: cpu.thread]-> () { let a = &uniq *x; () }"),
        (593, "if __ cond", "fn f() -[t: cpu.thread]-> () { if x { () } else { () } }"),
        (600, "for __ ident __ in __ nat_range", "fn f() -[t: cpu.thread]-> () { for i in 0..4 { () } }"),
        (603, "indep `)` __ nat __ exec_expr", "fn f() -[grid: gpu.grid<X<1>, X<8>>]-> () { indep(X) 4 grid { a => { () }, b => { () } } }"),
        (612, "sched __ ident __ in __ exec_expr", "fn f() -[grid: gpu.grid<X<1>, X<8>>]-> () { sched block in grid { () } }"),
        (612, "sched `)` __ ident", "fn f() -[grid: gpu.grid<XY<2,2>, XY<4,4>>]-> () { sched(Y) r in grid { () } }"),
        (661, "let_uninit: let __ mut __ ident", "fn f() -[t: cpu.thread]-> () { let mut x: i32; () }"),
        (661, "let_uninit: let `]` __ mut", "fn f() -[grid: gpu.grid<X<1>, X<8>>]-> () { let [grid.forall(X)] mut x: i32; () }"),
        (830, "& provenance __ ownership __ memory_kind __ dty", "fn f(x: &'a uniq gpu.global [i32; 8]) -[t: cpu.thread]-> () { () }"),
        (830, "memory_kind __ reference dty", "fn f(x: &'a uniq gpu.global &'b shrd gpu.global i32) -[t: cpu.thread]-> () { () }"),
    ];

    fn parses(text: &str) -> bool {
        let code = descend::parser::SourceCode::new(text.to_string());
        crate::capture::with_captured_output(|| descend::parser::parse(&code))
            .0
            .is_ok()
    }

    #[test]
    fn every_mandatory_whitespace_site_survives_rendering() {
        for (line, site, source) in MANDATORY_WHITESPACE_SITES {
            assert!(
                parses(source),
                "the fixture for src/parser/mod.rs:{line} ({site}) is not valid Descend:\n{source}"
            );
            let token_stream = TokenStream::from_str(source).expect("fixture must tokenize");
            let rendered = DescendSource::from_token_stream(token_stream).text;
            assert!(
                parses(&rendered),
                "src/parser/mod.rs:{line} ({site}) does not survive rendering:\n{rendered}"
            );
        }
    }

    #[test]
    fn every_cited_grammar_line_still_demands_whitespace() {
        let parser_source =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../src/parser/mod.rs"))
                .expect("the parser source must be readable");
        let lines: Vec<&str> = parser_source.lines().collect();

        for (line, site, _) in MANDATORY_WHITESPACE_SITES {
            let cited = lines.get(*line as usize - 1).unwrap_or_else(|| {
                panic!("src/parser/mod.rs has no line {line}, cited for ({site})")
            });
            assert!(
                cited.contains("__"),
                "src/parser/mod.rs:{line} no longer demands whitespace, so the entry ({site}) \
                 cites the wrong line:\n{cited}"
            );
        }

        let whitespace_rule = 1049;
        assert!(
            lines
                .get(whitespace_rule - 1)
                .is_some_and(|line| line.contains("rule __()")),
            "the `__` rule is no longer at src/parser/mod.rs:{whitespace_rule}"
        );
    }

    #[test]
    fn dotted_keywords_stay_glued() {
        for keyword in [
            "cpu.mem",
            "gpu.global",
            "gpu.shared",
            "cpu.thread",
            "gpu.grid",
            "gpu.block",
            "gpu.thread",
        ] {
            let rendered = rendered_source(keyword);
            assert!(
                rendered.contains(keyword),
                "{keyword} was split into {rendered}"
            );
        }
    }

    #[test]
    fn every_descend_example_round_trips() {
        fn collect(directory: &std::path::Path, paths: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(directory) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let skipped = path
                        .file_name()
                        .is_some_and(|name| name == "target" || name == ".git");
                    if !skipped {
                        collect(&path, paths);
                    }
                } else if path
                    .extension()
                    .is_some_and(|extension| extension == "desc")
                {
                    paths.push(path);
                }
            }
        }

        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let mut files = Vec::new();
        collect(std::path::Path::new(root), &mut files);
        files.sort();
        assert!(!files.is_empty(), "no Descend sources found under {root}");

        for path in &files {
            let text = std::fs::read_to_string(path).unwrap();
            if !parses(&text) {
                continue;
            }
            let token_stream = TokenStream::from_str(&text).unwrap_or_else(|error| {
                panic!("{} does not tokenize as Rust: {error}", path.display())
            });
            let source = DescendSource::from_token_stream(token_stream).text;

            assert!(
                parses(&source),
                "{} no longer parses after rendering:\n{source}",
                path.display()
            );
        }
    }

    #[test]
    fn the_execution_annotation_survives() {
        let rendered = rendered_source("-[grid: gpu.grid<X<64>, X<256>>]-> ()");
        assert!(rendered.contains("gpu.grid"), "{rendered}");
        assert!(rendered.contains("X <64 >"), "{rendered}");
    }

    #[test]
    fn adjacent_identifiers_do_not_merge() {
        let rendered = rendered_source("let mut x");
        assert!(rendered.contains("let mut x"), "{rendered}");
    }
}
