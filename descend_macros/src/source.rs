use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};

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
    pub fn from_token_stream(stream: TokenStream) -> DescendSource {
        let mut source = DescendSource {
            text: String::new(),
            spans: Vec::new(),
        };
        let mut last_was_ident_or_lit = false;
        flatten(stream, &mut source, &mut last_was_ident_or_lit);
        source
    }
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
                // A space here keeps `&uniq (*x)` from collapsing into a call, but after
                // punctuation it would detach an argument list from its turbofish.
                if *last_was_ident_or_lit {
                    source.text.push(' ');
                }
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
                // The grammar's mandatory-whitespace rule `__` always follows a word, so
                // a word is separated from the punctuation after it. The one exception is
                // `.`: it glues the dotted keywords (`gpu.global`, `cpu.thread`, ..) that
                // `keyword()` matches as whole strings, and the view and field chains.
                if *last_was_ident_or_lit && p.as_char() != '.' {
                    source.text.push(' ');
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn packed(source: &str) -> String {
        DescendSource::from_token_stream(
            TokenStream::from_str(source).expect("test input must tokenize"),
        )
        .text
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
        // Words are separated from punctuation, so the spelling is `grp ::<256 >`; what
        // must hold is that `::<` is not broken up and the argument list stays attached.
        let out = packed("(*a).to_view.grp::<256>[[block]][[thread]]");
        assert!(out.contains("::<256 >"), "{out}");
        assert!(
            packed("g::<64>(x)").contains("::<64 >(x)"),
            "{}",
            packed("g::<64>(x)")
        );
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
    fn a_keyword_is_separated_from_following_punctuation() {
        // The grammar's `__` is mandatory whitespace, so these must not collapse.
        assert!(packed("g(&shrd *x)").contains("&shrd *x"));
        assert!(packed("g(&uniq *x)").contains("&uniq *x"));
        assert!(packed("for c in &uniq *chunk { () }").contains("in &uniq"));
    }

    /// Every place where the grammar demands whitespace, with the `src/parser/mod.rs`
    /// line of the `__` it exercises. `__` (line 1049) is one-or-more whitespace, so a
    /// flattening that glued these tokens together would make the source unparsable.
    ///
    /// The two `__` in `lambda_parameter` (line 430) are left out: that rule is
    /// unreachable, because the lambda expression that used it is commented out at
    /// line 619.
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
    fn every_mandatory_whitespace_site_survives_flattening() {
        for (line, site, source) in MANDATORY_WHITESPACE_SITES {
            assert!(
                parses(source),
                "the fixture for src/parser/mod.rs:{line} ({site}) is not valid Descend:\n{source}"
            );
            let stream = TokenStream::from_str(source).expect("fixture must tokenize");
            let flattened = DescendSource::from_token_stream(stream).text;
            assert!(
                parses(&flattened),
                "src/parser/mod.rs:{line} ({site}) does not survive flattening:\n{flattened}"
            );
        }
    }

    /// The `.` exception to that rule: these must stay glued, because `keyword()`
    /// matches them as whole strings (`src/parser/mod.rs:908` and `:965`).
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
            let out = packed(keyword);
            assert!(out.contains(keyword), "{keyword} was split into {out}");
        }
    }

    /// Every Descend source in the repository that the compiler accepts must still be
    /// accepted after a round trip through the Rust lexer and `flatten_source`.
    #[test]
    fn every_descend_example_round_trips() {
        fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() && !path.ends_with("target") {
                    collect(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "desc") {
                    out.push(path);
                }
            }
        }

        fn parses(text: &str) -> bool {
            let code = descend::parser::SourceCode::new(text.to_string());
            crate::capture::with_captured_output(|| descend::parser::parse(&code))
                .0
                .is_ok()
        }

        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
        let mut files = Vec::new();
        collect(std::path::Path::new(root), &mut files);
        files.sort();
        assert!(!files.is_empty(), "no Descend sources found under {root}");

        for path in &files {
            let text = std::fs::read_to_string(path).unwrap();
            // Sources the compiler already rejects say nothing about the round trip.
            if !parses(&text) {
                continue;
            }
            let stream = TokenStream::from_str(&text).unwrap_or_else(|err| {
                panic!("{} does not tokenize as Rust: {err}", path.display())
            });
            let source = DescendSource::from_token_stream(stream).text;

            assert!(
                parses(&source),
                "{} no longer parses after flattening:\n{source}",
                path.display()
            );
        }
    }

    #[test]
    fn the_execution_annotation_survives() {
        let out = packed("-[grid: gpu.grid<X<64>, X<256>>]-> ()");
        // The dotted keyword stays glued; the dimensions may be spaced, as the grammar
        // allows optional whitespace around `<`.
        assert!(out.contains("gpu.grid"), "{out}");
        assert!(out.contains("X <64 >"), "{out}");
    }

    #[test]
    fn adjacent_identifiers_do_not_merge() {
        let out = packed("let mut x");
        assert!(out.contains("let mut x"), "{out}");
    }
}
