use proc_macro2::{Delimiter, TokenStream, TokenTree};

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
}
