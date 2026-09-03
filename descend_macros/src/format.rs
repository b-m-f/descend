use proc_macro::{TokenStream, TokenTree, Delimiter};

/// Converts a TokenStream into a densely packed string adhering to the strict spacing requirements of the Descend PEG parser.
/// 
/// What it does: Recursively visits `TokenTree` instances, intelligently suppressing whitespace between punctuation while 
/// ensuring literals and identifiers remain properly spaced.
/// Why it is needed: The native rustc stringification (`TokenStream::to_string()`) injects syntactic spaces globally 
/// (e.g. `gpu . global` instead of `gpu.global`). The Descend compiler parser is brittle regarding these injected spaces, 
/// so this function guarantees a clean syntax payload compatible directly with the compiler.
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
