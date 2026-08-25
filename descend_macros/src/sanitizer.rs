use proc_macro::{TokenStream, TokenTree, Spacing};

pub fn extract_source_str(input: &TokenStream) -> String {
    let mut first_span = None;
    let mut last_span = None;
    for token in input.clone() {
        if first_span.is_none() {
            first_span = Some(token.span());
        }
        last_span = Some(token.span());
    }

    if let (Some(first), Some(last)) = (first_span, last_span) {
        if let Some(joined) = first.join(last) {
            if let Some(text) = joined.source_text() {
                return text;
            }
        }
    }

    // Fallback: manually serialize tokens respecting Punct spacing
    token_stream_to_string(input.clone())
}

fn token_stream_to_string(ts: TokenStream) -> String {
    let mut out = String::new();
    let mut last_was_ident_or_lit = false;
    
    fn flatten(ts: TokenStream, out: &mut String, last_was_ident_or_lit: &mut bool) {
        for tt in ts {
            match tt {
                TokenTree::Group(g) => {
                    let delim = g.delimiter();
                    let (open, close) = match delim {
                        proc_macro::Delimiter::Parenthesis => ("(", ")"),
                        proc_macro::Delimiter::Brace => ("{", "}"),
                        proc_macro::Delimiter::Bracket => ("[", "]"),
                        proc_macro::Delimiter::None => ("", ""),
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
    flatten(ts, &mut out, &mut last_was_ident_or_lit);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    
    #[test]
    fn test_print_tokens() {
        let tokens = quote! {
            fn gpu_matmul<bs: nat, m: nat, n: nat, k: nat, r: prv>(
                a_mat: &r shrd gpu.global [i32; m*n],
                b_mat: &r shrd gpu.global [i32; n*k],
                c_mat: &r uniq gpu.global [i32; m*k]
            ) -[grid: gpu.grid<XY<k/bs, m/bs>, XY<bs, bs>>]-> () {
                sched(Y) block_row in grid {
                }
            }
        };
        // Proc macros use proc_macro::TokenStream, quote uses proc_macro2::TokenStream
        // We can't directly convert without a proc macro context.
    }
}
