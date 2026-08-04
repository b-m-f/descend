use proc_macro::TokenStream;

pub fn extract_source_str(input: &TokenStream) -> String {
    let mut first_span = None;
    let mut last_span = None;
    for token in input.clone() {
        if first_span.is_none() {
            first_span = Some(token.span());
        }
        last_span = Some(token.span());
    }

    let mut source_str = if let (Some(first), Some(last)) = (first_span, last_span) {
        if let Some(joined) = first.join(last) {
            joined.source_text().unwrap_or_else(|| input.to_string())
        } else {
            input.to_string()
        }
    } else {
        input.to_string()
    };

    sanitize_tokens_fallback(&mut source_str);
    source_str
}

/// Aggressive token-spacing sanitization fallback for IDEs (like rust-analyzer)
/// where `source_text()` returns `None` and tokenization inserts spaces around punctuation.
fn sanitize_tokens_fallback(source_str: &mut String) {
    let replacements = [
        ("- [", "-["),
        ("] ->", "]->"),
        ("& uniq", "&uniq "),
        ("& shrd", "&shrd "),
        ("& r uniq", "&r uniq "),
        ("& r shrd", "&r shrd "),
        ("&uniq", "&uniq "),
        ("&shrd", "&shrd "),
        ("gpu . global", "gpu.global"),
        ("gpu . grid", "gpu.grid"),
        ("cpu . mem", "cpu.mem"),
        ("cpu . thread", "cpu.thread"),
        (" . ", "."),
    ];

    for (from, to) in replacements {
        *source_str = source_str.replace(from, to);
    }
}
