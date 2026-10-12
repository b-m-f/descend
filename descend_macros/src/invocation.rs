use crate::diag::MacroError;
use proc_macro2::{Group, Ident, Literal, TokenStream, TokenTree};
use std::collections::HashMap;

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
            .next_if(|token_tree| matches!(token_tree, TokenTree::Ident(_)))?
        {
            TokenTree::Ident(ident) => Some(ident),
            _ => None,
        }
    }

    fn next_literal(&mut self) -> Option<Literal> {
        match self
            .tokens
            .next_if(|token_tree| matches!(token_tree, TokenTree::Literal(_)))?
        {
            TokenTree::Literal(literal) => Some(literal),
            _ => None,
        }
    }

    fn next_group(&mut self) -> Option<Group> {
        match self
            .tokens
            .next_if(|token_tree| matches!(token_tree, TokenTree::Group(_)))?
        {
            TokenTree::Group(group) => Some(group),
            _ => None,
        }
    }

    fn eat_punct(&mut self, character: char) -> bool {
        self.tokens
            .next_if(|token_tree| {
                matches!(token_tree, TokenTree::Punct(punct) if punct.as_char() == character)
            })
            .is_some()
    }

    fn rest(self) -> TokenStream {
        self.tokens.collect()
    }
}

pub fn parse_sizes(input: TokenStream) -> Result<HashMap<String, String>, MacroError> {
    let mut sizes = HashMap::new();
    let mut cursor = Cursor::new(input);

    match cursor.next() {
        Some(TokenTree::Punct(punct)) if punct.as_char() == '<' => {}
        None => return Ok(sizes),
        Some(other) => {
            return Err(MacroError::at(
                format!("expected `<` to open the generic bindings, found `{other}`"),
                Some(other.span()),
            ))
        }
    }

    loop {
        match cursor.next() {
            Some(TokenTree::Punct(punct)) if punct.as_char() == '>' => break,
            Some(TokenTree::Punct(punct)) if punct.as_char() == ',' => continue,
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
                let Some(size) = plain_integer(&value) else {
                    return Err(MacroError::at(
                        format!(
                            "generic `{name}` is bound to `{value}`, which is not a \
                             non-negative integer literal: a size is compiled into the kernel's \
                             code, so it cannot be a float, a string, or carry a numeric suffix"
                        ),
                        Some(value.span()),
                    ));
                };
                if sizes.insert(name.to_string(), size).is_some() {
                    return Err(MacroError::at(
                        format!(
                            "size `{name}` is bound twice; one value would silently win, so \
                             the instantiation would not be the one written here"
                        ),
                        Some(name.span()),
                    ));
                }
            }
            Some(other) => {
                return Err(MacroError::at(
                    format!("unexpected `{other}` in the generic bindings"),
                    Some(other.span()),
                ))
            }
            None => return Err("the generic bindings are not terminated by `>`".into()),
        }
    }

    Ok(sizes)
}

fn plain_integer(literal: &Literal) -> Option<String> {
    let spelled = literal.to_string();
    let is_plain_integer =
        !spelled.is_empty() && spelled.chars().all(|character| character.is_ascii_digit());
    is_plain_integer.then_some(spelled)
}

#[derive(Debug)]
pub struct FunctionInvocation {
    pub kernel: Ident,
    pub bindings: TokenStream,
}

pub fn check_kernel_name(kernel: &Ident) -> Result<(), MacroError> {
    let spelled = kernel.to_string();
    match spelled.strip_prefix("r#") {
        Some(bare) => Err(MacroError::at(
            format!(
                "`{spelled}` cannot name a kernel: `{bare}` is a Rust keyword, and a kernel's \
                 name also names the function generated from it"
            ),
            Some(kernel.span()),
        )),
        None => Ok(()),
    }
}

pub fn parse_function_invocation(input: TokenStream) -> Result<FunctionInvocation, MacroError> {
    let mut cursor = Cursor::new(input);

    let Some(kernel) = cursor.next_ident() else {
        return Err("expected a kernel name, as in `descend_fn!(gpu_vec_add, <N: 16384>)`".into());
    };
    check_kernel_name(&kernel)?;

    let bindings = if cursor.eat_punct(',') {
        cursor.rest()
    } else {
        let rest = cursor.rest();
        if !rest.is_empty() {
            return Err(MacroError::at(
                format!("expected `,` after the kernel `{kernel}`"),
                Some(kernel.span()),
            ));
        }
        rest
    };

    Ok(FunctionInvocation { kernel, bindings })
}

#[derive(Debug)]
pub struct FunctionInternalArguments {
    pub source_token_stream: TokenStream,
    pub kernel: Ident,
    pub bindings: TokenStream,
}

pub fn parse_function_internal_arguments(
    input: TokenStream,
) -> Result<FunctionInternalArguments, MacroError> {
    let mut cursor = Cursor::new(input);

    let Some(source_group) = cursor.next_group() else {
        return Err("expected the Descend source block as the first argument".into());
    };

    if !cursor.eat_punct(',') {
        return Err("expected `,` after the Descend source block".into());
    }

    let Some(kernel) = cursor.next_ident() else {
        return Err("expected the kernel name".into());
    };

    if !cursor.eat_punct(',') {
        return Err(MacroError::at(
            format!("expected `,` after the kernel `{kernel}`"),
            Some(kernel.span()),
        ));
    }

    Ok(FunctionInternalArguments {
        source_token_stream: source_group.stream(),
        kernel,
        bindings: cursor.rest(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn sizes(source: &str) -> Result<HashMap<String, String>, String> {
        parse_sizes(TokenStream::from_str(source).expect("test input must tokenize"))
            .map_err(|error| error.message)
    }

    fn invocation(source: &str) -> Result<FunctionInvocation, String> {
        parse_function_invocation(TokenStream::from_str(source).expect("test input must tokenize"))
            .map_err(|error| error.message)
    }

    #[test]
    fn reads_several_sizes() {
        let parsed = sizes("<bs: 16, m: 64, n: 128, k: 64>").unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed["bs"], "16");
        assert_eq!(parsed["k"], "64");
    }

    #[test]
    fn an_empty_binding_list_is_allowed() {
        assert!(sizes("<>").unwrap().is_empty());
        assert!(sizes("").unwrap().is_empty());
    }

    #[test]
    fn a_runtime_expression_is_rejected_with_the_reason() {
        let error = sizes("<N: n>").unwrap_err();
        assert!(
            error.contains("needs an integer literal") && error.contains("compiled into its code"),
            "unhelpful error: {error}"
        );
    }

    #[test]
    fn a_missing_colon_is_rejected() {
        assert!(sizes("<N 16384>").unwrap_err().contains("expected `:`"));
    }

    #[test]
    fn an_unterminated_list_is_rejected() {
        assert!(sizes("<N: 16384").unwrap_err().contains("not terminated"));
    }

    #[test]
    fn a_size_that_is_not_a_plain_integer_is_refused_at_the_binding() {
        for (source, spelled) in [
            ("<N: 1.5>", "1.5"),
            ("<N: \"x\">", "\"x\""),
            ("<N: 'a'>", "'a'"),
            ("<N: 16384u32>", "16384u32"),
        ] {
            let error = sizes(source).unwrap_err();
            assert!(
                error.contains(spelled) && error.contains("non-negative integer literal"),
                "unhelpful error for {source}: {error}"
            );
            assert!(error.contains('`') && error.contains("N"), "{error}");
        }
    }

    #[test]
    fn a_negative_size_is_refused() {
        assert!(sizes("<N: -4>")
            .unwrap_err()
            .contains("needs an integer literal"));
    }

    #[test]
    fn zero_is_a_size_like_any_other() {
        assert_eq!(sizes("<N: 0>").unwrap()["N"], "0");
    }

    #[test]
    fn a_kernel_name_is_read_alone_or_with_its_bindings() {
        assert_eq!(
            invocation("gpu_vec_add").unwrap().kernel.to_string(),
            "gpu_vec_add"
        );
        let with_bindings = invocation("gpu_vec_add, <N: 16384>").unwrap();
        assert_eq!(with_bindings.kernel.to_string(), "gpu_vec_add");
        assert_eq!(with_bindings.bindings.to_string(), "< N : 16384 >");
    }

    #[test]
    fn a_rust_keyword_cannot_name_a_kernel() {
        let error = invocation("r#match, <N: 256>").unwrap_err();
        assert!(
            error.contains("r#match") && error.contains("`match` is a Rust keyword"),
            "{error}"
        );
    }

    #[test]
    fn bindings_without_a_comma_are_refused_rather_than_guessed_at() {
        let error = invocation("gpu_vec_add <N: 256>").unwrap_err();
        assert!(error.contains("expected `,` after the kernel"), "{error}");
    }

    #[test]
    fn something_that_is_not_a_name_at_all_is_said_to_be_missing() {
        assert!(invocation("<N: 256>")
            .unwrap_err()
            .contains("expected a kernel name"));
    }

    #[test]
    fn internal_arguments_come_back_in_the_order_the_carrier_wrote_them() {
        let source = TokenStream::from_str("fn gpu_k() -[grid: gpu.grid<X<1>, X<1>>]-> () { () }")
            .expect("fixture must tokenize");
        let mut input = TokenStream::from(proc_macro2::TokenTree::Group(proc_macro2::Group::new(
            proc_macro2::Delimiter::Brace,
            source.clone(),
        )));
        input.extend(TokenStream::from_str(", gpu_k, <N: 256>").expect("must tokenize"));

        let arguments = parse_function_internal_arguments(input).unwrap();
        assert_eq!(arguments.kernel.to_string(), "gpu_k");
        assert_eq!(
            arguments.source_token_stream.to_string(),
            source.to_string()
        );
        assert_eq!(arguments.bindings.to_string(), "< N : 256 >");
    }

    #[test]
    fn internal_arguments_need_their_separators() {
        let source = proc_macro2::Group::new(
            proc_macro2::Delimiter::Brace,
            TokenStream::from_str("fn gpu_k() {}").expect("must tokenize"),
        );

        let missing_comma_after_source =
            TokenStream::from(proc_macro2::TokenTree::Group(source.clone()));
        assert!(
            parse_function_internal_arguments(missing_comma_after_source)
                .unwrap_err()
                .message
                .contains("expected `,` after the Descend source block")
        );

        let mut missing_comma_after_kernel =
            TokenStream::from(proc_macro2::TokenTree::Group(source.clone()));
        missing_comma_after_kernel.extend(TokenStream::from_str(", gpu_k").expect("must tokenize"));
        assert!(
            parse_function_internal_arguments(missing_comma_after_kernel)
                .unwrap_err()
                .message
                .contains("expected `,` after the kernel")
        );

        let no_source_block = TokenStream::from_str("gpu_k, <N: 256>").expect("must tokenize");
        assert!(parse_function_internal_arguments(no_source_block)
            .unwrap_err()
            .message
            .contains("expected the Descend source block"));
    }
}
