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
                if sizes.insert(name.to_string(), value.to_string()).is_some() {
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
            None => {
                return Err("the generic bindings are not terminated by `>`"
                    .to_string()
                    .into())
            }
        }
    }

    Ok(sizes)
}

pub struct FunctionInvocation {
    pub kernel: Ident,
    pub bindings: TokenStream,
}

pub fn check_kernel_name(kernel: &Ident) -> Result<(), String> {
    let spelled = kernel.to_string();
    match spelled.strip_prefix("r#") {
        Some(bare) => Err(format!(
            "`{spelled}` cannot name a kernel: `{bare}` is a Rust keyword, and a kernel's \
             name also names the function generated from it"
        )),
        None => Ok(()),
    }
}

pub fn parse_function_invocation(input: TokenStream) -> Result<FunctionInvocation, String> {
    let mut cursor = Cursor::new(input);

    let Some(kernel) = cursor.next_ident() else {
        return Err(
            "expected a kernel name, as in `descend_fn!(gpu_vec_add, <N: 16384>)`".to_string(),
        );
    };
    check_kernel_name(&kernel)?;

    // `descend_fn!(gpu_vec_add)` for a kernel with no sizes; the comma only separates
    // the kernel from a size list that is actually there.
    let bindings = if cursor.eat_punct(',') {
        cursor.rest()
    } else {
        let rest = cursor.rest();
        if !rest.is_empty() {
            return Err(format!("expected `,` after the kernel `{kernel}`"));
        }
        rest
    };

    Ok(FunctionInvocation { kernel, bindings })
}

pub struct FunctionInternalArguments {
    pub source_token_stream: TokenStream,
    pub kernel: Ident,
    pub bindings: TokenStream,
}

pub fn parse_function_internal_arguments(
    input: TokenStream,
) -> Result<FunctionInternalArguments, String> {
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
}
