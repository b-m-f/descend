use crate::diag::MacroError;
use crate::nat::Dimensions;
use proc_macro2::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
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

    fn count_punct(&mut self, character: char) -> usize {
        let mut counted = 0;
        while self
            .tokens
            .next_if(|token_tree| {
                matches!(token_tree, TokenTree::Punct(punct) if punct.as_char() == character)
            })
            .is_some()
        {
            counted += 1;
        }
        counted
    }

    fn peek_group(&mut self) -> Option<Group> {
        match self.tokens.peek()? {
            TokenTree::Group(group) => Some(group.clone()),
            _ => None,
        }
    }

    fn peek_span(&mut self) -> Option<Span> {
        self.tokens.peek().map(|token_tree| token_tree.span())
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

    let mut awaiting_separator = false;

    loop {
        match cursor.next() {
            Some(TokenTree::Punct(punct)) if punct.as_char() == '>' => break,
            Some(TokenTree::Punct(punct)) if punct.as_char() == ',' => {
                if !awaiting_separator {
                    return Err(MacroError::at(
                        "unexpected `,` in the generic bindings: a `,` separates two bindings, \
                         and no binding precedes this one"
                            .to_string(),
                        Some(punct.span()),
                    ));
                }
                awaiting_separator = false;
            }
            Some(TokenTree::Ident(name)) => {
                if awaiting_separator {
                    return Err(MacroError::at(
                        format!("expected `,` between the generic bindings, found `{name}`"),
                        Some(name.span()),
                    ));
                }
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
                awaiting_separator = true;
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

    if let Some(trailing) = cursor.next() {
        return Err(MacroError::at(
            format!("unexpected `{trailing}` after the generic bindings"),
            Some(trailing.span()),
        ));
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
    pub dimension_annotation: Option<DimensionAnnotation>,
    pub bindings: TokenStream,
}

#[derive(Debug, Clone, Copy)]
pub struct DimensionAnnotation {
    pub grid: Dimensions,
    pub block: Dimensions,
    pub span: Option<Span>,
}

impl DimensionAnnotation {
    pub fn to_token_tree(&self) -> TokenTree {
        let side = |dimensions: &Dimensions| {
            TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                [dimensions.x, dimensions.y, dimensions.z]
                    .into_iter()
                    .map(|axis| TokenTree::Literal(Literal::u32_unsuffixed(axis)))
                    .collect(),
            ))
        };

        let mut sides = TokenStream::new();
        sides.extend([side(&self.grid)]);
        sides.extend([TokenTree::Punct(Punct::new(',', Spacing::Alone))]);
        sides.extend([side(&self.block)]);
        TokenTree::Group(Group::new(Delimiter::Bracket, sides))
    }

    pub fn from_token_tree(token: &TokenTree) -> Result<Self, MacroError> {
        let TokenTree::Group(group) = token else {
            return Err("expected the launch dimensions as a bracketed group".into());
        };
        if group.delimiter() != Delimiter::Bracket {
            return Err(MacroError::at(
                "expected the launch dimensions as a bracketed group".to_string(),
                Some(group.span()),
            ));
        }

        let mut sides = group
            .stream()
            .into_iter()
            .filter(|token| !matches!(token, TokenTree::Punct(_)));
        let Some(grid) = sides.next() else {
            return Err("the launch dimensions carry no grid".into());
        };
        let Some(block) = sides.next() else {
            return Err("the launch dimensions carry no block".into());
        };
        if sides.next().is_some() {
            return Err("the launch dimensions carry more than a grid and a block".into());
        }

        Ok(Self {
            grid: read_axis_triple(&grid, "grid")?,
            block: read_axis_triple(&block, "block")?,
            span: Some(group.span()),
        })
    }
}

fn describe(delimiter: Delimiter) -> &'static str {
    match delimiter {
        Delimiter::Parenthesis => "`(` … `)`",
        Delimiter::Brace => "`{` … `}`",
        Delimiter::Bracket => "`[` … `]`",
        Delimiter::None => "no delimiter",
    }
}

fn read_axis(literal: &Literal, which: &str, axis: &str) -> Result<u32, MacroError> {
    let spelled = literal.to_string();
    if !spelled.chars().all(|character| character.is_ascii_digit()) {
        return Err(MacroError::at(
            format!("the {which} {axis} dimension must be a plain integer, found `{spelled}`"),
            Some(literal.span()),
        ));
    }
    spelled
        .parse::<u32>()
        .map_err(|_| {
            MacroError::at(
                format!(
                    "the {which} {axis} dimension `{spelled}` is above the largest value a launch \
                 axis takes"
                ),
                Some(literal.span()),
            )
        })
        .and_then(|value| {
            if value == 0 {
                return Err(MacroError::at(
                    format!(
                    "the {which} {axis} dimension is 0; an axis of 0 runs no work, and no kernel \
                     declares one"
                ),
                    Some(literal.span()),
                ));
            }
            Ok(value)
        })
}

fn read_axis_triple(token: &TokenTree, which: &str) -> Result<Dimensions, MacroError> {
    let TokenTree::Group(group) = token else {
        return Err(MacroError::at(
            format!("expected the {which} dimensions as a group of one to three axes"),
            Some(token.span()),
        ));
    };

    let mut axes: Vec<u32> = Vec::new();
    let names = ["x", "y", "z"];
    for (index, token) in group
        .stream()
        .into_iter()
        .filter(|token| !matches!(token, TokenTree::Punct(_)))
        .enumerate()
    {
        let Some(name) = names.get(index) else {
            return Err(MacroError::at(
                format!("the {which} dimensions declare more than three axes"),
                Some(token.span()),
            ));
        };
        let TokenTree::Literal(literal) = token else {
            return Err(MacroError::at(
                format!("expected the {which} {name} dimension to be a plain integer"),
                Some(token.span()),
            ));
        };
        axes.push(read_axis(&literal, which, name)?);
    }

    if axes.is_empty() {
        return Err(MacroError::at(
            format!("the {which} dimensions declare no axis at all"),
            Some(group.span()),
        ));
    }
    while axes.len() < 3 {
        axes.push(1);
    }

    Ok(Dimensions {
        x: axes[0],
        y: axes[1],
        z: axes[2],
    })
}

fn parse_dimensions_side(
    cursor: &mut Cursor,
    which: &str,
    dimension_annotation_span: Span,
) -> Result<Dimensions, MacroError> {
    if let Some(group) = cursor.next_group() {
        if group.delimiter() != Delimiter::Parenthesis {
            return Err(MacroError::at(
                format!(
                    "expected the {which} dimensions as a single number or as `(x, y, z)`, found \
                     a group with {}",
                    describe(group.delimiter())
                ),
                Some(group.span()),
            ));
        }
        let token = TokenTree::Group(group);
        return read_axis_triple(&token, which);
    }

    let Some(literal) = cursor.next_literal() else {
        return Err(MacroError::at(
            format!("expected the {which} dimensions of `<<<grid, block>>>`, found something else"),
            Some(dimension_annotation_span),
        ));
    };

    let mut axes = [1u32; 3];
    axes[0] = read_axis(&literal, which, "x")?;
    Ok(Dimensions {
        x: axes[0],
        y: axes[1],
        z: axes[2],
    })
}

fn parse_dimension_annotation(
    cursor: &mut Cursor,
) -> Result<Option<DimensionAnnotation>, MacroError> {
    let Some(dimension_annotation_span) = cursor.peek_span() else {
        return Ok(None);
    };

    let opened = cursor.count_punct('<');
    if opened < 2 {
        return Ok(None);
    }
    if opened != 3 {
        return Err(MacroError::at(
            format!(
                "expected the launch dimensions as `<<<grid, block>>>`, found {opened} `<` \
                 before the grid"
            ),
            Some(dimension_annotation_span),
        ));
    }

    let grid = parse_dimensions_side(cursor, "grid", dimension_annotation_span)?;

    if !cursor.eat_punct(',') {
        return Err(MacroError::at(
            "expected `,` between the grid and the block of `<<<grid, block>>>`".to_string(),
            Some(dimension_annotation_span),
        ));
    }

    let block = parse_dimensions_side(cursor, "block", dimension_annotation_span)?;

    let closed = cursor.count_punct('>');
    if closed != 3 {
        return Err(MacroError::at(
            format!(
                "expected `>>>` to close the launch dimensions, found {closed} `>` after the block"
            ),
            Some(dimension_annotation_span),
        ));
    }

    Ok(Some(DimensionAnnotation {
        grid,
        block,
        span: Some(dimension_annotation_span),
    }))
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
        return Err(
            "expected a kernel name, as in `descend_fn!(gpu_vec_add, <N: 16384>)` or \
             `descend_fn!(gpu_vec_add<<<64, 256>>>, <N: 16384>)`"
                .into(),
        );
    };
    check_kernel_name(&kernel)?;

    let dimension_annotation = parse_dimension_annotation(&mut cursor)?;

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

    Ok(FunctionInvocation {
        kernel,
        dimension_annotation,
        bindings,
    })
}

#[derive(Debug)]
pub struct FunctionInternalArguments {
    pub source_token_stream: TokenStream,
    pub kernel: Ident,
    pub dimension_annotation: Option<DimensionAnnotation>,
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

    let dimension_annotation = match cursor.peek_group() {
        Some(group) if group.delimiter() == Delimiter::Bracket => {
            let token = cursor.next().expect("the peeked token is still next");
            let dimension_annotation = DimensionAnnotation::from_token_tree(&token)?;
            if !cursor.eat_punct(',') {
                return Err(MacroError::at(
                    "expected `,` after the launch dimensions".to_string(),
                    dimension_annotation.span,
                ));
            }
            Some(dimension_annotation)
        }
        _ => None,
    };

    Ok(FunctionInternalArguments {
        source_token_stream: source_group.stream(),
        kernel,
        dimension_annotation,
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
    fn a_missing_comma_between_bindings_is_refused_rather_than_guessed_at() {
        let error = sizes("<N: 256 M: 64>").unwrap_err();
        assert!(
            error.contains("expected `,` between the generic bindings") && error.contains("`M`"),
            "{error}"
        );
    }

    #[test]
    fn a_comma_with_no_binding_before_it_is_refused() {
        for source in ["<, N: 256>", "<N: 256,, M: 64>"] {
            let error = sizes(source).unwrap_err();
            assert!(
                error.contains("unexpected `,` in the generic bindings"),
                "unhelpful error for {source}: {error}"
            );
        }
    }

    #[test]
    fn tokens_after_the_closing_angle_bracket_are_refused_rather_than_ignored() {
        let error = sizes("<N: 256> leftover").unwrap_err();
        assert!(
            error.contains("after the generic bindings") && error.contains("leftover"),
            "{error}"
        );
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

    fn dimensions(x: u32, y: u32, z: u32) -> Dimensions {
        Dimensions { x, y, z }
    }

    fn annotated(source: &str) -> DimensionAnnotation {
        invocation(source)
            .expect("the invocation must parse")
            .dimension_annotation
            .expect("a dimension annotation must be read")
    }

    #[test]
    fn a_scalar_dimension_annotation_pads_both_sides_to_one() {
        let parsed = annotated("gpu_vec_add<<<64, 256>>>");
        assert_eq!(parsed.grid, dimensions(64, 1, 1));
        assert_eq!(parsed.block, dimensions(256, 1, 1));
    }

    #[test]
    fn a_tuple_dimension_annotation_reads_each_axis_it_lists() {
        let parsed = annotated("gpu_matmul<<<(16, 16), (32, 32)>>>");
        assert_eq!(parsed.grid, dimensions(16, 16, 1));
        assert_eq!(parsed.block, dimensions(32, 32, 1));
    }

    #[test]
    fn a_dimension_annotation_reads_three_axes() {
        let parsed = annotated("gpu<<<(2, 3, 4), (5, 6, 7)>>>");
        assert_eq!(parsed.grid, dimensions(2, 3, 4));
        assert_eq!(parsed.block, dimensions(5, 6, 7));
    }

    #[test]
    fn a_dimension_annotation_keeps_the_bindings_that_follow_it() {
        let parsed = invocation("gpu_vec_add<<<64, 256>>>, <N: 16384>").unwrap();
        assert!(parsed.dimension_annotation.is_some());
        assert_eq!(parsed.bindings.to_string(), "< N : 16384 >");
    }

    #[test]
    fn an_invocation_without_dimensions_is_read_exactly_as_before() {
        let parsed = invocation("gpu_vec_add, <N: 16384>").unwrap();
        assert!(parsed.dimension_annotation.is_none());
        assert_eq!(parsed.bindings.to_string(), "< N : 16384 >");
    }

    #[test]
    fn a_lone_angle_bracket_is_not_read_as_dimensions() {
        let error = invocation("gpu_vec_add <N: 256>").unwrap_err();
        assert!(error.contains("expected `,` after the kernel"), "{error}");
    }

    #[test]
    fn an_unclosed_dimension_annotation_is_refused() {
        let error = invocation("gpu_vec_add<<<64, 256>>").unwrap_err();
        assert!(
            error.contains("expected `>>>` to close the launch dimensions"),
            "{error}"
        );
    }

    #[test]
    fn a_dimension_annotation_needs_both_a_grid_and_a_block() {
        let error = invocation("gpu_vec_add<<<64>>>").unwrap_err();
        assert!(
            error.contains("expected `,` between the grid and the block"),
            "{error}"
        );
    }

    #[test]
    fn a_dimension_annotation_refuses_more_than_three_axes() {
        let error = invocation("gpu<<<(1, 2, 3, 4), (1)>>>").unwrap_err();
        assert!(error.contains("more than three axes"), "{error}");
    }

    #[test]
    fn a_dimension_annotation_refuses_an_axis_of_zero() {
        let error = invocation("gpu<<<0, 256>>>").unwrap_err();
        assert!(error.contains("an axis of 0 runs no work"), "{error}");
    }

    #[test]
    fn a_dimension_annotation_refuses_an_axis_that_is_not_a_number() {
        let error = invocation("gpu<<<block, 256>>>").unwrap_err();
        assert!(error.contains("expected the grid dimensions"), "{error}");
    }

    #[test]
    fn a_dimension_annotation_survives_the_carrier_round_trip() {
        let parsed = annotated("gpu<<<(2, 3), (256, 1, 1)>>>");
        let token = parsed.to_token_tree();
        let read_back =
            DimensionAnnotation::from_token_tree(&token).expect("the routed group must parse");
        assert_eq!(read_back.grid, parsed.grid);
        assert_eq!(read_back.block, parsed.block);
    }

    #[test]
    fn the_internal_arguments_read_a_routed_dimension_annotation() {
        let routed = TokenStream::from_str(
            "{ fn gpu_k() {} }, gpu_k, [(64, 1, 1), (256, 1, 1)], <N: 16384>",
        )
        .expect("must tokenize");
        let parsed = parse_function_internal_arguments(routed).expect("the routed call must parse");
        let annotated = parsed
            .dimension_annotation
            .expect("a dimension annotation must be read");
        assert_eq!(annotated.grid, dimensions(64, 1, 1));
        assert_eq!(annotated.block, dimensions(256, 1, 1));
        assert_eq!(parsed.bindings.to_string(), "< N : 16384 >");
    }

    #[test]
    fn the_internal_arguments_still_read_a_call_without_dimensions() {
        let plain =
            TokenStream::from_str("{ fn gpu_k() {} }, gpu_k, <N: 256>").expect("must tokenize");
        let parsed = parse_function_internal_arguments(plain).expect("the plain call must parse");
        assert!(parsed.dimension_annotation.is_none());
        assert_eq!(parsed.bindings.to_string(), "< N : 256 >");
    }
}
