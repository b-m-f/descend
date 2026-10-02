pub fn eval_const_integer(expression: &str) -> Result<u64, String> {
    let without_whitespace: String = expression.split_whitespace().collect();
    if without_whitespace.is_empty() {
        return Err("the shared-memory byte count is empty".to_string());
    }
    if let Some(unexpected_character) = without_whitespace
        .chars()
        .find(|character| !(character.is_ascii_digit() || "+-*/%()".contains(*character)))
    {
        return Err(format!(
            "the shared-memory byte count `{without_whitespace}` is not a constant integer expression \
             (found `{unexpected_character}`), so a kernel size was not resolved to a literal"
        ));
    }

    let mut parser = ExpressionParser {
        bytes: without_whitespace.as_bytes(),
        position: 0,
    };
    let value = parser.expression()?;
    if parser.position != parser.bytes.len() {
        return Err(format!(
            "trailing input in the shared-memory byte count `{without_whitespace}`"
        ));
    }
    Ok(value)
}

struct ExpressionParser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl ExpressionParser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn expression(&mut self) -> Result<u64, String> {
        let mut value = self.term()?;
        while let Some(operator @ (b'+' | b'-')) = self.peek() {
            self.position += 1;
            let right_operand = self.term()?;
            value = match operator {
                b'+' => value.checked_add(right_operand),
                _ => value.checked_sub(right_operand),
            }
            .ok_or_else(|| {
                format!(
                    "the shared-memory byte count overflows: {value} {} {right_operand}",
                    operator as char
                )
            })?;
        }
        Ok(value)
    }

    fn term(&mut self) -> Result<u64, String> {
        let mut value = self.atom()?;
        while let Some(operator @ (b'*' | b'/' | b'%')) = self.peek() {
            self.position += 1;
            let right_operand = self.atom()?;
            value = match operator {
                b'*' => value.checked_mul(right_operand),
                b'/' => value.checked_div(right_operand),
                _ => value.checked_rem(right_operand),
            }
            .ok_or_else(|| {
                format!(
                    "the shared-memory byte count is invalid: {value} {} {right_operand}",
                    operator as char
                )
            })?;
        }
        Ok(value)
    }

    fn atom(&mut self) -> Result<u64, String> {
        match self.peek() {
            Some(b'(') => {
                self.position += 1;
                let value = self.expression()?;
                if self.peek() != Some(b')') {
                    return Err(
                        "unbalanced parentheses in the shared-memory byte count".to_string()
                    );
                }
                self.position += 1;
                Ok(value)
            }
            Some(character) if character.is_ascii_digit() => {
                let start = self.position;
                while self
                    .peek()
                    .is_some_and(|character| character.is_ascii_digit())
                {
                    self.position += 1;
                }
                std::str::from_utf8(&self.bytes[start..self.position])
                    .unwrap()
                    .parse::<u64>()
                    .map_err(|_| {
                        "a literal in the shared-memory byte count is too large".to_string()
                    })
            }
            _ => Err("the shared-memory byte count is malformed".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_is_an_error() {
        assert!(eval_const_integer("18446744073709551615 * 2").is_err());
    }

    #[test]
    fn division_and_modulo_are_supported() {
        assert_eq!(eval_const_integer("(4 * (128 / 8))").unwrap(), 64);
        assert_eq!(eval_const_integer("(10 % 4)").unwrap(), 2);
        assert!(eval_const_integer("(4 / 0)").is_err());
    }
}
