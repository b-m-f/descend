pub fn eval_const_int(expr: &str) -> Result<u64, String> {
    let compact: String = expr.split_whitespace().collect();
    if compact.is_empty() {
        return Err("the shared-memory byte count is empty".to_string());
    }
    if let Some(bad) = compact
        .chars()
        .find(|c| !(c.is_ascii_digit() || "+-*/%()".contains(*c)))
    {
        return Err(format!(
            "the shared-memory byte count `{compact}` is not a constant integer expression \
             (found `{bad}`), so a kernel size was not resolved to a literal"
        ));
    }

    let mut parser = ExprParser {
        bytes: compact.as_bytes(),
        pos: 0,
    };
    let value = parser.expr()?;
    if parser.pos != parser.bytes.len() {
        return Err(format!(
            "trailing input in the shared-memory byte count `{compact}`"
        ));
    }
    Ok(value)
}

struct ExprParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl ExprParser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expr(&mut self) -> Result<u64, String> {
        let mut value = self.term()?;
        while let Some(op @ (b'+' | b'-')) = self.peek() {
            self.pos += 1;
            let rhs = self.term()?;
            value = match op {
                b'+' => value.checked_add(rhs),
                _ => value.checked_sub(rhs),
            }
            .ok_or_else(|| {
                format!(
                    "the shared-memory byte count overflows: {value} {} {rhs}",
                    op as char
                )
            })?;
        }
        Ok(value)
    }

    fn term(&mut self) -> Result<u64, String> {
        let mut value = self.atom()?;
        while let Some(op @ (b'*' | b'/' | b'%')) = self.peek() {
            self.pos += 1;
            let rhs = self.atom()?;
            value = match op {
                b'*' => value.checked_mul(rhs),
                b'/' => value.checked_div(rhs),
                _ => value.checked_rem(rhs),
            }
            .ok_or_else(|| {
                format!(
                    "the shared-memory byte count is invalid: {value} {} {rhs}",
                    op as char
                )
            })?;
        }
        Ok(value)
    }

    fn atom(&mut self) -> Result<u64, String> {
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let value = self.expr()?;
                if self.peek() != Some(b')') {
                    return Err(
                        "unbalanced parentheses in the shared-memory byte count".to_string()
                    );
                }
                self.pos += 1;
                Ok(value)
            }
            Some(c) if c.is_ascii_digit() => {
                let start = self.pos;
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.pos += 1;
                }
                std::str::from_utf8(&self.bytes[start..self.pos])
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
        assert!(eval_const_int("18446744073709551615 * 2").is_err());
    }

    #[test]
    fn division_and_modulo_are_supported() {
        assert_eq!(eval_const_int("(4 * (128 / 8))").unwrap(), 64);
        assert_eq!(eval_const_int("(10 % 4)").unwrap(), 2);
        assert!(eval_const_int("(4 / 0)").is_err());
    }
}
