//! The dynamic shared-memory byte count, read back from the code the Descend
//! compiler generates.
//!
//! A `gpu.shared` kernel parameter becomes dynamic shared memory: the kernel
//! body receives one `extern __shared__ descend::byte $buffer[]` and slices it
//! into the declared arrays, so the size is not part of the kernel at all — it
//! is a launch argument. The Descend code generator computes that size for its
//! own host code and prints it as the third argument of the `<<<...>>>` launch.
//! Taking the number from there, rather than recomputing it, means the host's
//! request cannot disagree with the packing the device code performs.
//!
//! This module holds no `proc_macro` types, so it can be unit-tested outside a
//! macro invocation.

/// The dynamic shared-memory argument of the kernel launch in `host_fn`, with
/// all whitespace removed.
///
/// `host_fn` is the synthesized host function the Descend code generator
/// produced; it contains exactly one kernel launch. The text is searched as one
/// region rather than line by line, because the generated code is passed
/// through `clang-format`, which wraps a launch across lines.
pub fn shared_bytes_expr(host_fn: &str) -> Result<String, String> {
    let open = host_fn
        .find("<<<")
        .ok_or_else(|| "no kernel launch found in the generated host code".to_string())?;
    let rest = &host_fn[open + 3..];
    let close = rest
        .find(">>>")
        .ok_or_else(|| "the generated kernel launch is not terminated".to_string())?;
    let inside = &rest[..close];

    let args = split_top_level(inside);
    match args.as_slice() {
        [_grid, _block, bytes] => Ok(bytes.split_whitespace().collect()),
        other => Err(format!(
            "expected 3 launch arguments, found {} in `<<<{}>>>`",
            other.len(),
            inside.trim()
        )),
    }
}

/// Splits `args` at commas outside parentheses.
///
/// The launch arguments are `dim3(x, y, z)` values, whose own commas must not
/// split the list.
fn split_top_level(args: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in args.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&args[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&args[start..]);
    parts
}

/// Evaluates a constant integer expression of literals, `+ - * / %` and
/// parentheses.
///
/// Anything else — an identifier above all — means a kernel size never became a
/// literal, which is a bug in the macro's own substitution rather than something
/// to paper over with a guessed number.
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

/// The shared-memory byte count of the launch in `host_fn`.
pub fn shared_mem_bytes(host_fn: &str) -> Result<u64, String> {
    eval_const_int(&shared_bytes_expr(host_fn)?)
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

    /// The launch `descend::codegen` emits for `cuda-examples/asplos/sgemm-descend.cu`,
    /// wrapped across lines the way `clang-format` wraps it.
    const SGEMM_LAUNCH: &str = "__host__ auto dummy_main() -> void {
  sgemm_gpu<<<dim3(8, 16, 1), dim3(32, 8, 1),
              ((0 + (4 * ((1 * (32 * 2)) * 8))) + (4 * ((1 * (32 * 4)) * 8)))>>>(
      a, b, c);
}";

    #[test]
    fn reads_the_count_across_wrapped_lines() {
        assert_eq!(
            shared_bytes_expr(SGEMM_LAUNCH).unwrap(),
            "((0+(4*((1*(32*2))*8)))+(4*((1*(32*4))*8)))"
        );
        // 4 * (32*2 * 8) = 2048 for `a_tile`, 4 * (32*4 * 8) = 4096 for `b_tile`.
        assert_eq!(shared_mem_bytes(SGEMM_LAUNCH).unwrap(), 6144);
    }

    #[test]
    fn a_kernel_without_shared_parameters_requests_nothing() {
        let launch = "gpu_vec_add<<<dim3(64, 1, 1), dim3(256, 1, 1), 0>>>(a, b);";
        assert_eq!(shared_mem_bytes(launch).unwrap(), 0);
    }

    #[test]
    fn dim3_commas_do_not_split_the_argument_list() {
        let launch = "k<<<dim3(1, 2, 3), dim3(4, 5, 6), (4 * 64)>>>(x);";
        assert_eq!(shared_mem_bytes(launch).unwrap(), 256);
    }

    #[test]
    fn an_unresolved_size_is_an_error() {
        let launch = "k<<<dim3(1, 1, 1), dim3(1, 1, 1), (4 * bs)>>>(x);";
        let error = shared_mem_bytes(launch).unwrap_err();
        assert!(
            error.contains("not a constant integer expression"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_missing_launch_is_an_error() {
        assert!(shared_bytes_expr("__host__ auto dummy_main() -> void {}").is_err());
    }

    #[test]
    fn a_launch_with_two_arguments_is_an_error() {
        let launch = "k<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(x);";
        let error = shared_bytes_expr(launch).unwrap_err();
        assert!(error.contains("expected 3 launch arguments"), "{error}");
    }

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
