use descend::ast::Nat;
use std::collections::HashMap;

pub fn eval_nat(nat: &Nat, bindings: &HashMap<String, String>) -> Result<u32, String> {
    match nat {
        Nat::Lit(val) => {
            u32::try_from(*val).map_err(|_| format!("dimension literal {val} exceeds u32::MAX"))
        }
        Nat::Ident(ident) => {
            let name = ident.name.to_string();
            if let Some(val_str) = bindings.get(&name) {
                val_str
                    .parse::<u32>()
                    .map_err(|_| format!("generic dimension {name}={val_str:?} is not a valid u32"))
            } else {
                Err(format!(
                    "unknown identifier in dimension expression: {name}"
                ))
            }
        }
        Nat::BinOp(op, lhs, rhs) => {
            let l = eval_nat(lhs, bindings)?;
            let r = eval_nat(rhs, bindings)?;
            match op {
                descend::ast::BinOpNat::Add => l.checked_add(r),
                descend::ast::BinOpNat::Sub => l.checked_sub(r),
                descend::ast::BinOpNat::Mul => l.checked_mul(r),
                descend::ast::BinOpNat::Div => l.checked_div(r),
                descend::ast::BinOpNat::Mod => l.checked_rem(r),
            }
            .ok_or_else(|| format!("invalid or overflowing dimension expression: {l} {op:?} {r}"))
        }
        _ => Err(format!("unsupported dimension expression: {nat:?}")),
    }
}

pub fn eval_dim(
    dim: &descend::ast::Dim,
    bindings: &HashMap<String, String>,
) -> Result<(u32, u32, u32), String> {
    Ok(match dim {
        descend::ast::Dim::X(n) => (eval_nat(&n.0, bindings)?, 1, 1),
        descend::ast::Dim::Y(n) => (1, eval_nat(&n.0, bindings)?, 1),
        descend::ast::Dim::Z(n) => (1, 1, eval_nat(&n.0, bindings)?),
        descend::ast::Dim::XY(d) => (eval_nat(&d.0, bindings)?, eval_nat(&d.1, bindings)?, 1),
        descend::ast::Dim::XZ(d) => (eval_nat(&d.0, bindings)?, 1, eval_nat(&d.1, bindings)?),
        descend::ast::Dim::YZ(d) => (1, eval_nat(&d.0, bindings)?, eval_nat(&d.1, bindings)?),
        descend::ast::Dim::XYZ(d) => (
            eval_nat(&d.0, bindings)?,
            eval_nat(&d.1, bindings)?,
            eval_nat(&d.2, bindings)?,
        ),
    })
}

pub fn substitute_word(s: &str, word: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut last_end = 0;

    let bytes = s.as_bytes();
    let word_bytes = word.as_bytes();
    let mut i = 0;
    while i + word.len() <= bytes.len() {
        if &bytes[i..i + word.len()] == word_bytes {
            let prev_ok = if i == 0 {
                true
            } else {
                !bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_'
            };
            let next_idx = i + word.len();
            let next_ok = if next_idx == bytes.len() {
                true
            } else {
                !bytes[next_idx].is_ascii_alphanumeric() && bytes[next_idx] != b'_'
            };

            if prev_ok && next_ok {
                result.push_str(&s[last_end..i]);
                result.push_str(replacement);
                i += word.len();
                last_end = i;
                continue;
            }
        }
        i += 1;
    }
    result.push_str(&s[last_end..]);
    result
}
