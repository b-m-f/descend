use descend::ast::Nat;

pub fn eval_nat(nat: &Nat) -> Result<u32, String> {
    match nat {
        Nat::Lit(val) => {
            u32::try_from(*val).map_err(|_| format!("dimension literal {val} exceeds u32::MAX"))
        }
        Nat::Ident(ident) => Err(format!(
            "unknown identifier in dimension expression: {}",
            ident.name
        )),
        Nat::BinOp(op, lhs, rhs) => {
            let l = eval_nat(lhs)?;
            let r = eval_nat(rhs)?;
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

pub fn eval_dim(dim: &descend::ast::Dim) -> Result<(u32, u32, u32), String> {
    Ok(match dim {
        descend::ast::Dim::X(n) => (eval_nat(&n.0)?, 1, 1),
        descend::ast::Dim::Y(n) => (1, eval_nat(&n.0)?, 1),
        descend::ast::Dim::Z(n) => (1, 1, eval_nat(&n.0)?),
        descend::ast::Dim::XY(d) => (eval_nat(&d.0)?, eval_nat(&d.1)?, 1),
        descend::ast::Dim::XZ(d) => (eval_nat(&d.0)?, 1, eval_nat(&d.1)?),
        descend::ast::Dim::YZ(d) => (1, eval_nat(&d.0)?, eval_nat(&d.1)?),
        descend::ast::Dim::XYZ(d) => (eval_nat(&d.0)?, eval_nat(&d.1)?, eval_nat(&d.2)?),
    })
}
