use crate::spell::{format_binop, format_nat};
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
            .ok_or_else(|| {
                format!(
                    "invalid or overflowing dimension expression: {l} {} {r}",
                    format_binop(op)
                )
            })
        }
        other => Err(match format_nat(other) {
            Ok(spelling) => format!("unsupported dimension expression: {spelling}"),
            Err(reason) => reason,
        }),
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

#[cfg(test)]
mod tests {
    use super::*;
    use descend::ast::{BinOpNat, Dim, Dim1d, Dim2d, Dim3d};

    /// The grid and block dimensions a kernel declares, as the macro derives them.
    /// This is the path `emit::generate_launch_fn` takes to build a launch
    /// configuration, minus the code generation and nvcc.
    fn dims_of(grid: Dim, block: Dim) -> ((u32, u32, u32), (u32, u32, u32)) {
        (eval_dim(&grid).unwrap(), eval_dim(&block).unwrap())
    }

    fn x(n: usize) -> Dim {
        Dim::X(Box::new(Dim1d(Nat::Lit(n))))
    }

    #[test]
    fn one_dimensional_execution_fills_the_unused_axes_with_one() {
        let (grid, block) = dims_of(x(64), x(256));
        assert_eq!(grid, (64, 1, 1));
        assert_eq!(block, (256, 1, 1));
    }

    #[test]
    fn two_dimensional_execution_keeps_the_axis_order() {
        let (grid, block) = dims_of(
            Dim::XY(Box::new(Dim2d(Nat::Lit(4), Nat::Lit(2)))),
            Dim::XY(Box::new(Dim2d(Nat::Lit(16), Nat::Lit(8)))),
        );
        assert_eq!(grid, (4, 2, 1));
        assert_eq!(block, (16, 8, 1));
    }

    #[test]
    fn three_dimensional_execution_is_evaluated_on_every_axis() {
        let (grid, block) = dims_of(
            Dim::XYZ(Box::new(Dim3d(Nat::Lit(2), Nat::Lit(3), Nat::Lit(4)))),
            Dim::XYZ(Box::new(Dim3d(Nat::Lit(8), Nat::Lit(4), Nat::Lit(2)))),
        );
        assert_eq!(grid, (2, 3, 4));
        assert_eq!(block, (8, 4, 2));
    }

    #[test]
    fn arithmetic_in_a_dimension_is_evaluated() {
        // The shape a generic kernel takes once its size has been specialised.
        let grid = Dim::X(Box::new(Dim1d(Nat::BinOp(
            BinOpNat::Div,
            Box::new(Nat::Lit(4096)),
            Box::new(Nat::Lit(256)),
        ))));
        assert_eq!(eval_dim(&grid).unwrap(), (16, 1, 1));
    }

    #[test]
    fn an_unspecialised_size_is_reported_rather_than_guessed() {
        let error = eval_nat(&Nat::Ident(descend::ast::Ident::new("N"))).unwrap_err();
        assert!(error.contains("unknown identifier"), "{error}");
    }

    #[test]
    fn an_overflowing_dimension_is_an_error() {
        let overflow = Nat::BinOp(
            BinOpNat::Mul,
            Box::new(Nat::Lit(u32::MAX as usize)),
            Box::new(Nat::Lit(2)),
        );
        let error = eval_nat(&overflow).unwrap_err();
        assert!(error.contains("overflowing"), "{error}");
        assert!(
            error.contains(" * "),
            "the operator should be spelled, not debug-printed: {error}"
        );
    }

    #[test]
    fn a_dimension_literal_wider_than_u32_is_an_error() {
        let too_wide = Nat::Lit(u32::MAX as usize + 1);
        assert!(eval_nat(&too_wide).unwrap_err().contains("exceeds u32::MAX"));
    }

    #[test]
    fn a_runtime_position_is_not_debug_printed_into_a_diagnostic() {
        let error = eval_nat(&Nat::LaneIdx).unwrap_err();
        assert!(error.contains("lane_idx"), "{error}");
        assert!(!error.contains("LaneIdx"), "{error}");
    }
}
