use crate::spell::{format_binop, format_nat};
use descend::ast::Nat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub fn eval_nat(nat: &Nat) -> Result<usize, String> {
    match nat {
        Nat::Lit(value) => Ok(*value),
        Nat::Ident(ident) => Err(format!("unknown identifier `{}`", ident.name)),
        Nat::BinOp(op, lhs, rhs) => {
            let left = eval_nat(lhs)?;
            let right = eval_nat(rhs)?;
            match op {
                descend::ast::BinOpNat::Add => left.checked_add(right),
                descend::ast::BinOpNat::Sub => left.checked_sub(right),
                descend::ast::BinOpNat::Mul => left.checked_mul(right),
                descend::ast::BinOpNat::Div => left.checked_div(right),
                descend::ast::BinOpNat::Mod => left.checked_rem(right),
            }
            .ok_or_else(|| {
                format!(
                    "invalid or overflowing expression: {left} {} {right}",
                    format_binop(op)
                )
            })
        }
        _ => Err(format!("a size cannot depend on `{}`", format_nat(nat))),
    }
}

pub fn eval_dimensions(dimensions: &descend::ast::Dim) -> Result<Dimensions, String> {
    let (x, y, z) = match dimensions {
        descend::ast::Dim::X(dim_1d) => (Some(&dim_1d.0), None, None),
        descend::ast::Dim::Y(dim_1d) => (None, Some(&dim_1d.0), None),
        descend::ast::Dim::Z(dim_1d) => (None, None, Some(&dim_1d.0)),
        descend::ast::Dim::XY(dim_2d) => (Some(&dim_2d.0), Some(&dim_2d.1), None),
        descend::ast::Dim::XZ(dim_2d) => (Some(&dim_2d.0), None, Some(&dim_2d.1)),
        descend::ast::Dim::YZ(dim_2d) => (None, Some(&dim_2d.0), Some(&dim_2d.1)),
        descend::ast::Dim::XYZ(dim_3d) => (Some(&dim_3d.0), Some(&dim_3d.1), Some(&dim_3d.2)),
    };

    Ok(Dimensions {
        x: eval_axis(x)?,
        y: eval_axis(y)?,
        z: eval_axis(z)?,
    })
}

fn eval_axis(axis: Option<&Nat>) -> Result<u32, String> {
    match axis {
        Some(nat) => {
            let value = eval_nat(nat)?;
            u32::try_from(value).map_err(|_| format!("the dimension {value} exceeds u32::MAX"))
        }
        None => Ok(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use descend::ast::{BinOpNat, Dim, Dim1d, Dim2d, Dim3d, Ident};

    fn dimensions(x: u32, y: u32, z: u32) -> Dimensions {
        Dimensions { x, y, z }
    }

    fn along_x(size: usize) -> Dim {
        Dim::X(Box::new(Dim1d(Nat::Lit(size))))
    }

    fn lit(value: usize) -> Nat {
        Nat::Lit(value)
    }

    fn binop(op: BinOpNat, lhs: Nat, rhs: Nat) -> Nat {
        Nat::BinOp(op, Box::new(lhs), Box::new(rhs))
    }

    fn dimensions_of(grid: Dim, block: Dim) -> (Dimensions, Dimensions) {
        (
            eval_dimensions(&grid).unwrap(),
            eval_dimensions(&block).unwrap(),
        )
    }

    #[test]
    fn one_dimensional_execution_fills_the_unused_axes_with_one() {
        let (grid, block) = dimensions_of(along_x(64), along_x(256));
        assert_eq!(grid, dimensions(64, 1, 1));
        assert_eq!(block, dimensions(256, 1, 1));
    }

    #[test]
    fn a_single_axis_execution_puts_its_size_on_the_axis_it_names() {
        let along_y = eval_dimensions(&Dim::Y(Box::new(Dim1d(Nat::Lit(8))))).unwrap();
        assert_eq!(along_y, dimensions(1, 8, 1));

        let along_z = eval_dimensions(&Dim::Z(Box::new(Dim1d(Nat::Lit(8))))).unwrap();
        assert_eq!(along_z, dimensions(1, 1, 8));
    }

    #[test]
    fn two_dimensional_execution_keeps_the_axis_order() {
        let (grid, block) = dimensions_of(
            Dim::XY(Box::new(Dim2d(Nat::Lit(4), Nat::Lit(2)))),
            Dim::XY(Box::new(Dim2d(Nat::Lit(16), Nat::Lit(8)))),
        );
        assert_eq!(grid, dimensions(4, 2, 1));
        assert_eq!(block, dimensions(16, 8, 1));
    }

    #[test]
    fn the_two_dimensional_executions_that_skip_an_axis_pad_it_with_one() {
        let xz = eval_dimensions(&Dim::XZ(Box::new(Dim2d(Nat::Lit(4), Nat::Lit(2))))).unwrap();
        assert_eq!(xz, dimensions(4, 1, 2));

        let yz = eval_dimensions(&Dim::YZ(Box::new(Dim2d(Nat::Lit(4), Nat::Lit(2))))).unwrap();
        assert_eq!(yz, dimensions(1, 4, 2));
    }

    #[test]
    fn three_dimensional_execution_is_evaluated_on_every_axis() {
        let (grid, block) = dimensions_of(
            Dim::XYZ(Box::new(Dim3d(Nat::Lit(2), Nat::Lit(3), Nat::Lit(4)))),
            Dim::XYZ(Box::new(Dim3d(Nat::Lit(8), Nat::Lit(4), Nat::Lit(2)))),
        );
        assert_eq!(grid, dimensions(2, 3, 4));
        assert_eq!(block, dimensions(8, 4, 2));
    }

    #[test]
    fn arithmetic_in_a_dimension_is_evaluated() {
        let grid = Dim::X(Box::new(Dim1d(Nat::BinOp(
            BinOpNat::Div,
            Box::new(Nat::Lit(4096)),
            Box::new(Nat::Lit(256)),
        ))));
        assert_eq!(eval_dimensions(&grid).unwrap(), dimensions(16, 1, 1));
    }

    #[test]
    fn an_unspecialised_size_is_reported_rather_than_guessed() {
        let error = eval_nat(&Nat::Ident(Ident::new("N"))).unwrap_err();
        assert!(error.contains("unknown identifier"), "{error}");
    }

    #[test]
    fn an_application_in_a_size_is_reported_rather_than_panicking() {
        let application = Nat::App(Ident::new("twice"), Box::new([Nat::Lit(2)]));
        let error = eval_nat(&application).unwrap_err();
        assert!(error.contains("a size cannot depend on"), "{error}");
        assert!(
            error.contains("twice(2)"),
            "the application should be spelled as Descend wrote it, not debug-printed: {error}"
        );
    }

    #[test]
    fn the_fold_agrees_with_descends_own_evaluator_where_both_are_defined() {
        let nat_context = descend::ast::NatCtx::new();
        for (expression, expected) in [
            (lit(256), 256),
            (binop(BinOpNat::Add, lit(200), lit(56)), 256),
            (binop(BinOpNat::Sub, lit(256), lit(56)), 200),
            (binop(BinOpNat::Mul, lit(16), lit(16)), 256),
            (binop(BinOpNat::Div, lit(4096), lit(256)), 16),
            (binop(BinOpNat::Mod, lit(260), lit(256)), 4),
            (
                binop(
                    BinOpNat::Mul,
                    binop(BinOpNat::Add, lit(2), lit(6)),
                    binop(BinOpNat::Div, lit(4096), lit(256)),
                ),
                128,
            ),
            (
                binop(
                    BinOpNat::Add,
                    binop(BinOpNat::Mul, lit(64), lit(64)),
                    binop(BinOpNat::Mod, lit(10), lit(4)),
                ),
                4098,
            ),
            (binop(BinOpNat::Mul, lit(65536), lit(65536)), 4_294_967_296),
        ] {
            assert_eq!(eval_nat(&expression).unwrap(), expected);
            assert_eq!(
                expression.eval(&nat_context).unwrap(),
                expected,
                "Descend's own evaluator disagrees with the macro's fold"
            );
        }
    }

    #[test]
    fn a_size_is_evaluated_in_the_width_descend_models_it_in() {
        let wider_than_u32 = binop(BinOpNat::Mul, lit(65536), lit(65536));
        assert_eq!(eval_nat(&wider_than_u32).unwrap(), 4_294_967_296);
    }

    #[test]
    fn a_subtraction_that_would_underflow_is_reported_rather_than_wrapped() {
        let error = eval_nat(&binop(BinOpNat::Sub, lit(4), lit(8))).unwrap_err();
        assert!(
            error.contains("invalid or overflowing expression: 4 - 8"),
            "{error}"
        );
    }

    #[test]
    fn a_division_by_zero_is_reported_rather_than_panicking() {
        for (expression, expected) in [
            (binop(BinOpNat::Div, lit(64), lit(0)), "64 / 0"),
            (binop(BinOpNat::Mod, lit(64), lit(0)), "64 % 0"),
        ] {
            let error = eval_nat(&expression).unwrap_err();
            assert!(
                error.contains("invalid or overflowing expression") && error.contains(expected),
                "{error}"
            );
        }
    }

    #[test]
    fn an_overflowing_expression_is_an_error() {
        let overflow = binop(BinOpNat::Mul, lit(usize::MAX), lit(2));
        let error = eval_nat(&overflow).unwrap_err();
        assert!(error.contains("overflowing"), "{error}");
        assert!(
            error.contains(" * "),
            "the operator should be spelled, not debug-printed: {error}"
        );
    }

    #[test]
    fn a_dimension_wider_than_a_device_takes_is_an_error() {
        let error = eval_dimensions(&along_x(u32::MAX as usize + 1)).unwrap_err();
        assert!(error.contains("exceeds u32::MAX"), "{error}");
    }

    #[test]
    fn a_runtime_position_is_not_debug_printed_into_a_diagnostic() {
        let error = eval_nat(&Nat::LaneIdx).unwrap_err();
        assert!(error.contains("lane_idx"), "{error}");
        assert!(!error.contains("LaneIdx"), "{error}");
    }
}
