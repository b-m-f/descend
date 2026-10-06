use crate::spell::{format_binop, format_nat};
use descend::ast::Nat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dimensions {
    fn along_x(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    fn along_y(y: u32) -> Self {
        Self { x: 1, y, z: 1 }
    }

    fn along_z(z: u32) -> Self {
        Self { x: 1, y: 1, z }
    }

    fn plane(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    fn depth(x: u32, z: u32) -> Self {
        Self { x, y: 1, z }
    }

    fn height(y: u32, z: u32) -> Self {
        Self { x: 1, y, z }
    }

    fn space(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

pub fn eval_nat(nat: &Nat) -> Result<u32, String> {
    match nat {
        Nat::Lit(value) => {
            u32::try_from(*value).map_err(|_| format!("the literal {value} exceeds u32::MAX"))
        }
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
        other => Err(match format_nat(other) {
            Ok(spelling) => format!("unsupported expression: {spelling}"),
            Err(reason) => reason,
        }),
    }
}

pub fn eval_dimensions(dimensions: &descend::ast::Dim) -> Result<Dimensions, String> {
    Ok(match dimensions {
        descend::ast::Dim::X(dim_1d) => Dimensions::along_x(eval_nat(&dim_1d.0)?),
        descend::ast::Dim::Y(dim_1d) => Dimensions::along_y(eval_nat(&dim_1d.0)?),
        descend::ast::Dim::Z(dim_1d) => Dimensions::along_z(eval_nat(&dim_1d.0)?),
        descend::ast::Dim::XY(dim_2d) => {
            Dimensions::plane(eval_nat(&dim_2d.0)?, eval_nat(&dim_2d.1)?)
        }
        descend::ast::Dim::XZ(dim_2d) => {
            Dimensions::depth(eval_nat(&dim_2d.0)?, eval_nat(&dim_2d.1)?)
        }
        descend::ast::Dim::YZ(dim_2d) => {
            Dimensions::height(eval_nat(&dim_2d.0)?, eval_nat(&dim_2d.1)?)
        }
        descend::ast::Dim::XYZ(dim_3d) => Dimensions::space(
            eval_nat(&dim_3d.0)?,
            eval_nat(&dim_3d.1)?,
            eval_nat(&dim_3d.2)?,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use descend::ast::{BinOpNat, Dim, Dim1d, Dim2d, Dim3d};

    /// The grid and block dimensions a kernel declares, as the macro derives them.
    /// This is the path `emit::generate_launch_function` takes to build a launch
    /// configuration, minus the code generation and the NVRTC compilation.
    fn dimensions_of(grid: Dim, block: Dim) -> (Dimensions, Dimensions) {
        (
            eval_dimensions(&grid).unwrap(),
            eval_dimensions(&block).unwrap(),
        )
    }

    fn x(size: usize) -> Dim {
        Dim::X(Box::new(Dim1d(Nat::Lit(size))))
    }

    #[test]
    fn one_dimensional_execution_fills_the_unused_axes_with_one() {
        let (grid, block) = dimensions_of(x(64), x(256));
        assert_eq!(grid, Dimensions::along_x(64));
        assert_eq!(block, Dimensions::along_x(256));
    }

    #[test]
    fn two_dimensional_execution_keeps_the_axis_order() {
        let (grid, block) = dimensions_of(
            Dim::XY(Box::new(Dim2d(Nat::Lit(4), Nat::Lit(2)))),
            Dim::XY(Box::new(Dim2d(Nat::Lit(16), Nat::Lit(8)))),
        );
        assert_eq!(grid, Dimensions::plane(4, 2));
        assert_eq!(block, Dimensions::plane(16, 8));
    }

    #[test]
    fn three_dimensional_execution_is_evaluated_on_every_axis() {
        let (grid, block) = dimensions_of(
            Dim::XYZ(Box::new(Dim3d(Nat::Lit(2), Nat::Lit(3), Nat::Lit(4)))),
            Dim::XYZ(Box::new(Dim3d(Nat::Lit(8), Nat::Lit(4), Nat::Lit(2)))),
        );
        assert_eq!(grid, Dimensions::space(2, 3, 4));
        assert_eq!(block, Dimensions::space(8, 4, 2));
    }

    #[test]
    fn arithmetic_in_a_dimension_is_evaluated() {
        // The shape a generic kernel takes once its size has been specialised.
        let grid = Dim::X(Box::new(Dim1d(Nat::BinOp(
            BinOpNat::Div,
            Box::new(Nat::Lit(4096)),
            Box::new(Nat::Lit(256)),
        ))));
        assert_eq!(eval_dimensions(&grid).unwrap(), Dimensions::along_x(16));
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
        assert!(eval_nat(&too_wide)
            .unwrap_err()
            .contains("exceeds u32::MAX"));
    }

    #[test]
    fn a_runtime_position_is_not_debug_printed_into_a_diagnostic() {
        let error = eval_nat(&Nat::LaneIdx).unwrap_err();
        assert!(error.contains("lane_idx"), "{error}");
        assert!(!error.contains("LaneIdx"), "{error}");
    }
}
