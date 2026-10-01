//! Spelling Descend's execution and size expressions back out as source text.
//!
//! Data types are not handled here: they go through [`crate::host_ty`], which is the
//! single place that reads `DataTyKind`. What is left is the execution annotation and
//! the nat expressions inside it, neither of which has a host representation to decide.

use descend::ast::*;

pub fn parse_grid_dim(exec_kind: &ExecExprKind) -> Option<String> {
    if let BaseExec::GpuGrid(dim_x, dim_y) = &exec_kind.base {
        return Some(format!("{}, {}", format_dim(dim_x), format_dim(dim_y)));
    }
    None
}

pub fn format_dim(dim: &Dim) -> String {
    match dim {
        Dim::X(nat) => format!("X<{}>", format_nat(&nat.0)),
        Dim::Y(nat) => format!("Y<{}>", format_nat(&nat.0)),
        Dim::Z(nat) => format!("Z<{}>", format_nat(&nat.0)),
        Dim::XY(d) => format!("XY<{}, {}>", format_nat(&d.0), format_nat(&d.1)),
        Dim::XZ(d) => format!("XZ<{}, {}>", format_nat(&d.0), format_nat(&d.1)),
        Dim::YZ(d) => format!("YZ<{}, {}>", format_nat(&d.0), format_nat(&d.1)),
        Dim::XYZ(d) => format!(
            "XYZ<{}, {}, {}>",
            format_nat(&d.0),
            format_nat(&d.1),
            format_nat(&d.2)
        ),
    }
}

/// A nat expression as Descend source.
///
/// Descend's type checker compares sizes syntactically: `[i32; 8192]` does not unify
/// with `[i32; 64 * 128]`. So a size the macro passes back to Descend has to keep the
/// expression the kernel wrote, which is why this is spelled rather than evaluated.
///
/// Exhaustive on purpose: the kinds that depend on a running thread have no spelling in
/// a size, and a new `Nat` variant should be a compile error here rather than silently
/// becoming `?` in generated source.
pub(crate) fn format_nat(nat: &Nat) -> String {
    match nat {
        Nat::Lit(value) => value.to_string(),
        Nat::Ident(ident) => ident.name.to_string(),
        Nat::BinOp(op, lhs, rhs) => {
            format!("{} {} {}", format_nat(lhs), format_binop(op), format_nat(rhs))
        }
        Nat::App(ident, args) => {
            let args: Vec<String> = args.iter().map(format_nat).collect();
            format!("{}({})", ident.name, args.join(", "))
        }
        // These read the hardware's position at run time, so they are not sizes. They
        // cannot appear in a parameter type or an execution annotation.
        Nat::ThreadIdx(dim) => format!("thread_idx_{}", format_dim_compo(*dim)),
        Nat::BlockIdx(dim) => format!("block_idx_{}", format_dim_compo(*dim)),
        Nat::BlockDim(dim) => format!("block_dim_{}", format_dim_compo(*dim)),
        Nat::GridIdx => "grid_idx".to_string(),
        Nat::WarpGrpIdx => "warp_grp_idx".to_string(),
        Nat::WarpIdx => "warp_idx".to_string(),
        Nat::LaneIdx => "lane_idx".to_string(),
    }
}

fn format_binop(op: &BinOpNat) -> &'static str {
    match op {
        BinOpNat::Add => "+",
        BinOpNat::Sub => "-",
        BinOpNat::Mul => "*",
        BinOpNat::Div => "/",
        BinOpNat::Mod => "%",
    }
}

fn format_dim_compo(dim: DimCompo) -> &'static str {
    match dim {
        DimCompo::X => "x",
        DimCompo::Y => "y",
        DimCompo::Z => "z",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_size_is_spelled_as_descend_wrote_it() {
        assert_eq!(format_nat(&Nat::Lit(256)), "256");
        assert_eq!(format_nat(&Nat::Ident(Ident::new("N"))), "N");
        assert_eq!(
            format_nat(&Nat::BinOp(
                BinOpNat::Div,
                Box::new(Nat::Ident(Ident::new("N"))),
                Box::new(Nat::Lit(256)),
            )),
            "N / 256"
        );
    }

    #[test]
    fn every_dimensionality_is_spelled_out() {
        assert_eq!(format_dim(&Dim::X(Box::new(Dim1d(Nat::Lit(64))))), "X<64>");
        assert_eq!(
            format_dim(&Dim::XY(Box::new(Dim2d(Nat::Lit(4), Nat::Lit(2))))),
            "XY<4, 2>"
        );
        assert_eq!(
            format_dim(&Dim::XYZ(Box::new(Dim3d(
                Nat::Lit(2),
                Nat::Lit(3),
                Nat::Lit(4)
            )))),
            "XYZ<2, 3, 4>"
        );
    }
}
