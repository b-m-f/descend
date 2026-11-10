use descend::ast::*;

pub(crate) fn format_nat(nat: &Nat) -> String {
    match nat {
        Nat::Lit(value) => value.to_string(),
        Nat::Ident(ident) => ident.name.to_string(),
        Nat::BinOp(op, lhs, rhs) => {
            format!(
                "{} {} {}",
                format_nat(lhs),
                format_binop(op),
                format_nat(rhs)
            )
        }
        Nat::App(ident, arguments) => {
            let spelled_arguments: Vec<String> = arguments.iter().map(format_nat).collect();
            format!("{}({})", ident.name, spelled_arguments.join(", "))
        }
        Nat::ThreadIdx(component) => {
            format!("thread_idx_{}", format_dimension_component(*component))
        }
        Nat::BlockIdx(component) => {
            format!("block_idx_{}", format_dimension_component(*component))
        }
        Nat::BlockDim(component) => {
            format!("block_dim_{}", format_dimension_component(*component))
        }
        Nat::GridIdx => "grid_idx".to_string(),
        Nat::WarpGrpIdx => "warp_grp_idx".to_string(),
        Nat::WarpIdx => "warp_idx".to_string(),
        Nat::LaneIdx => "lane_idx".to_string(),
    }
}

pub(crate) fn format_binop(op: &BinOpNat) -> &'static str {
    match op {
        BinOpNat::Add => "+",
        BinOpNat::Sub => "-",
        BinOpNat::Mul => "*",
        BinOpNat::Div => "/",
        BinOpNat::Mod => "%",
    }
}

fn format_dimension_component(component: DimCompo) -> &'static str {
    match component {
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
    fn an_application_is_spelled_with_its_arguments() {
        let application = Nat::App(
            Ident::new("twice"),
            Box::new([Nat::Lit(2), Nat::Ident(Ident::new("N"))]),
        );
        assert_eq!(format_nat(&application), "twice(2, N)");
    }

    #[test]
    fn a_runtime_position_is_spelled_as_the_hardware_names_it() {
        assert_eq!(format_nat(&Nat::LaneIdx), "lane_idx");
        assert_eq!(format_nat(&Nat::GridIdx), "grid_idx");
        assert_eq!(format_nat(&Nat::WarpIdx), "warp_idx");
        assert_eq!(format_nat(&Nat::WarpGrpIdx), "warp_grp_idx");
        assert_eq!(
            format_nat(&Nat::ThreadIdx(DimCompo::X)),
            "thread_idx_x"
        );
        assert_eq!(format_nat(&Nat::BlockIdx(DimCompo::Y)), "block_idx_y");
        assert_eq!(format_nat(&Nat::BlockDim(DimCompo::Z)), "block_dim_z");
    }
}
