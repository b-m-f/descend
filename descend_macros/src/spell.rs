use descend::ast::*;

pub(crate) fn format_nat(nat: &Nat) -> Result<String, String> {
    Ok(match nat {
        Nat::Lit(value) => value.to_string(),
        Nat::Ident(ident) => ident.name.to_string(),
        Nat::BinOp(op, lhs, rhs) => {
            format!(
                "{} {} {}",
                format_nat(lhs)?,
                format_binop(op),
                format_nat(rhs)?
            )
        }
        Nat::App(ident, arguments) => {
            let spelled_arguments: Vec<String> = arguments
                .iter()
                .map(format_nat)
                .collect::<Result<Vec<String>, String>>()?;
            format!("{}({})", ident.name, spelled_arguments.join(", "))
        }
        Nat::ThreadIdx(component) => {
            return Err(position_in_a_size(&format!(
                "thread_idx_{}",
                format_dimension_component(*component)
            )))
        }
        Nat::BlockIdx(component) => {
            return Err(position_in_a_size(&format!(
                "block_idx_{}",
                format_dimension_component(*component)
            )))
        }
        Nat::BlockDim(component) => {
            return Err(position_in_a_size(&format!(
                "block_dim_{}",
                format_dimension_component(*component)
            )))
        }
        Nat::GridIdx => return Err(position_in_a_size("grid_idx")),
        Nat::WarpGrpIdx => return Err(position_in_a_size("warp_grp_idx")),
        Nat::WarpIdx => return Err(position_in_a_size("warp_idx")),
        Nat::LaneIdx => return Err(position_in_a_size("lane_idx")),
    })
}

fn position_in_a_size(spelling: &str) -> String {
    format!(
        "a size cannot depend on `{spelling}`, which reads the hardware's position at \
         run time"
    )
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
        assert_eq!(format_nat(&Nat::Lit(256)).unwrap(), "256");
        assert_eq!(format_nat(&Nat::Ident(Ident::new("N"))).unwrap(), "N");
        assert_eq!(
            format_nat(&Nat::BinOp(
                BinOpNat::Div,
                Box::new(Nat::Ident(Ident::new("N"))),
                Box::new(Nat::Lit(256)),
            ))
            .unwrap(),
            "N / 256"
        );
    }

    #[test]
    fn a_runtime_position_is_refused_rather_than_misspelled() {
        let error = format_nat(&Nat::LaneIdx).unwrap_err();
        assert!(error.contains("lane_idx"), "{error}");
        assert!(error.contains("run time"), "{error}");

        let nested = format_nat(&Nat::BinOp(
            BinOpNat::Mul,
            Box::new(Nat::Lit(4)),
            Box::new(Nat::ThreadIdx(DimCompo::X)),
        ))
        .unwrap_err();
        assert!(nested.contains("thread_idx_x"), "{nested}");
    }
}
