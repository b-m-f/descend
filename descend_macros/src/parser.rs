use descend::ast::*;

pub fn parse_descend_source<'a>(
    source: &'a descend::parser::SourceCode,
) -> Result<CompilUnit<'a>, String> {
    descend::parser::parse(source)
        .map_err(|_| "Failed to parse descend macro input. Ensure syntax is valid.".to_string())
}

pub fn get_dim_size(d: &Dim) -> String {
    if let Dim::X(d1) = d {
        if let Nat::Lit(v) = &d1.0 {
            return v.to_string();
        } else if let Nat::Ident(id) = &d1.0 {
            return id.name.to_string();
        }
    }
    "1".to_string()
}

pub fn parse_grid_dim(exec_kind: &ExecExprKind) -> Option<String> {
    if let BaseExec::GpuGrid(dim_x, dim_y) = &exec_kind.base {
        let x_str = get_dim_size(dim_x);
        let y_str = get_dim_size(dim_y);
        return Some(format!("X<{}>, X<{}>", x_str, y_str));
    }
    None
}

pub fn extract_size_from_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        let inner_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
            &ref_dty.dty.dty
        } else {
            &dty.dty
        };
        if let DataTyKind::Array(_, nat) = inner_dty {
            if let Nat::Lit(v) = nat {
                return v.to_string();
            } else if let Nat::Ident(id) = nat {
                return id.name.to_string();
            }
        }
    }
    "1".to_string()
}
