#![allow(unreachable_patterns)]
use descend::ast::*;

pub fn parse_grid_dim(exec_kind: &ExecExprKind) -> Option<String> {
    if let BaseExec::GpuGrid(dim_x, dim_y) = &exec_kind.base {
        let x_str = format_dim(dim_x);
        let y_str = format_dim(dim_y);
        return Some(format!("{}, {}", x_str, y_str));
    }
    None
}

pub fn format_dim(dim: &descend::ast::Dim) -> String {
    match dim {
        descend::ast::Dim::X(nat) => format!("X<{}>", nat_to_string(&nat.0)),
        descend::ast::Dim::Y(nat) => format!("Y<{}>", nat_to_string(&nat.0)),
        descend::ast::Dim::Z(nat) => format!("Z<{}>", nat_to_string(&nat.0)),
        descend::ast::Dim::XY(dim2d) => format!(
            "XY<{}, {}>",
            nat_to_string(&dim2d.0),
            nat_to_string(&dim2d.1)
        ),
        descend::ast::Dim::XZ(dim2d) => format!(
            "XZ<{}, {}>",
            nat_to_string(&dim2d.0),
            nat_to_string(&dim2d.1)
        ),
        descend::ast::Dim::YZ(dim2d) => format!(
            "YZ<{}, {}>",
            nat_to_string(&dim2d.0),
            nat_to_string(&dim2d.1)
        ),
        descend::ast::Dim::XYZ(dim3d) => format!(
            "XYZ<{}, {}, {}>",
            nat_to_string(&dim3d.0),
            nat_to_string(&dim3d.1),
            nat_to_string(&dim3d.2)
        ),
    }
}

fn nat_to_string(nat: &Nat) -> String {
    match nat {
        Nat::Lit(v) => v.to_string(),
        Nat::Ident(id) => id.name.to_string(),
        Nat::BinOp(op, n1, n2) => {
            let op_str = match op {
                descend::ast::BinOpNat::Add => "+",
                descend::ast::BinOpNat::Sub => "-",
                descend::ast::BinOpNat::Mul => "*",
                descend::ast::BinOpNat::Div => "/",
                descend::ast::BinOpNat::Mod => "%",
            };
            format!("{} {} {}", nat_to_string(n1), op_str, nat_to_string(n2))
        }
        Nat::App(id, args) => {
            let args_str: Vec<String> = args.iter().map(nat_to_string).collect();
            format!("{}({})", id.name, args_str.join(", "))
        }
        _ => "?".to_string(),
    }
}

pub fn format_cpu_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        if let DataTyKind::Ref(ref_dty) = &dty.dty {
            let own_str = match ref_dty.own {
                descend::ast::Ownership::Shrd => "shrd",
                descend::ast::Ownership::Uniq => "uniq",
            };
            let inner_ty_str = format_array_dty(&ref_dty.dty);
            return format!("&'a {} cpu.mem {}", own_str, inner_ty_str);
        }
    }
    "UNKNOWN_TY".to_string()
}

pub fn format_value_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        if !matches!(dty.dty, DataTyKind::Ref(_)) {
            return format_array_dty(dty);
        }
    }
    String::new()
}

pub fn format_shared_array_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        if let DataTyKind::Ref(ref_dty) = &dty.dty {
            return format_array_dty(&ref_dty.dty);
        }
    }
    "UNKNOWN_ARRAY".to_string()
}

pub fn format_array_dty(dty: &DataTy) -> String {
    match &dty.dty {
        DataTyKind::Array(inner, nat) => {
            let inner_str = format_array_dty(inner);
            let size_str = nat_to_string(nat);
            format!("[{}; {}]", inner_str, size_str)
        }
        DataTyKind::Scalar(scalar) => match scalar {
            descend::ast::ScalarTy::U8 => "u8",
            descend::ast::ScalarTy::U32 => "u32",
            descend::ast::ScalarTy::U64 => "u64",
            descend::ast::ScalarTy::I32 => "i32",
            descend::ast::ScalarTy::I64 => "i64",
            descend::ast::ScalarTy::F32 => "f32",
            descend::ast::ScalarTy::F64 => "f64",
            descend::ast::ScalarTy::Bool => "bool",
            _ => "i32",
        }
        .to_string(),
        DataTyKind::Ident(ident) => ident.name.to_string(),
        DataTyKind::Struct(s) => s.ident.name.to_string(),
        DataTyKind::At(dty, _) => format_array_dty(dty),
        _ => "i32".to_string(),
    }
}

pub fn format_dty_eval(dty: &DataTy) -> Result<String, String> {
    match &dty.dty {
        DataTyKind::Array(inner, nat) => {
            let size = crate::nat::eval_nat(nat)?;
            Ok(format!("[{}; {}]", format_dty_eval(inner)?, size))
        }
        DataTyKind::At(inner, _) => format_dty_eval(inner),
        _ => Ok(format_array_dty(dty)),
    }
}

pub fn format_base_ty(ty: &Ty) -> Option<String> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
        &ref_dty.dty.dty
    } else {
        &dty.dty
    };
    while let DataTyKind::Array(elem_ty, _) = current_dty {
        current_dty = &elem_ty.dty;
    }
    match current_dty {
        DataTyKind::Scalar(scalar) => match scalar {
            ScalarTy::U8 => Some("u8"),
            ScalarTy::U32 => Some("u32"),
            ScalarTy::U64 => Some("u64"),
            ScalarTy::I32 => Some("i32"),
            ScalarTy::I64 => Some("i64"),
            ScalarTy::F32 => Some("f32"),
            ScalarTy::F64 => Some("f64"),
            ScalarTy::Bool => Some("bool"),
            ScalarTy::Unit | ScalarTy::Gpu => None,
        }
        .map(str::to_string),
        DataTyKind::Ident(ident) => Some(ident.name.to_string()),
        DataTyKind::Struct(s) => Some(s.ident.name.to_string()),
        _ => None,
    }
}

pub fn elem_byte_size(ty: &Ty) -> Option<u32> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
        &ref_dty.dty.dty
    } else {
        &dty.dty
    };
    while let DataTyKind::Array(elem_ty, _) = current_dty {
        current_dty = &elem_ty.dty;
    }
    match current_dty {
        DataTyKind::Scalar(ScalarTy::Bool) => Some(1),
        DataTyKind::Scalar(ScalarTy::U32 | ScalarTy::I32 | ScalarTy::F32) => Some(4),
        DataTyKind::Atomic(descend::ast::AtomicTy::AtomicU32) => Some(4),
        DataTyKind::Scalar(ScalarTy::U64 | ScalarTy::I64 | ScalarTy::F64) => Some(8),
        _ => None,
    }
}

pub fn extract_dim_nats(ty: &Ty) -> Vec<Nat> {
    let TyKind::Data(dty) = &ty.ty else {
        return Vec::new();
    };
    let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
        &ref_dty.dty.dty
    } else {
        &dty.dty
    };
    let mut dims = Vec::new();
    while let DataTyKind::Array(elem_ty, nat) = current_dty {
        dims.push(nat.clone());
        current_dty = &elem_ty.dty;
    }
    dims
}
