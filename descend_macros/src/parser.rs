use descend::ast::*;
use std::collections::HashMap;

use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, LitInt, Token};

pub struct LaunchConfig {
    pub kernel_name: String,
    pub generic_to_size: HashMap<String, String>,
    pub array_args: Vec<Expr>,
}

impl Parse for LaunchConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let kernel_ident: Ident = input.parse()?;
        let kernel_name = kernel_ident.to_string();

        let mut generic_to_size = HashMap::new();

        // Parse optional generics: `<N: 16384, M = 10>`
        if input.peek(Token![<]) {
            input.parse::<Token![<]>()?;
            while !input.is_empty() && !input.peek(Token![>]) {
                let gen_ident: Ident = input.parse()?;

                if input.peek(Token![:]) {
                    input.parse::<Token![:]>()?;
                } else {
                    return Err(input.error("Expected ':' in generic argument"));
                }

                let val: LitInt = input.parse()?;
                generic_to_size.insert(gen_ident.to_string(), val.base10_digits().to_string());

                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
            }
            input.parse::<Token![>]>()?;
        }

        // Parse args
        let mut args = Vec::new();
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let expr: Expr = input.parse()?;
            args.push(expr);
        }

        Ok(LaunchConfig {
            kernel_name,
            generic_to_size,
            array_args: args,
        })
    }
}
pub fn parse_descend_source<'a>(
    source: &'a descend::parser::SourceCode,
) -> Result<CompilUnit<'a>, String> {
    descend::parser::parse(source)
        .map_err(|e| format!("Failed to parse descend macro input: {:?}", e))
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
                _ => "?",
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

pub fn extract_size_from_ty(ty: &Ty) -> String {
    if let TyKind::Data(dty) = &ty.ty {
        let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
            &ref_dty.dty.dty
        } else {
            &dty.dty
        };

        let mut dims = Vec::new();
        while let DataTyKind::Array(elem_ty, nat) = current_dty {
            dims.push(nat_to_string(nat));
            current_dty = &elem_ty.dty;
        }

        if !dims.is_empty() {
            return dims.join(" * ");
        }
    }
    "1".to_string()
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
        _ => "i32".to_string(),
    }
}

pub fn extract_byte_size_from_ty(ty: &Ty) -> u32 {
    if let TyKind::Data(dty) = &ty.ty {
        let mut current_dty = if let DataTyKind::Ref(ref_dty) = &dty.dty {
            &ref_dty.dty.dty
        } else {
            &dty.dty
        };

        while let DataTyKind::Array(elem_ty, _) = current_dty {
            current_dty = &elem_ty.dty;
        }

        if let DataTyKind::Scalar(scalar) = current_dty {
            return match scalar {
                descend::ast::ScalarTy::U8 | descend::ast::ScalarTy::Bool => 1,
                descend::ast::ScalarTy::U32
                | descend::ast::ScalarTy::I32
                | descend::ast::ScalarTy::F32 => 4,
                descend::ast::ScalarTy::U64
                | descend::ast::ScalarTy::I64
                | descend::ast::ScalarTy::F64 => 8,
                _ => 4,
            };
        }
    }
    4 // fallback
}

pub fn extract_call_args(source_str: &mut String) -> Vec<String> {
    let mut call_args = Vec::new();
    if let Some(call_idx) = source_str.find("call(") {
        let rest = &source_str[call_idx + 5..];
        if let Some(close_idx) = rest.find(')') {
            let args_str = &rest[..close_idx];
            for arg in args_str.split(',') {
                let trimmed = arg.trim();
                if !trimmed.is_empty() {
                    call_args.push(trimmed.to_string());
                }
            }
        }
        source_str.truncate(call_idx);
    }
    call_args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_call_args() {
        let mut source = "fn foo() {} call(a_array, b_array);".to_string();
        let args = extract_call_args(&mut source);
        assert_eq!(source, "fn foo() {} ");
        assert_eq!(args, vec!["a_array", "b_array"]);
    }
}
