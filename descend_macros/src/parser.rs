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
