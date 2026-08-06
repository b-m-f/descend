use descend::ast::*;
use quote::quote;

pub struct KernelInfo {
    pub fn_name: String,
    pub grid_dim_str: String,
    pub gen_params_str: String,
    pub main_vars: Vec<String>,
    pub main_args: Vec<String>,
    pub main_params: Vec<String>,
    pub rust_params: Vec<proc_macro2::TokenStream>,
    pub rust_args: Vec<proc_macro2::TokenStream>,
    pub arg_idents: Vec<syn::Ident>,
}

impl KernelInfo {
    pub fn extract(compil_unit: &CompilUnit) -> Result<Self, String> {
        for item in &compil_unit.items {
            match item {
                Item::FunDef(f) => {
                    let fn_name = f.ident.name.to_string();
                    let grid_dim_str = parse_grid_dim(&f.exec.exec).unwrap_or_else(|| "X<1>, X<1>".to_string());

                    let mut gen_args = Vec::new();
                    for param in &f.generic_params {
                        if let IdentKinded {
                            kind: Kind::Nat,
                            ident,
                        } = param
                        {
                            gen_args.push(ident.name.to_string());
                        }
                    }
                    let gen_params_str = if !gen_args.is_empty() {
                        format!("::<{}>", gen_args.join(", "))
                    } else {
                        String::new()
                    };

                    let mut main_vars = Vec::new();
                    let mut main_args = Vec::new();
                    let mut main_params = Vec::new();
                    let mut rust_params = Vec::new();
                    let mut rust_args = Vec::new();
                    let mut arg_idents = Vec::new();

                    for param in &f.param_decls {
                        let p_name = &param.ident.name;
                        let arg_ident = syn::Ident::new(p_name, proc_macro2::Span::call_site());
                        let size_str = extract_size_from_ty(param.ty.as_ref().unwrap());

                        let is_uniq = format!("{:?}", param.ty).contains("Uniq");
                        let (rgn, qual, c_qual) = if is_uniq {
                            ("uniq", "&mut", "*mut")
                        } else {
                            ("shrd", "&", "*const")
                        };

                        main_params.push(format!(
                            "h_{}: &'a {} cpu.mem [i32; {}]",
                            p_name, rgn, size_str
                        ));
                        main_vars.push(format!(
                            "let mut {} = gpu_alloc_copy(&uniq gpu, &shrd *h_{});",
                            p_name, p_name
                        ));
                        main_args.push(format!("&{} {}", rgn, p_name));

                        let rust_type: syn::Type =
                            syn::parse_str(&format!("{} gpu::GpuBuffer<i32>", qual)).unwrap();
                        let c_type: syn::Type = syn::parse_str(&format!("{} i32", c_qual)).unwrap();
                        rust_params.push(quote! { #arg_ident: #rust_type });
                        rust_args.push(quote! { #arg_ident.device_ptr as #c_type });
                        arg_idents.push(arg_ident.clone());
                    }

                    return Ok(KernelInfo {
                        fn_name,
                        grid_dim_str,
                        gen_params_str,
                        main_vars,
                        main_args,
                        main_params,
                        rust_params,
                        rust_args,
                        arg_idents,
                    });
                }
                Item::FunDecl(_) | Item::StructDecl(_) => {}
            }
        }

        Err("No function definition found in descend! macro".to_string())
    }
}

fn get_dim_size(d: &Dim) -> String {
    if let Dim::X(d1) = d {
        if let Nat::Lit(v) = &d1.0 {
            return v.to_string();
        } else if let Nat::Ident(id) = &d1.0 {
            return id.name.to_string();
        }
    }
    "1".to_string()
}

fn parse_grid_dim(exec_kind: &ExecExprKind) -> Option<String> {
    if let BaseExec::GpuGrid(dim_x, dim_y) = &exec_kind.base {
        let x_str = get_dim_size(dim_x);
        let y_str = get_dim_size(dim_y);
        return Some(format!("X<{}>, X<{}>", x_str, y_str));
    }
    None
}

fn extract_size_from_ty(ty: &Ty) -> String {
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
