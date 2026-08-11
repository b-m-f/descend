use descend::ast::*;

#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub is_uniq: bool,
    pub size_str: String,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub function_name: String,
    pub grid_dim_str: String,
    pub generic_params_str: String,
    pub params: Vec<KernelParam>,
}

impl Kernel {
    pub fn extract(compil_unit: &CompilUnit) -> Result<Self, String> {
        for item in &compil_unit.items {
            match item {
                Item::FunDef(f) => {
                    let function_name = f.ident.name.to_string();
                    // TODO: rethink the default 1,1?
                    let grid_dim_str =
                        parse_grid_dim(&f.exec.exec).unwrap_or_else(|| "X<1>, X<1>".to_string());

                    let mut generic_args = Vec::new();
                    for param in &f.generic_params {
                        if let IdentKinded {
                            kind: Kind::Nat,
                            ident,
                        } = param
                        {
                            generic_args.push(ident.name.to_string());
                        }
                    }
                    let generic_params_str = if !generic_args.is_empty() {
                        format!("::<{}>", generic_args.join(", "))
                    } else {
                        String::new()
                    };
                    // TODO: turn into debug
                    println!("Kernel: generic_params = {}", generic_params_str);

                    let mut params = Vec::new();

                    for param in &f.param_decls {
                        let p_name = &param.ident.name;
                        let size_str = extract_size_from_ty(param.ty.as_ref().unwrap());
                        let is_uniq = format!("{:?}", param.ty).contains("Uniq");

                        params.push(KernelParam {
                            name: p_name.to_string(),
                            is_uniq,
                            size_str,
                        });
                    }

                    return Ok(Kernel {
                        function_name,
                        grid_dim_str,
                        generic_params_str,
                        params,
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

impl std::fmt::Display for KernelParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let qual = if self.is_uniq { "uniq" } else { "shrd" };
        write!(f, "{}: {} [{}]", self.name, qual, self.size_str)
    }
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Kernel: {}{}",
            self.function_name, self.generic_params_str
        )?;
        writeln!(f, "  Grid Dimensions: {}", self.grid_dim_str)?;
        writeln!(f, "  Parameters:")?;
        for param in &self.params {
            writeln!(f, "    - {}", param)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_info_extraction() {
        let source_str = r#"
fn test_gpu_vec_add<r: prv>(
    a_array: &r uniq gpu.global [i32; 16384],
    b_array: &r shrd gpu.global [i32; 16384]
) -[grid: gpu.grid<X<64>, X<256>>]-> () {
    sched block in grid {
        sched thread in block {
            let a_ref = &uniq  (*a_array).to_view.grp::<256>[[block]][[thread]];
            *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
        }
    }
}
        "#;
        let source = descend::parser::SourceCode::new(source_str.to_string());
        let compilation_unit = descend::parser::parse(&source).unwrap();
        let kernel = Kernel::extract(&compilation_unit).unwrap();

        assert_eq!(kernel.function_name, "test_gpu_vec_add");
        assert_eq!(kernel.grid_dim_str, "X<64>, X<256>");
        assert_eq!(kernel.generic_params_str, "");
        assert_eq!(kernel.params.len(), 2);

        assert_eq!(kernel.params[0].name, "a_array");
        assert_eq!(kernel.params[0].is_uniq, true);
        assert_eq!(kernel.params[0].size_str, "16384");

        assert_eq!(kernel.params[1].name, "b_array");
        assert_eq!(kernel.params[1].is_uniq, false);
        assert_eq!(kernel.params[1].size_str, "16384");
    }
}
