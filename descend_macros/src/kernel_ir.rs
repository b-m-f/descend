#![allow(dead_code)]
use crate::parser::*;
use descend::ast::*;

#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub is_uniq: bool,
    pub is_shared_mem: bool,
    pub size_str: String,
    pub elem_byte_size: u32,
    pub cpu_ty_str: String,
    pub shared_ty_str: String,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub function_name: String,
    pub grid_dim_str: String,
    pub generic_params_str: String,
    pub generic_size_bindings: Vec<(String, String)>,
    pub params: Vec<KernelParam>,
    pub call_args: Vec<String>,
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
                    debug!("Kernel: generic_params = {}", generic_params_str);

                    let mut params = Vec::new();

                    for param in &f.param_decls {
                        let p_name = &param.ident.name;
                        let size_str = extract_size_from_ty(param.ty.as_ref().unwrap());
                        let elem_byte_size = extract_byte_size_from_ty(param.ty.as_ref().unwrap());
                        let ty_debug = format!("{:?}", param.ty);
                        let is_uniq = ty_debug.contains("Uniq");
                        let is_shared_mem = ty_debug.contains("GpuShared");
                        let cpu_ty_str = crate::parser::format_cpu_ty(param.ty.as_ref().unwrap());
                        let shared_ty_str =
                            crate::parser::format_shared_array_ty(param.ty.as_ref().unwrap());

                        params.push(KernelParam {
                            name: p_name.to_string(),
                            is_uniq,
                            is_shared_mem,
                            size_str,
                            elem_byte_size,
                            cpu_ty_str,
                            shared_ty_str,
                        });
                    }

                    let mut generic_size_bindings = Vec::new();
                    for generic_arg in &generic_args {
                        for param in &params {
                            if param.size_str == *generic_arg {
                                generic_size_bindings
                                    .push((generic_arg.clone(), param.name.clone()));
                            }
                        }
                    }

                    return Ok(Kernel {
                        function_name,
                        grid_dim_str,
                        generic_params_str,
                        params,
                        generic_size_bindings,
                        call_args: Vec::new(),
                    });
                }
                Item::FunDecl(_) | Item::StructDecl(_) => {}
            }
        }

        Err("No function definition found in descend! macro".to_string())
    }

    pub fn grid_total_size(&self) -> String {
        let mut product = 1usize;
        let mut found_any = false;

        for part in self.grid_dim_str.split(',') {
            let s = part.trim();
            if let Some(inner) = s.strip_prefix("X<").and_then(|s| s.strip_suffix('>')) {
                if let Ok(val) = inner.parse::<usize>() {
                    product *= val;
                    found_any = true;
                }
            }
        }

        if found_any {
            product.to_string()
        } else {
            "16384".to_string()
        }
    }
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
        if !self.generic_size_bindings.is_empty() {
            writeln!(f, "  Bindings:")?;
            for (gen, param_name) in &self.generic_size_bindings {
                writeln!(f, "    - {} inferred from {}", gen, param_name)?;
            }
        }
        if !self.call_args.is_empty() {
            writeln!(f, "  Call Args: call({})", self.call_args.join(", "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_ir_extraction() {
        let source_str = r#"
fn test_gpu_vec_add<r: prv>(
    a_array: &r uniq gpu.global [i64; 16384],
    b_array: &r shrd gpu.global [i64; 16384]
) -[grid: gpu.grid<X<1>, X<128>>]-> () {
    sched block in grid {
        sched thread in block {
            let a_ref = &uniq  (*a_array).to_view.grp::<256>[[block]][[thread]];
            *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
        }
    }
}
        "#;
        let source = descend::parser::SourceCode::new(source_str.to_string());
        let compilation_unit = parse_descend_source(&source).unwrap();
        let kernel = Kernel::extract(&compilation_unit).unwrap();

        assert_eq!(kernel.function_name, "test_gpu_vec_add");
        assert_eq!(kernel.grid_dim_str, "X<1>, X<128>");
        assert_eq!(kernel.generic_params_str, "");
        assert_eq!(kernel.params.len(), 2);

        assert_eq!(kernel.params[0].name, "a_array");
        assert_eq!(kernel.params[0].is_uniq, true);
        assert_eq!(kernel.params[0].size_str, "16384");

        assert_eq!(kernel.params[1].name, "b_array");
        assert_eq!(kernel.params[1].is_uniq, false);
        assert_eq!(kernel.params[1].size_str, "16384");

        assert_eq!(kernel.generic_size_bindings.len(), 0);
    }
    #[test]
    fn test_kernel_ir_extraction_generic() {
        let source_str = r#"
fn test_gpu_vec_add<N: nat, r: prv>(
    a_array: &r uniq gpu.global [i32; N],
    b_array: &r shrd gpu.global [i32; N]
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
        let compilation_unit = parse_descend_source(&source).unwrap();
        let kernel = Kernel::extract(&compilation_unit).unwrap();

        assert_eq!(kernel.function_name, "test_gpu_vec_add");
        assert_eq!(kernel.grid_dim_str, "X<64>, X<256>");
        assert_eq!(kernel.generic_params_str, "::<N>");
        assert_eq!(kernel.params.len(), 2);

        assert_eq!(kernel.params[0].name, "a_array");
        assert_eq!(kernel.params[0].is_uniq, true);
        assert_eq!(kernel.params[0].size_str, "N");

        assert_eq!(kernel.params[1].name, "b_array");
        assert_eq!(kernel.params[1].is_uniq, false);
        assert_eq!(kernel.params[1].size_str, "N");

        assert_eq!(kernel.generic_size_bindings.len(), 2);
        assert_eq!(kernel.generic_size_bindings[0].0, "N");
        assert_eq!(kernel.generic_size_bindings[0].1, "a_array");
        assert_eq!(kernel.generic_size_bindings[1].0, "N");
        assert_eq!(kernel.generic_size_bindings[1].1, "b_array");
    }
}
