#![allow(dead_code)]
use crate::parser::*;
use descend::ast::*;

/// Representation of a GPU kernel parameter.
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

/// Intermediate Representation for a specific GPU kernel.
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
    /// Extracts Kernel IR for a given target function name from the parsed compilation unit.
    /// 
    /// What it does: Scans the Descend AST for the `FunDef` matching `target_name`.
    /// Why it is needed: Extracts specifically requested FunDef to compile to PTX and bind to the Rust invocation.
    pub fn extract(compil_unit: &CompilUnit, target_name: &str) -> Result<Self, String> {
        for item in &compil_unit.items {
            if let Item::FunDef(f) = item {
                let function_name = f.ident.name.to_string();
                if function_name != target_name {
                    continue;
                }
                
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
        }

        Err(format!("Kernel definition '{}' not found in macro body", target_name))
    }

    /// Calculates total grid size statically if possible, otherwise defaults to 16384.
    /// 
    /// What it does: Computes grid limits mathematically.
    /// Why it is needed: Determines exact sizes for host orchestrator data movement buffering.
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
