use crate::parser::*;
use descend::ast::*;

/// Representation of a GPU kernel parameter.
#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub is_uniq: bool,
    pub is_shared_mem: bool,
    /// A parameter that is not a reference: the kernel takes its value, so the
    /// host passes the value itself rather than a device address.
    pub is_by_value: bool,
    pub size_str: String,
    /// The array dimensions as `Nat`s, for evaluation against the launch's
    /// generic bindings. Empty for a non-array parameter.
    pub dim_nats: Vec<Nat>,
    /// The element type's byte size as the Descend code generator defines it,
    /// or `None` when it defines none.
    pub elem_bytes: Option<u32>,
    pub elem_base_ty: Option<String>,
    pub cpu_ty_str: String,
    /// The parameter's own type, for a parameter taken by value: `i32`,
    /// `[i32; 3]`, a struct name. Empty for a reference parameter.
    pub value_ty_str: String,
    pub shared_ty_str: String,
}

/// Intermediate Representation for a specific GPU kernel.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub function_name: String,
    /// The names of the kernel's `nat` generic parameters. A kernel with none
    /// can be compiled where it is defined; a kernel with some cannot, because
    /// its sizes only become literals at a launch site.
    pub nat_generics: Vec<String>,
    pub grid_dim: Option<descend::ast::Dim>,
    pub block_dim: Option<descend::ast::Dim>,
    pub grid_dim_str: String,
    pub params: Vec<KernelParam>,
}

/// Whether a Descend function is a kernel that can be launched, i.e. one that
/// executes over a GPU grid. A function executing on a thread or on the host is
/// callable from Descend code but is not an entry point.
pub fn is_launchable(f: &FunDef) -> bool {
    matches!(f.exec.exec.base, BaseExec::GpuGrid(_, _))
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

                let mut grid_dim = None;
                let mut block_dim = None;
                if let descend::ast::BaseExec::GpuGrid(g, b) = &f.exec.exec.base {
                    grid_dim = Some(g.clone());
                    block_dim = Some(b.clone());
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
                let mut params = Vec::new();

                for param in &f.param_decls {
                    let p_name = &param.ident.name;
                    let size_str = extract_size_from_ty(param.ty.as_ref().unwrap());
                    let elem_base_ty = crate::parser::format_base_ty(param.ty.as_ref().unwrap());
                    // Read ownership and memory kind from the type itself. A
                    // `Debug`-string search would also match a nested type that
                    // merely mentions them, and both flags decide how the
                    // parameter is passed to the device.
                    let (is_uniq, is_shared_mem, is_by_value) = match &param.ty.as_ref().unwrap().ty
                    {
                        TyKind::Data(dty) => match &dty.dty {
                            DataTyKind::Ref(ref_dty) => (
                                matches!(ref_dty.own, Ownership::Uniq),
                                matches!(ref_dty.mem, Memory::GpuShared),
                                false,
                            ),
                            _ => (false, false, true),
                        },
                        _ => (false, false, true),
                    };
                    let cpu_ty_str = crate::parser::format_cpu_ty(param.ty.as_ref().unwrap());
                    let value_ty_str = crate::parser::format_value_ty(param.ty.as_ref().unwrap());
                    let shared_ty_str =
                        crate::parser::format_shared_array_ty(param.ty.as_ref().unwrap());

                    params.push(KernelParam {
                        name: p_name.to_string(),
                        is_uniq,
                        is_shared_mem,
                        is_by_value,
                        size_str,
                        dim_nats: crate::parser::extract_dim_nats(param.ty.as_ref().unwrap()),
                        elem_bytes: crate::parser::elem_byte_size(param.ty.as_ref().unwrap()),
                        elem_base_ty,
                        cpu_ty_str,
                        value_ty_str,
                        shared_ty_str,
                    });
                }

                return Ok(Kernel {
                    function_name,
                    nat_generics: generic_args,
                    grid_dim,
                    block_dim,
                    grid_dim_str,
                    params,
                });
            }
        }

        Err(format!(
            "Kernel definition '{}' not found in macro body",
            target_name
        ))
    }
}
