use crate::host_ty::{self, HostTy};
use crate::parsing::DescendSource;
use crate::types::parse_grid_dim;
use descend::ast::*;
use proc_macro2::Span;

#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub span: Option<Span>,
    pub is_uniq: bool,
    pub is_shared_mem: bool,
    pub is_by_value: bool,
    /// The data this parameter names -- what a reference points at, or the value
    /// itself. Both the Rust signature and the synthesized Descend `main` are spelled
    /// from this, so the two cannot drift apart.
    pub host: HostTy,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub function_name: String,
    pub name_span: Option<Span>,
    pub grid_dim: Option<descend::ast::Dim>,
    pub block_dim: Option<descend::ast::Dim>,
    pub grid_dim_str: String,
    pub params: Vec<KernelParam>,
}

fn span_of(ident: &Ident, source: &DescendSource) -> Option<Span> {
    source.span_at(ident.span?.begin)
}

pub fn is_launchable(f: &FunDef) -> bool {
    matches!(f.exec.exec.base, BaseExec::GpuGrid(_, _))
}

impl Kernel {
    pub fn extract(
        compil_unit: &CompilUnit,
        target_name: &str,
        source: &DescendSource,
    ) -> Result<Self, String> {
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

                let mut params = Vec::new();

                for param in &f.param_decls {
                    let p_name = &param.ident.name;
                    let ty = param.ty.as_ref().unwrap();

                    // A parameter is either a reference to memory or a value, and that
                    // decides both how it reaches the kernel and how the host holds it.
                    let (is_uniq, is_shared_mem, is_by_value) = match host_ty::reference(ty) {
                        Some(reference) => (
                            matches!(reference.own, Ownership::Uniq),
                            matches!(reference.mem, Memory::GpuShared),
                            false,
                        ),
                        None => (false, false, true),
                    };

                    let dty = host_ty::parameter_dty(ty).ok_or_else(|| {
                        format!(
                            "kernel `{function_name}`: parameter `{p_name}` has no data type"
                        )
                    })?;
                    let host = host_ty::host_ty(dty).map_err(|reason| {
                        format!(
                            "kernel `{function_name}`: parameter `{p_name}` {}",
                            reason.reason()
                        )
                    })?;

                    params.push(KernelParam {
                        name: p_name.to_string(),
                        span: span_of(&param.ident, source),
                        is_uniq,
                        is_shared_mem,
                        is_by_value,
                        host,
                    });
                }

                return Ok(Kernel {
                    function_name,
                    name_span: span_of(&f.ident, source),
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
