use crate::parsing::DescendSource;
use crate::types::*;
use descend::ast::*;
use proc_macro2::Span;

#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub span: Option<Span>,
    pub is_uniq: bool,
    pub is_shared_mem: bool,
    pub is_by_value: bool,
    pub dim_nats: Vec<Nat>,
    pub elem_bytes: Option<u32>,
    pub elem_base_ty: Option<String>,
    pub cpu_ty_str: String,
    pub value_ty_str: String,
    pub value_dty: Option<DataTy>,
    pub shared_ty_str: String,
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
                    let elem_base_ty = crate::types::format_base_ty(param.ty.as_ref().unwrap());
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
                    let cpu_ty_str = crate::types::format_cpu_ty(param.ty.as_ref().unwrap());
                    let value_ty_str = crate::types::format_value_ty(param.ty.as_ref().unwrap());
                    let value_dty = match &param.ty.as_ref().unwrap().ty {
                        TyKind::Data(dty) if !matches!(dty.dty, DataTyKind::Ref(_)) => {
                            Some(dty.as_ref().clone())
                        }
                        _ => None,
                    };
                    let shared_ty_str =
                        crate::types::format_shared_array_ty(param.ty.as_ref().unwrap());

                    params.push(KernelParam {
                        name: p_name.to_string(),
                        span: span_of(&param.ident, source),
                        is_uniq,
                        is_shared_mem,
                        is_by_value,
                        dim_nats: crate::types::extract_dim_nats(param.ty.as_ref().unwrap()),
                        elem_bytes: crate::types::elem_byte_size(param.ty.as_ref().unwrap()),
                        elem_base_ty,
                        cpu_ty_str,
                        value_ty_str,
                        value_dty,
                        shared_ty_str,
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
