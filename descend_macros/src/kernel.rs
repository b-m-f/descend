use crate::host_ty::{self, HostTy};
use crate::source::DescendSource;
use descend::ast::*;
use proc_macro2::Span;

#[derive(Debug, Clone)]
pub struct KernelParameter {
    pub name: String,
    pub span: Option<Span>,
    pub is_uniq: bool,
    pub is_shared_memory: bool,
    pub is_by_value: bool,
    /// The data this parameter names -- what a reference points at, or the value
    /// itself. Both the Rust signature and the synthesized Descend `main` are spelled
    /// from this, so the two cannot drift apart.
    pub host: HostTy,
}

#[derive(Debug, Clone)]
pub struct ExecutionDimensions {
    pub grid: Dim,
    pub block: Dim,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub function_name: String,
    pub name_span: Option<Span>,
    pub dimensions: ExecutionDimensions,
    pub parameters: Vec<KernelParameter>,
}

fn span_of(ident: &Ident, source: &DescendSource) -> Option<Span> {
    source.span_at(ident.span?.begin)
}

pub fn is_launchable(fun_def: &FunDef) -> bool {
    matches!(fun_def.exec.exec.base, BaseExec::GpuGrid(_, _))
}

impl Kernel {
    pub fn extract(
        compilation_unit: &CompilUnit,
        kernel_name: &str,
        source: &DescendSource,
    ) -> Result<Self, String> {
        for item in &compilation_unit.items {
            if let Item::FunDef(fun_def) = item {
                let function_name = fun_def.ident.name.to_string();
                if function_name != kernel_name {
                    continue;
                }

                let BaseExec::GpuGrid(grid, block) = &fun_def.exec.exec.base else {
                    return Err(format!(
                        "kernel `{function_name}` must declare explicit grid and block \
                         dimensions, as `-[grid: gpu.grid<X<..>, X<..>>]->`"
                    ));
                };
                let dimensions = ExecutionDimensions {
                    grid: grid.clone(),
                    block: block.clone(),
                };

                let mut parameters = Vec::new();

                for parameter in &fun_def.param_decls {
                    let parameter_name = &parameter.ident.name;
                    let ty = parameter.ty.as_ref().unwrap();

                    // A parameter is either a reference to memory or a value, and that
                    // decides both how it reaches the kernel and how the host holds it.
                    let (is_uniq, is_shared_memory, is_by_value) = match host_ty::reference(ty) {
                        Some(reference) => (
                            matches!(reference.own, Ownership::Uniq),
                            matches!(reference.mem, Memory::GpuShared),
                            false,
                        ),
                        None => (false, false, true),
                    };

                    let dty = host_ty::parameter_dty(ty).ok_or_else(|| {
                        format!(
                            "kernel `{function_name}`: parameter `{parameter_name}` has no data type"
                        )
                    })?;
                    let host = host_ty::host_ty(dty).map_err(|reason| {
                        format!(
                            "kernel `{function_name}`: parameter `{parameter_name}` {}",
                            reason.reason()
                        )
                    })?;

                    parameters.push(KernelParameter {
                        name: parameter_name.to_string(),
                        span: span_of(&parameter.ident, source),
                        is_uniq,
                        is_shared_memory,
                        is_by_value,
                        host,
                    });
                }

                return Ok(Kernel {
                    function_name,
                    name_span: span_of(&fun_def.ident, source),
                    dimensions,
                    parameters,
                });
            }
        }

        Err(format!(
            "Kernel definition '{kernel_name}' not found in macro body"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::TokenStream;
    use std::str::FromStr;

    fn extract(source_text: &str, kernel_name: &str) -> Result<Kernel, String> {
        let source = crate::source::DescendSource::from_token_stream(
            TokenStream::from_str(source_text).expect("must tokenize"),
        );
        let code = descend::parser::SourceCode::new(source.text.clone());
        let unit = crate::source::parse_descend_source(&code)
            .unwrap_or_else(|error| panic!("fixture did not parse: {error}\n{}", source.text));
        Kernel::extract(&unit, kernel_name, &source)
    }

    #[test]
    fn a_gpu_grid_kernel_keeps_both_declared_dimensions() {
        let kernel = extract(
            "fn gpu_k<r: prv>(a: &r uniq gpu.global [i32; 256])
                 -[grid: gpu.grid<X<4>, X<64>>]-> () {
                 sched block in grid { sched thread in block { () } }
             }",
            "gpu_k",
        )
        .expect("a gpu.grid kernel must extract");

        assert_eq!(
            crate::nat::eval_dimensions(&kernel.dimensions.grid).unwrap(),
            crate::nat::Dimensions { x: 4, y: 1, z: 1 }
        );
        assert_eq!(
            crate::nat::eval_dimensions(&kernel.dimensions.block).unwrap(),
            crate::nat::Dimensions { x: 64, y: 1, z: 1 }
        );
    }

    #[test]
    fn a_function_without_grid_dimensions_is_refused_rather_than_defaulted() {
        let error = extract(
            "fn cpu_only(a: i32) -[t: cpu.thread]-> i32 { a + 1 }",
            "cpu_only",
        )
        .expect_err("a cpu.thread function has no grid to launch");

        assert!(
            error.contains("must declare explicit grid and block dimensions"),
            "{error}"
        );
    }
}
