use crate::diag::MacroError;
use crate::host_ty::{from_dty, HostTy};
use crate::source::DescendSource;
use descend::ast::*;
use proc_macro2::Span;

#[derive(Debug, Clone)]
pub struct KernelParameter {
    pub name: String,
    pub span: Option<Span>,
    pub kind: ParameterType,
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

pub(crate) fn span_of(ident: &Ident, source: &DescendSource) -> Option<Span> {
    source.span_at(ident.span?.begin)
}

pub fn is_launchable(fun_def: &FunDef) -> bool {
    matches!(fun_def.exec.exec.base, BaseExec::GpuGrid(_, _))
}

pub(crate) fn not_defined(kernel_name: &str) -> MacroError {
    MacroError::from(format!(
        "kernel `{kernel_name}` is not defined in this `descend!` block"
    ))
}

#[derive(Debug, Clone)]
pub enum ParameterType {
    Value(DataTy),
    Reference {
        ownership: Ownership,
        memory: Memory,
        dty: DataTy,
    },
}

impl ParameterType {
    pub(crate) fn dty(&self) -> &DataTy {
        match self {
            ParameterType::Value(dty) => dty,
            ParameterType::Reference { dty, .. } => dty,
        }
    }
}

fn parameter_type(ty: &Ty) -> Option<ParameterType> {
    let TyKind::Data(dty) = &ty.ty else {
        return None;
    };
    match &dty.dty {
        DataTyKind::Ref(reference_dty) => Some(ParameterType::Reference {
            ownership: reference_dty.own,
            memory: reference_dty.mem.clone(),
            dty: reference_dty.dty.as_ref().clone(),
        }),
        _ => Some(ParameterType::Value(dty.as_ref().clone())),
    }
}

impl Kernel {
    pub fn extract(
        compilation_unit: &CompilUnit,
        kernel_name: &str,
        source: &DescendSource,
    ) -> Result<Self, MacroError> {
        for item in &compilation_unit.items {
            if let Item::FunDef(fun_def) = item {
                let function_name = fun_def.ident.name.to_string();
                if function_name != kernel_name {
                    continue;
                }

                let name_span = span_of(&fun_def.ident, source);

                let BaseExec::GpuGrid(grid, block) = &fun_def.exec.exec.base else {
                    return Err(MacroError::at(
                        format!(
                            "kernel `{function_name}` must declare explicit grid and block \
                             dimensions, as `-[grid: gpu.grid<X<..>, X<..>>]->`"
                        ),
                        name_span,
                    ));
                };
                let dimensions = ExecutionDimensions {
                    grid: grid.clone(),
                    block: block.clone(),
                };

                let mut parameters = Vec::new();

                for parameter in &fun_def.param_decls {
                    let parameter_name = &parameter.ident.name;
                    let parameter_span = span_of(&parameter.ident, source);
                    let ty = parameter.ty.as_ref().unwrap();

                    let Some(kind) = parameter_type(ty) else {
                        return Err(MacroError::at(
                            format!(
                                "kernel `{function_name}`: parameter `{parameter_name}` has \
                                 no data type"
                            ),
                            parameter_span,
                        ));
                    };
                    let host = from_dty(kind.dty()).map_err(|reason| {
                        MacroError::at(
                            format!(
                                "kernel `{function_name}`: parameter `{parameter_name}` {reason}"
                            ),
                            parameter_span,
                        )
                    })?;
                    if let HostTy::Declared(type_name) = host.base() {
                        return Err(MacroError::at(
                            format!(
                                "kernel `{function_name}`: parameter `{parameter_name}` has the \
                                 custom type `{type_name}`, which cannot cross to the device: \
                                 Descend's code generator spells a struct by name and emits no \
                                 definition for it"
                            ),
                            parameter_span,
                        ));
                    }

                    parameters.push(KernelParameter {
                        name: parameter_name.to_string(),
                        span: parameter_span,
                        kind,
                        host,
                    });
                }

                return Ok(Kernel {
                    function_name,
                    name_span,
                    dimensions,
                    parameters,
                });
            }
        }

        Err(not_defined(kernel_name))
    }

    pub fn shared_memory_dtys(&self) -> Vec<DataTy> {
        self.parameters
            .iter()
            .filter_map(|parameter| match &parameter.kind {
                ParameterType::Reference {
                    memory: Memory::GpuShared,
                    dty,
                    ..
                } => Some(dty.clone()),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::TokenStream;
    use std::str::FromStr;

    fn extract(source_text: &str, kernel_name: &str) -> Result<Kernel, MacroError> {
        let source = crate::source::DescendSource::from_token_stream(
            TokenStream::from_str(source_text).expect("must tokenize"),
        );
        let code = descend::parser::SourceCode::new(source.text.clone());
        let unit = crate::source::parse_descend_source(&code)
            .unwrap_or_else(|error| panic!("fixture did not parse: {error}\n{}", source.text));
        Kernel::extract(&unit, kernel_name, &source)
    }

    #[test]
    fn only_shared_parameters_contribute_a_shared_memory_type() {
        let kernel = extract(
            "fn gpu_k<r1: prv, r2: prv>(
                 global_array: &r1 uniq gpu.global [i32; 256],
                 [grid.forall(X)] scratch: &r2 uniq gpu.shared [i32; 64]
             ) -[grid: gpu.grid<X<4>, X<64>>]-> () {
                 sched block in grid { sched thread in block { () } }
             }",
            "gpu_k",
        )
        .expect("a gpu.grid kernel must extract");

        let shared_memory_dtys = kernel.shared_memory_dtys();
        assert_eq!(shared_memory_dtys.len(), 1, "{shared_memory_dtys:?}");
        assert!(
            matches!(
                shared_memory_dtys[0].dty,
                DataTyKind::Array(_, Nat::Lit(64))
            ),
            "{shared_memory_dtys:?}"
        );
    }

    #[test]
    fn a_value_between_two_references_keeps_its_own_form() {
        let kernel = extract(
            "fn gpu_k<r1: prv, r2: prv>(
                 read_only: &r1 shrd gpu.global [i32; 256],
                 factor: i32,
                 [grid.forall(X)] scratch: &r2 uniq gpu.shared [i32; 64]
             ) -[grid: gpu.grid<X<4>, X<64>>]-> () {
                 sched block in grid { sched thread in block { () } }
             }",
            "gpu_k",
        )
        .expect("a gpu.grid kernel must extract");

        assert!(
            matches!(
                &kernel.parameters[0].kind,
                ParameterType::Reference {
                    ownership: Ownership::Shrd,
                    memory: Memory::GpuGlobal,
                    ..
                }
            ),
            "{:?}",
            kernel.parameters[0]
        );
        assert!(
            matches!(&kernel.parameters[1].kind, ParameterType::Value(_)),
            "{:?}",
            kernel.parameters[1]
        );
        assert!(
            matches!(
                &kernel.parameters[2].kind,
                ParameterType::Reference {
                    ownership: Ownership::Uniq,
                    memory: Memory::GpuShared,
                    ..
                }
            ),
            "{:?}",
            kernel.parameters[2]
        );
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
    fn a_parameter_of_a_custom_type_is_refused_at_its_own_span() {
        let error = extract(
            "struct Point { x: i32, y: i32 }
             fn gpu_k<r: prv>(points: &r uniq gpu.global [Point; 64])
                 -[grid: gpu.grid<X<1>, X<64>>]-> () {
                 sched block in grid { sched thread in block { () } }
             }",
            "gpu_k",
        )
        .expect_err("Descend's code generator emits no definition for a struct");

        assert!(error.message.contains("custom type `Point`"), "{error}");
        assert!(
            error.span.is_some(),
            "the refusal should point at the parameter, not at the kernel: {error}"
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
            error
                .message
                .contains("must declare explicit grid and block dimensions"),
            "{error}"
        );
        assert!(
            error.span.is_some(),
            "the refusal should point at the function's name, not the whole block: {error}"
        );
    }

    #[test]
    fn a_kernel_that_is_not_there_is_said_to_be_missing() {
        let error = extract(
            "fn gpu_k<r: prv>(a: &r uniq gpu.global [i32; 256])
                 -[grid: gpu.grid<X<4>, X<64>>]-> () {
                 sched block in grid { sched thread in block { () } }
             }",
            "gpu_other",
        )
        .expect_err("there is no kernel by that name");

        assert!(
            error.message.contains("gpu_other") && error.message.contains("not defined"),
            "{error}"
        );
    }
}
