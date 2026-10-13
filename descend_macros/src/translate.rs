use crate::diag::MacroError;

pub fn translate(
    source_code: &descend::parser::SourceCode,
    compilation_unit: &mut descend::ast::CompilUnit,
    kernel_name: &str,
    shared_memory_dtys: Vec<descend::ast::DataTy>,
) -> Result<String, MacroError> {
    let (checked, captured) = crate::capture::with_captured_output(|| {
        descend::ty_check::ext_ty_check_kernel(compilation_unit, kernel_name, &[])
            .map_err(|type_error| type_error.emit(source_code))
    });
    if checked.is_err() {
        return Err(crate::capture::descend_message(&captured)
            .unwrap_or_else(|| {
                "Descend type checking failed; the Descend compiler reported the reason above"
                    .to_string()
            })
            .into());
    }

    keep_only_type_checked_functions(compilation_unit);
    let device_code = generate_device_code(compilation_unit, kernel_name, shared_memory_dtys);

    debug!("========== GENERATED CUDA/C++ CODE for {} ==========\n{}\n=====================================================", kernel_name, device_code);

    Ok(device_code)
}

fn keep_only_type_checked_functions(compilation_unit: &mut descend::ast::CompilUnit) {
    compilation_unit.items.retain(|item| match item {
        descend::ast::Item::FunDef(function_definition) => function_definition.exec.ty.is_some(),
        _ => true,
    });
}

fn generate_device_code(
    compilation_unit: &descend::ast::CompilUnit,
    kernel_name: &str,
    shared_memory_dtys: Vec<descend::ast::DataTy>,
) -> String {
    let mut device_code =
        descend::codegen::ext_gen_kernel(compilation_unit, kernel_name, shared_memory_dtys);

    device_code = device_code.replace("std::size_t", "size_t");

    device_code.replace(
        &format!("__global__ auto {}(", kernel_name),
        &format!("extern \"C\" __global__ auto {}(", kernel_name),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use descend::ast::{DataTy, DataTyKind, Nat, ScalarTy};

    fn translate_source(
        kernel_source: &str,
        kernel_name: &str,
        shared_memory_dtys: Vec<DataTy>,
    ) -> Result<String, MacroError> {
        let source_code = descend::parser::SourceCode::new(kernel_source.to_string());
        let mut compilation_unit = crate::source::parse_descend_source(&source_code)?;

        translate(
            &source_code,
            &mut compilation_unit,
            kernel_name,
            shared_memory_dtys,
        )
    }

    fn array_of_i32(elements: usize) -> DataTy {
        DataTy::new(DataTyKind::Array(
            Box::new(DataTy::new(DataTyKind::Scalar(ScalarTy::I32))),
            Nat::Lit(elements),
        ))
    }

    #[test]
    fn a_parse_failure_keeps_the_compilers_message() {
        let error = translate_source("fn gpu_bad(", "gpu_bad", Vec::new())
            .expect_err("the source is incomplete");

        assert!(!error.message.is_empty(), "{error:?}");
        assert!(
            error.span.is_none(),
            "the Descend compiler reports the position itself: {error:?}"
        );
    }

    const KERNEL: &str = "\
fn gpu_add<r: prv>(
    a_array: &r uniq gpu.global [i32; 64]
) -[grid: gpu.grid<X<1>, X<64>>]-> () {
    sched block in grid {
        sched thread in block {
            let a_ref = &uniq (*a_array).to_view.grp::<64>[[block]][[thread]];
            *a_ref = *a_ref + 1
        }
    }
}";

    #[test]
    fn the_device_code_has_no_host_function_and_an_unmangled_entry() {
        let device_code =
            translate_source(KERNEL, "gpu_add", Vec::new()).expect("the kernel must compile");

        assert!(
            !device_code.contains("__host__"),
            "a host function reached the device code:\n{device_code}"
        );
        assert!(
            device_code.contains("extern \"C\" __global__ auto gpu_add"),
            "the kernel is missing or unmarked:\n{device_code}"
        );
    }

    #[test]
    fn a_kernel_is_type_checked_without_a_main_function() {
        assert!(translate_source(KERNEL, "gpu_add", Vec::new()).is_ok());
    }

    #[test]
    fn a_kernel_that_does_not_type_check_is_reported() {
        let error = translate_source(
            "\
fn gpu_bad<r: prv>(
    a_array: &r uniq gpu.global [i32; 64]
) -[grid: gpu.grid<X<1>, X<64>>]-> () {
    sched block in grid {
        sched thread in block {
            let a_ref = &uniq (*a_array).to_view.grp::<64>[[block]][[thread]];
            *a_ref = *a_ref + not_a_value
        }
    }
}",
            "gpu_bad",
            Vec::new(),
        )
        .unwrap_err();

        assert!(!error.message.is_empty(), "{error:?}");
    }

    #[test]
    fn a_checked_kernel_declares_its_shared_memory_without_a_launch_site() {
        let device_code = translate_source(
            "fn gpu_shared<r: prv>([grid.forall(X)] tmp: &r uniq gpu.shared [i32; 64]) \
             -[grid: gpu.grid<X<1>, X<64>>]-> () { sched block in grid { \
             sched thread in block { () } } }",
            "gpu_shared",
            vec![array_of_i32(64)],
        )
        .expect("the kernel must compile");

        assert!(
            device_code.contains("__global__ auto gpu_shared"),
            "{device_code}"
        );
        assert!(device_code.contains("extern __shared__"), "{device_code}");
        assert!(device_code.contains("gen_buffer"), "{device_code}");
    }

    #[test]
    fn only_the_checked_kernel_is_generated_from_multiple_definitions() {
        let device_code = translate_source(
            "fn first() -[grid: gpu.grid<X<1>, X<1>>]-> () { () } \
             fn second() -[grid: gpu.grid<X<1>, X<1>>]-> () { () }",
            "second",
            Vec::new(),
        )
        .expect("the selected kernel must compile");

        assert!(
            device_code.contains("__global__ auto second"),
            "{device_code}"
        );
        assert!(!device_code.contains("auto first("), "{device_code}");
    }
}
