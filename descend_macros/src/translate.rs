use crate::diag::MacroError;

pub fn translate(
    kernel_source: &str,
    kernel_name: &str,
    span_at: impl Fn(descend::ast::Span) -> Option<proc_macro2::Span>,
) -> Result<String, MacroError> {
    debug!("FULL DESCEND:\n{}\nEND FULL DESCEND", kernel_source);
    let full_source = descend::parser::SourceCode::new(kernel_source.to_string());
    let mut compilation_unit = descend::parser::parse_with_diagnostics(&full_source)
        .map_err(|error| MacroError::at(error.message(), span_at(error.span())))?;

    let checked = descend::ty_check::ext_ty_check_kernel(&mut compilation_unit, kernel_name, &[]);
    if let Err(error) = checked {
        let rust_span = error.span().and_then(span_at);
        return Err(MacroError::at(error.message(), rust_span));
    }

    let device_code = generate_device_code(&mut compilation_unit, kernel_name);

    debug!("========== GENERATED CUDA/C++ CODE for {} ==========\n{}\n=====================================================", kernel_name, device_code);

    Ok(device_code)
}

fn generate_device_code(
    compilation_unit: &mut descend::ast::CompilUnit,
    kernel_name: &str,
) -> String {
    let mut device_code = descend::codegen::gen(compilation_unit);

    device_code = device_code.replace("std::size_t", "size_t");

    device_code.replace(
        &format!("__global__ auto {}(", kernel_name),
        &format!("extern \"C\" __global__ auto {}(", kernel_name),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_failure_keeps_its_message_and_mapped_span() {
        let error = translate("fn gpu_bad(", "gpu_bad", |_| {
            Some(proc_macro2::Span::call_site())
        })
        .expect_err("the source is incomplete");

        assert!(error.message.starts_with("expected "));
        assert!(error.span.is_some());
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
        let device_code = translate(KERNEL, "gpu_add", |_| None).expect("the kernel must compile");

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
        assert!(translate(KERNEL, "gpu_add", |_| None).is_ok());
    }

    #[test]
    fn a_kernel_that_does_not_type_check_is_reported() {
        let error = translate(
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
            |_| None,
        )
        .unwrap_err();

        assert!(!error.message.is_empty(), "{error:?}");
    }

    #[test]
    fn a_type_error_uses_the_span_supplied_by_the_caller() {
        let source = "fn gpu_bad() -[grid: gpu.grid<X<1>, X<1>>]-> () { missing_value }";
        let error = translate(source, "gpu_bad", |span| {
            assert_eq!(
                &source[span.begin as usize..span.end as usize],
                "missing_value"
            );
            Some(proc_macro2::Span::call_site())
        })
        .unwrap_err();

        assert!(error.span.is_some());
        assert!(error.message.contains("missing_value"), "{error:?}");
    }

    #[test]
    fn a_unique_borrow_through_a_shared_reference_points_to_the_dereference() {
        let source = "\
fn gpu_vec_add<r: prv>(
    a_array: &r uniq gpu.global [i32; 16384],
    b_array: &r shrd gpu.global [i32; 16384]
) -[grid: gpu.grid<X<64>, X<256>>]-> () {
    sched block in grid {
        sched thread in block {
            let a_ref = &uniq (*b_array).to_view.grp::<256>[[block]][[thread]];
            *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
        }
    }
}";
        let error = translate(source, "gpu_vec_add", |span| {
            assert_eq!(&source[span.begin as usize..span.end as usize], "*b_array");
            Some(proc_macro2::Span::call_site())
        })
        .unwrap_err();

        assert!(error.span.is_some());
        assert!(error.message.contains("shared reference"), "{error:?}");
    }

    #[test]
    fn a_checked_kernel_declares_its_shared_memory_without_a_launch_site() {
        let device_code = translate(
            "fn gpu_shared<r: prv>([grid.forall(X)] tmp: &r uniq gpu.shared [i32; 64]) \
             -[grid: gpu.grid<X<1>, X<64>>]-> () { sched block in grid { \
             sched thread in block { () } } }",
            "gpu_shared",
            |_| None,
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
        let device_code = translate(
            "fn first() -[grid: gpu.grid<X<1>, X<1>>]-> () { () } \
             fn second() -[grid: gpu.grid<X<1>, X<1>>]-> () { () }",
            "second",
            |_| None,
        )
        .expect("the selected kernel must compile");

        assert!(
            device_code.contains("__global__ auto second"),
            "{device_code}"
        );
        assert!(!device_code.contains("auto first("), "{device_code}");
    }
}
