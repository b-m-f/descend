use crate::artifact::LaunchConfig;
use crate::diag::MacroError;
use crate::invocation::DimensionAnnotation;
use crate::kernel::{Kernel, KernelParameter, ParameterType};
use crate::limits;
use crate::nat::{eval_dimensions, Dimensions};
use crate::shared_memory::SharedMemoryBytes;
use descend::ast::Memory;

pub fn check_kernel(kernel: &Kernel) -> Result<LaunchConfig, MacroError> {
    check_shared_parameter_order(kernel)?;

    let shared_memory_bytes = shared_memory_bytes(kernel)?;

    let evaluate = |expression, axis| {
        eval_dimensions(expression).map_err(|error| {
            MacroError::at(
                format!(
                    "kernel `{}`: its {axis} dimensions are not a constant: {error}",
                    kernel.function_name
                ),
                kernel.name_span,
            )
        })
    };

    let grid = evaluate(&kernel.dimensions.grid, "grid")?;
    let block = evaluate(&kernel.dimensions.block, "block")?;
    check_dimensions(kernel, &grid, &block)?;

    Ok(LaunchConfig {
        grid,
        block,
        shared_memory_bytes,
    })
}

fn check_dimensions(
    kernel: &Kernel,
    grid: &Dimensions,
    block: &Dimensions,
) -> Result<(), MacroError> {
    let function = &kernel.function_name;

    let refused = |axis: &str, declared: u32, bound: &str| {
        MacroError::at(
            format!(
                "kernel `{function}` declares a {axis} of {declared}, {bound}, so its launch \
                 would be refused"
            ),
            kernel.name_span,
        )
    };

    for (axis, declared, allowed) in [
        ("block x dimension", block.x, limits::MAX_BLOCK_DIM_X),
        ("block y dimension", block.y, limits::MAX_BLOCK_DIM_Y),
        ("block z dimension", block.z, limits::MAX_BLOCK_DIM_Z),
        ("grid x dimension", grid.x, limits::MAX_GRID_DIM_X),
        ("grid y dimension", grid.y, limits::MAX_GRID_DIM_Y),
        ("grid z dimension", grid.z, limits::MAX_GRID_DIM_Z),
    ] {
        if declared == 0 {
            return Err(refused(axis, declared, "below the 1 a device accepts"));
        }
        if declared > allowed {
            return Err(refused(
                axis,
                declared,
                &format!("above the {allowed} a device accepts"),
            ));
        }
    }

    let threads = u64::from(block.x) * u64::from(block.y) * u64::from(block.z);
    if threads > u64::from(limits::MAX_THREADS_PER_BLOCK) {
        return Err(MacroError::at(
            format!(
                "kernel `{function}` declares a block of {threads} threads \
                 ({} x {} x {}), above the {} a device runs in one block; each axis is within \
                 its own limit, but a block is measured by their product",
                block.x,
                block.y,
                block.z,
                limits::MAX_THREADS_PER_BLOCK
            ),
            kernel.name_span,
        ));
    }

    Ok(())
}

pub fn confirm_dimensions(
    kernel: &Kernel,
    declared: &LaunchConfig,
    annotated: &DimensionAnnotation,
) -> Result<(), MacroError> {
    let function = &kernel.function_name;

    let axes = [
        ("grid", "x", declared.grid.x, annotated.grid.x),
        ("grid", "y", declared.grid.y, annotated.grid.y),
        ("grid", "z", declared.grid.z, annotated.grid.z),
        ("block", "x", declared.block.x, annotated.block.x),
        ("block", "y", declared.block.y, annotated.block.y),
        ("block", "z", declared.block.z, annotated.block.z),
    ];

    let declared_grid = format!(
        "({}, {}, {})",
        declared.grid.x, declared.grid.y, declared.grid.z
    );
    let declared_block = format!(
        "({}, {}, {})",
        declared.block.x, declared.block.y, declared.block.z
    );
    let annotated_grid = format!(
        "({}, {}, {})",
        annotated.grid.x, annotated.grid.y, annotated.grid.z
    );
    let annotated_block = format!(
        "({}, {}, {})",
        annotated.block.x, annotated.block.y, annotated.block.z
    );

    let differing = axes
        .into_iter()
        .find(|(_, _, declared_value, annotated_value)| declared_value != annotated_value);

    let Some((which, axis, declared_value, annotated_value)) = differing else {
        return Ok(());
    };

    Err(MacroError::at(
        format!(
            "kernel `{function}` is launched with a {which} {axis} dimension of {annotated_value}, \
             but its declaration evaluates to {declared_value}. The launch reads grid \
             {annotated_grid} and block {annotated_block}; the declaration reads grid \
             {declared_grid} and block {declared_block}. The kernel was checked against the \
             declaration, so the launch site cannot change it: correct the `<<<grid, block>>>` \
             annotation or drop it"
        ),
        annotated.span.or(kernel.name_span),
    ))
}

fn shared_memory_element_bytes(parameter: &KernelParameter) -> Option<usize> {
    parameter
        .host
        .scalar()
        .and_then(crate::host_ty::shared_memory_bytes)
}

fn check_shared_parameter_order(kernel: &Kernel) -> Result<(), MacroError> {
    let function = &kernel.function_name;
    let mut seen_shared = false;

    for parameter in &kernel.parameters {
        if !matches!(
            &parameter.kind,
            ParameterType::Reference {
                memory: Memory::GpuShared,
                ..
            }
        ) {
            if seen_shared {
                return Err(MacroError::at(
                    format!(
                        "kernel `{function}`: parameter `{}` follows a gpu.shared parameter. \
                     Shared-memory parameters must come last, because the compiler removes \
                     them from the end of the kernel signature",
                        parameter.name
                    ),
                    parameter.span,
                ));
            }
            continue;
        }
        seen_shared = true;
    }

    Ok(())
}

pub fn shared_memory_bytes(kernel: &Kernel) -> Result<SharedMemoryBytes, MacroError> {
    let function = &kernel.function_name;
    let mut total: usize = 0;

    for parameter in &kernel.parameters {
        if !matches!(
            &parameter.kind,
            ParameterType::Reference {
                memory: Memory::GpuShared,
                ..
            }
        ) {
            continue;
        }

        let element_bytes = shared_memory_element_bytes(parameter).ok_or_else(|| {
            MacroError::at(
                format!(
                    "kernel `{function}`: shared parameter `{}` has an element type whose byte \
                     size the Descend compiler does not define, so its shared-memory allocation \
                     cannot be sized",
                    parameter.name
                ),
                parameter.span,
            )
        })?;

        let elements = parameter.host.total_elements();
        let parameter_bytes = element_bytes.checked_mul(elements);
        let Some(total_with_parameter) = parameter_bytes.and_then(|bytes| total.checked_add(bytes))
        else {
            return Err(MacroError::at(
                format!(
                    "kernel `{function}`: shared parameter `{}` requests more shared memory than \
                     can be counted",
                    parameter.name
                ),
                parameter.span,
            ));
        };
        total = total_with_parameter;
    }

    SharedMemoryBytes::new(total as u64).ok_or_else(|| {
        MacroError::at(
            format!(
                "kernel `{function}` requests {total} bytes of dynamic shared memory, above the \
                 {} KiB a launch may request without opting in through cuFuncSetAttribute",
                limits::MAX_SHARED_MEMORY_BYTES / 1024
            ),
            kernel.name_span,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use descend::ast::{DataTy, DataTyKind, Dim, Dim1d, Dim2d, Nat, Ownership, ScalarTy};

    fn dimensions(x: u32, y: u32, z: u32) -> Dimensions {
        Dimensions { x, y, z }
    }

    fn one_dimensional(nat: usize) -> Dim {
        Dim::X(Box::new(Dim1d(Nat::Lit(nat))))
    }

    fn kernel_with(parameters: Vec<KernelParameter>) -> Kernel {
        Kernel {
            function_name: "gpu_k".to_string(),
            name_span: None,
            dimensions: crate::kernel::ExecutionDimensions {
                grid: one_dimensional(1),
                block: one_dimensional(1),
            },
            parameters,
        }
    }

    fn shared_parameter(name: &str, element: ScalarTy, elements: usize) -> KernelParameter {
        reference(name, Memory::GpuShared, element, elements)
    }

    fn global_parameter(name: &str) -> KernelParameter {
        reference(name, Memory::GpuGlobal, ScalarTy::I32, 64)
    }

    fn reference(
        name: &str,
        memory: Memory,
        element: ScalarTy,
        elements: usize,
    ) -> KernelParameter {
        let dty = DataTy::new(DataTyKind::Array(
            Box::new(DataTy::new(DataTyKind::Scalar(element))),
            Nat::Lit(elements),
        ));
        let host = crate::host_ty::from_dty(&dty).expect("an array of scalars crosses to the host");
        KernelParameter {
            name: name.to_string(),
            span: None,
            kind: ParameterType::Reference {
                ownership: Ownership::Uniq,
                memory,
                dty,
            },
            host,
        }
    }

    fn check(grid: Dimensions, block: Dimensions) -> Result<(), MacroError> {
        check_dimensions(&kernel_with(Vec::new()), &grid, &block)
    }

    #[test]
    fn a_launch_at_every_limit_is_accepted() {
        assert!(check(dimensions(1, 1, 1), dimensions(1024, 1, 1)).is_ok());
        assert!(check(dimensions(1, 1, 1), dimensions(1, 1024, 1)).is_ok());
        assert!(check(dimensions(1, 1, 1), dimensions(16, 16, 4)).is_ok());
        assert!(check(
            dimensions(2_147_483_647, 65535, 65535),
            dimensions(32, 32, 1)
        )
        .is_ok());
    }

    #[test]
    fn a_zero_axis_is_refused_before_the_device_is_asked() {
        for (grid, block, axis) in [
            (dimensions(0, 1, 1), dimensions(1, 1, 1), "grid x dimension"),
            (dimensions(1, 0, 1), dimensions(1, 1, 1), "grid y dimension"),
            (
                dimensions(1, 1, 1),
                dimensions(0, 1, 1),
                "block x dimension",
            ),
            (
                dimensions(1, 1, 1),
                dimensions(1, 0, 1),
                "block y dimension",
            ),
            (
                dimensions(1, 1, 1),
                dimensions(1, 1, 0),
                "block z dimension",
            ),
        ] {
            let error = check(grid, block).unwrap_err();
            assert!(error.message.contains(axis), "{error}");
            assert!(
                error.message.contains("below the 1 a device accepts"),
                "the message should state the bound it derived: {error}"
            );
        }
    }

    #[test]
    fn each_block_axis_is_refused_on_its_own() {
        for (block, axis) in [
            (dimensions(1025, 1, 1), "block x dimension"),
            (dimensions(1, 1025, 1), "block y dimension"),
            (dimensions(1, 1, 65), "block z dimension"),
        ] {
            let error = check(dimensions(1, 1, 1), block).unwrap_err();
            assert!(error.message.contains(axis), "{error}");
            assert!(
                error.message.contains("would be refused"),
                "the message should say what would happen: {error}"
            );
        }
    }

    #[test]
    fn each_grid_axis_is_refused_on_its_own() {
        for (grid, axis) in [
            (dimensions(2_147_483_648, 1, 1), "grid x dimension"),
            (dimensions(1, 65536, 1), "grid y dimension"),
            (dimensions(1, 1, 65536), "grid z dimension"),
        ] {
            let error = check(grid, dimensions(1, 1, 1)).unwrap_err();
            assert!(error.message.contains(axis), "{error}");
        }
    }

    #[test]
    fn a_block_whose_axes_are_legal_but_whose_product_is_not_is_refused() {
        let error = check(dimensions(1, 1, 1), dimensions(32, 32, 2)).unwrap_err();
        assert!(error.message.contains("2048 threads"), "{error}");
        assert!(
            error.message.contains("each axis is within its own limit"),
            "the reader needs to know why the axes did not warn them: {error}"
        );
    }

    #[test]
    fn a_block_at_exactly_the_thread_limit_is_accepted() {
        assert!(check(dimensions(1, 1, 1), dimensions(32, 32, 1)).is_ok());
        assert!(check(dimensions(1, 1, 1), dimensions(1024, 1, 1)).is_ok());
    }

    #[test]
    fn shared_memory_at_the_ceiling_is_accepted_and_above_it_is_not() {
        let at_the_ceiling = kernel_with(vec![shared_parameter("tmp", ScalarTy::I32, 12288)]);
        assert!(check_kernel(&at_the_ceiling).is_ok());

        let over_the_ceiling = kernel_with(vec![shared_parameter("tmp", ScalarTy::I32, 12289)]);
        let error = check_kernel(&over_the_ceiling).unwrap_err();
        assert!(error.message.contains("49156 bytes"), "{error}");
        assert!(
            error.message.contains("48 KiB"),
            "the message should state the ceiling it derived from the constant: {error}"
        );
    }

    #[test]
    fn several_shared_parameters_add_up() {
        let kernel = kernel_with(vec![
            shared_parameter("left", ScalarTy::F64, 256),
            shared_parameter("right", ScalarTy::Bool, 256),
        ]);
        assert_eq!(shared_memory_bytes(&kernel).unwrap().value(), 2048 + 256);
    }

    #[test]
    fn a_shared_parameter_before_an_ordinary_one_is_refused() {
        let kernel = kernel_with(vec![
            shared_parameter("tmp", ScalarTy::I32, 64),
            global_parameter("vec"),
        ]);
        let error = check_shared_parameter_order(&kernel).unwrap_err();
        assert!(
            error.message.contains("follows a gpu.shared parameter"),
            "{error}"
        );
        assert!(error.message.contains("`vec`"), "{error}");
    }

    #[test]
    fn a_shared_parameter_of_an_element_descend_cannot_size_is_refused() {
        let kernel = kernel_with(vec![shared_parameter("tmp", ScalarTy::U8, 64)]);
        let error = shared_memory_bytes(&kernel).unwrap_err();
        assert!(
            error.message.contains("element type whose byte size"),
            "{error}"
        );
        assert!(error.message.contains("`tmp`"), "{error}");
    }

    #[test]
    fn a_kernel_with_no_shared_parameters_requests_no_shared_memory() {
        let kernel = kernel_with(vec![global_parameter("vec")]);
        assert_eq!(shared_memory_bytes(&kernel).unwrap().value(), 0);
    }

    #[test]
    fn a_shared_request_above_the_portable_ceiling_is_refused() {
        let kernel = kernel_with(vec![shared_parameter("big", ScalarTy::I32, 16384)]);
        let error = shared_memory_bytes(&kernel).unwrap_err();

        assert!(error.message.contains("65536 bytes"), "{error}");
        assert!(error.message.contains("48 KiB"), "{error}");
        assert!(error.message.contains("`gpu_k`"), "{error}");
    }

    fn launch_of(grid: Dim, block: Dim) -> (Kernel, LaunchConfig) {
        let kernel = Kernel {
            function_name: "gpu_k".to_string(),
            name_span: None,
            dimensions: crate::kernel::ExecutionDimensions { grid, block },
            parameters: Vec::new(),
        };
        let declared = check_kernel(&kernel).expect("the declared dimensions are valid");
        (kernel, declared)
    }

    fn along(x: usize, y: usize) -> Dim {
        Dim::XY(Box::new(Dim2d(Nat::Lit(x), Nat::Lit(y))))
    }

    fn annotated(grid: Dimensions, block: Dimensions) -> DimensionAnnotation {
        DimensionAnnotation {
            grid,
            block,
            span: None,
        }
    }

    #[test]
    fn an_annotation_matching_the_declaration_is_accepted() {
        let (kernel, declared) = launch_of(one_dimensional(4), one_dimensional(256));
        assert!(confirm_dimensions(
            &kernel,
            &declared,
            &annotated(dimensions(4, 1, 1), dimensions(256, 1, 1))
        )
        .is_ok());
    }

    #[test]
    fn a_scalar_annotation_is_compared_against_the_padded_declaration() {
        let (kernel, declared) = launch_of(one_dimensional(64), one_dimensional(256));
        let error = confirm_dimensions(
            &kernel,
            &declared,
            &annotated(dimensions(64, 1, 2), dimensions(256, 1, 1)),
        )
        .expect_err("a grid z of 2 was not declared");
        assert!(
            error.message.contains("grid z dimension of 2"),
            "{}",
            error.message
        );
        assert!(
            error.message.contains("evaluates to 1"),
            "{}",
            error.message
        );
    }

    #[test]
    fn an_annotation_naming_a_different_grid_is_refused_with_both_triples() {
        let (kernel, declared) = launch_of(one_dimensional(4), one_dimensional(256));
        let error = confirm_dimensions(
            &kernel,
            &declared,
            &annotated(dimensions(8, 1, 1), dimensions(256, 1, 1)),
        )
        .expect_err("the kernel declares a grid of 4");
        let message = &error.message;
        assert!(message.contains("grid x dimension of 8"), "{message}");
        assert!(message.contains("evaluates to 4"), "{message}");
        assert!(message.contains("grid (8, 1, 1)"), "{message}");
        assert!(message.contains("grid (4, 1, 1)"), "{message}");
        assert!(message.contains("`gpu_k`"), "{message}");
    }

    #[test]
    fn an_annotation_naming_a_different_block_is_refused() {
        let (kernel, declared) = launch_of(one_dimensional(4), one_dimensional(256));
        let error = confirm_dimensions(
            &kernel,
            &declared,
            &annotated(dimensions(4, 1, 1), dimensions(128, 1, 1)),
        )
        .expect_err("the kernel declares a block of 256");
        assert!(
            error.message.contains("block x dimension of 128"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_two_dimensional_declaration_is_matched_axis_by_axis() {
        let (kernel, declared) = launch_of(along(16, 16), along(32, 32));
        assert!(confirm_dimensions(
            &kernel,
            &declared,
            &annotated(dimensions(16, 16, 1), dimensions(32, 32, 1))
        )
        .is_ok());

        let error = confirm_dimensions(
            &kernel,
            &declared,
            &annotated(dimensions(16, 1, 1), dimensions(32, 32, 1)),
        )
        .expect_err("the second grid axis is declared, not 1");
        assert!(
            error.message.contains("grid y dimension of 1"),
            "{}",
            error.message
        );
    }
}
