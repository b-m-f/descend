use crate::artifact::LaunchConfig;
use crate::diag::MacroError;
use crate::kernel::{Kernel, KernelParameter};
use crate::limits;
use crate::nat::{eval_dimensions, Dimensions};
use crate::shared_memory::SharedMemoryBytes;

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

    let over_limit = |axis: &str, declared: u32, allowed: u32| {
        MacroError::at(
            format!(
                "kernel `{function}` declares a {axis} of {declared}, above the {allowed} a \
                 device accepts, so its launch would be refused"
            ),
            kernel.name_span,
        )
    };

    if block.x > limits::MAX_BLOCK_DIM_X {
        return Err(over_limit(
            "block x dimension",
            block.x,
            limits::MAX_BLOCK_DIM_X,
        ));
    }
    if block.y > limits::MAX_BLOCK_DIM_Y {
        return Err(over_limit(
            "block y dimension",
            block.y,
            limits::MAX_BLOCK_DIM_Y,
        ));
    }
    if block.z > limits::MAX_BLOCK_DIM_Z {
        return Err(over_limit(
            "block z dimension",
            block.z,
            limits::MAX_BLOCK_DIM_Z,
        ));
    }
    if grid.x > limits::MAX_GRID_DIM_X {
        return Err(over_limit(
            "grid x dimension",
            grid.x,
            limits::MAX_GRID_DIM_X,
        ));
    }
    if grid.y > limits::MAX_GRID_DIM_Y {
        return Err(over_limit(
            "grid y dimension",
            grid.y,
            limits::MAX_GRID_DIM_Y,
        ));
    }
    if grid.z > limits::MAX_GRID_DIM_Z {
        return Err(over_limit(
            "grid z dimension",
            grid.z,
            limits::MAX_GRID_DIM_Z,
        ));
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

fn shared_memory_element_bytes(parameter: &KernelParameter) -> Option<usize> {
    parameter
        .host
        .scalar()
        .and_then(crate::host_ty::ScalarKind::shared_memory_bytes)
}

fn check_shared_parameter_order(kernel: &Kernel) -> Result<(), MacroError> {
    let function = &kernel.function_name;
    let mut seen_shared = false;

    for parameter in &kernel.parameters {
        if !parameter.is_shared_memory {
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
        if !parameter.is_shared_memory {
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
        let Some(total_with_parameter) =
            parameter_bytes.and_then(|bytes| total.checked_add(bytes))
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
    use descend::ast::{DataTy, DataTyKind, Dim, Dim1d, Nat, ScalarTy};

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
        let shared_memory_dty = DataTy::new(DataTyKind::Array(
            Box::new(DataTy::new(DataTyKind::Scalar(element))),
            Nat::Lit(elements),
        ));
        let host = crate::host_ty::from_dty(&shared_memory_dty)
            .expect("an array of scalars crosses to the host");
        KernelParameter {
            name: name.to_string(),
            span: None,
            is_uniq: true,
            is_shared_memory: true,
            is_by_value: false,
            host,
            shared_memory_dty: Some(shared_memory_dty),
        }
    }

    fn global_parameter(name: &str) -> KernelParameter {
        KernelParameter {
            is_shared_memory: false,
            shared_memory_dty: None,
            ..shared_parameter(name, ScalarTy::I32, 64)
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
}
