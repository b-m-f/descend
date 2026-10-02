use crate::artifact::LaunchConfig;
use crate::diag::MacroError;
use crate::kernel::{Kernel, KernelParameter};
use crate::nat::eval_dimensions;

const MAX_SHARED_MEMORY_BYTES: u64 = 49152;

pub fn check_kernel(kernel: &Kernel) -> Result<LaunchConfig, MacroError> {
    check_shared_parameters(kernel)?;

    let shared_memory_bytes = shared_memory_bytes(kernel)?;
    if shared_memory_bytes > MAX_SHARED_MEMORY_BYTES {
        return Err(MacroError::at(
            format!(
                "kernel `{}` requests {shared_memory_bytes} bytes of dynamic shared memory, above \
                 the 48 KiB a launch may request without opting in through cuFuncSetAttribute",
                kernel.function_name
            ),
            kernel.name_span,
        ));
    }

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

    Ok(LaunchConfig {
        grid: evaluate(&kernel.dimensions.grid, "grid")?,
        block: evaluate(&kernel.dimensions.block, "block")?,
        shared_memory_bytes,
    })
}

/// The per-element byte width Descend's shared-memory sizing defines for a parameter.
///
/// `None` means Descend emits no byte count for that element type, so the macro cannot
/// check the allocation rather than guess at it.
fn shared_memory_element_bytes(parameter: &KernelParameter) -> Option<u32> {
    parameter
        .host
        .scalar()
        .and_then(crate::host_ty::ScalarKind::shared_memory_bytes)
}

fn check_shared_parameters(kernel: &Kernel) -> Result<(), MacroError> {
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

        if shared_memory_element_bytes(parameter).is_none() {
            return Err(MacroError::at(
                format!(
                    "kernel `{function}`: shared parameter `{}` has an element type whose byte \
                 size the Descend compiler does not define, so its shared-memory allocation \
                 cannot be sized",
                    parameter.name
                ),
                parameter.span,
            ));
        }
    }

    Ok(())
}

/// The dynamic shared memory the kernel's launch needs, from the parameter types.
///
/// This is the macro's own answer, not one read back out of the generated CUDA. The
/// agreement between the two is pinned by
/// `launch_scrape::tests::the_typed_count_agrees_with_the_generated_launch`.
pub fn shared_memory_bytes(kernel: &Kernel) -> Result<u64, MacroError> {
    let function = &kernel.function_name;
    let mut total: u64 = 0;

    for parameter in &kernel.parameters {
        if !parameter.is_shared_memory {
            continue;
        }

        let elem_bytes = shared_memory_element_bytes(parameter).ok_or_else(|| {
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

        let elements = parameter.host.elements().ok_or_else(|| {
            MacroError::at(
                format!(
                    "kernel `{function}`: shared parameter `{}` has more elements than can be \
                     counted",
                    parameter.name
                ),
                parameter.span,
            )
        })?;
        total += u64::from(elem_bytes) * elements;
    }

    Ok(total)
}
