use crate::artifact::LaunchConfig;
use crate::diag::MacroError;
use crate::kernel::{Kernel, KernelParam};
use crate::nat::eval_dim;

const MAX_SHARED_MEM_BYTES: u64 = 49152;

pub fn check_kernel(kernel: &Kernel) -> Result<LaunchConfig, MacroError> {
    check_shared_params(kernel)?;

    let shared_mem_bytes = shared_mem_bytes(kernel)?;
    if shared_mem_bytes > MAX_SHARED_MEM_BYTES {
        return Err(MacroError::at(
            format!(
                "kernel `{}` requests {shared_mem_bytes} bytes of dynamic shared memory, above \
                 the 48 KiB a launch may request without opting in through cuFuncSetAttribute",
                kernel.function_name
            ),
            kernel.name_span,
        ));
    }

    let dims = |dim, axis| {
        eval_dim(dim).map_err(|error| {
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
        grid: dims(&kernel.dims.grid, "grid")?,
        block: dims(&kernel.dims.block, "block")?,
        shared_mem_bytes,
    })
}

/// The per-element byte width Descend's shared-memory sizing defines for a parameter.
///
/// `None` means Descend emits no byte count for that element type, so the macro cannot
/// check the allocation rather than guess at it.
fn shared_mem_elem_bytes(param: &KernelParam) -> Option<u32> {
    param
        .host
        .scalar()
        .and_then(crate::host_ty::ScalarKind::shared_mem_bytes)
}

fn check_shared_params(kernel: &Kernel) -> Result<(), MacroError> {
    let function = &kernel.function_name;
    let mut seen_shared = false;

    for param in &kernel.params {
        if !param.is_shared_mem {
            if seen_shared {
                return Err(MacroError::at(
                    format!(
                        "kernel `{function}`: parameter `{}` follows a gpu.shared parameter. \
                     Shared-memory parameters must come last, because the compiler removes \
                     them from the end of the kernel signature",
                        param.name
                    ),
                    param.span,
                ));
            }
            continue;
        }
        seen_shared = true;

        if shared_mem_elem_bytes(param).is_none() {
            return Err(MacroError::at(
                format!(
                    "kernel `{function}`: shared parameter `{}` has an element type whose byte \
                 size the Descend compiler does not define, so its shared-memory allocation \
                 cannot be sized",
                    param.name
                ),
                param.span,
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
pub fn shared_mem_bytes(kernel: &Kernel) -> Result<u64, MacroError> {
    let function = &kernel.function_name;
    let mut total: u64 = 0;

    for param in &kernel.params {
        if !param.is_shared_mem {
            continue;
        }

        let elem_bytes = shared_mem_elem_bytes(param).ok_or_else(|| {
            MacroError::at(
                format!(
                    "kernel `{function}`: shared parameter `{}` has an element type whose byte \
                     size the Descend compiler does not define, so its shared-memory allocation \
                     cannot be sized",
                    param.name
                ),
                param.span,
            )
        })?;

        let elements = param.host.elements().ok_or_else(|| {
            MacroError::at(
                format!(
                    "kernel `{function}`: shared parameter `{}` has more elements than can be \
                     counted",
                    param.name
                ),
                param.span,
            )
        })?;
        total += u64::from(elem_bytes) * elements;
    }

    Ok(total)
}
