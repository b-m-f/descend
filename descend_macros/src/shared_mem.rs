use crate::const_expr::eval_const_int;
use crate::diag::MacroError;
use crate::kernel::Kernel;

pub fn shared_bytes_expr(host_fn: &str) -> Result<String, String> {
    let open = host_fn
        .find("<<<")
        .ok_or_else(|| "no kernel launch found in the generated host code".to_string())?;
    let rest = &host_fn[open + 3..];
    let close = rest
        .find(">>>")
        .ok_or_else(|| "the generated kernel launch is not terminated".to_string())?;
    let inside = &rest[..close];

    let args = split_top_level(inside);
    match args.as_slice() {
        [_grid, _block, bytes] => Ok(bytes.split_whitespace().collect()),
        other => Err(format!(
            "expected 3 launch arguments, found {} in `<<<{}>>>`",
            other.len(),
            inside.trim()
        )),
    }
}

fn split_top_level(args: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in args.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&args[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&args[start..]);
    parts
}

pub fn shared_mem_bytes(host_fn: &str) -> Result<u64, String> {
    eval_const_int(&shared_bytes_expr(host_fn)?)
}

/// The per-element byte width Descend's shared-memory sizing defines for a parameter.
///
/// `None` means Descend emits no byte count for that element type, so the macro cannot
/// check the allocation rather than guess at it.
fn shared_mem_elem_bytes(param: &crate::kernel::KernelParam) -> Option<u32> {
    param
        .host
        .scalar()
        .and_then(crate::host_ty::ScalarKind::shared_mem_bytes)
}

pub fn check_shared_params(kernel: &Kernel) -> Result<(), MacroError> {
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

pub(crate) fn check_shared_mem(kernel: &Kernel, emitted: u64) -> Result<(), MacroError> {
    let function = &kernel.function_name;
    let mut expected: u64 = 0;

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
        expected += u64::from(elem_bytes) * elements;
    }

    if expected != emitted {
        return Err(MacroError::at(
            format!(
                "internal error: shared-memory byte count disagreement for kernel `{function}`: \
             the Descend compiler's generated launch requests {emitted} bytes, the parameter \
                 types give {expected}. Refusing to launch"
            ),
            kernel.name_span,
        ));
    }

    if emitted > 49152 {
        return Err(MacroError::at(
            format!(
                "kernel `{function}` requests {emitted} bytes of dynamic shared memory, above the \
                 48 KiB a launch may request without opting in through cuFuncSetAttribute"
            ),
            kernel.name_span,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SGEMM_LAUNCH: &str = "__host__ auto dummy_main() -> void {
  sgemm_gpu<<<dim3(8, 16, 1), dim3(32, 8, 1),
              ((0 + (4 * ((1 * (32 * 2)) * 8))) + (4 * ((1 * (32 * 4)) * 8)))>>>(
      a, b, c);
}";

    #[test]
    fn reads_the_count_across_wrapped_lines() {
        assert_eq!(
            shared_bytes_expr(SGEMM_LAUNCH).unwrap(),
            "((0+(4*((1*(32*2))*8)))+(4*((1*(32*4))*8)))"
        );
        assert_eq!(shared_mem_bytes(SGEMM_LAUNCH).unwrap(), 6144);
    }

    #[test]
    fn a_kernel_without_shared_parameters_requests_nothing() {
        let launch = "gpu_vec_add<<<dim3(64, 1, 1), dim3(256, 1, 1), 0>>>(a, b);";
        assert_eq!(shared_mem_bytes(launch).unwrap(), 0);
    }

    #[test]
    fn dim3_commas_do_not_split_the_argument_list() {
        let launch = "k<<<dim3(1, 2, 3), dim3(4, 5, 6), (4 * 64)>>>(x);";
        assert_eq!(shared_mem_bytes(launch).unwrap(), 256);
    }

    #[test]
    fn an_unresolved_size_is_an_error() {
        let launch = "k<<<dim3(1, 1, 1), dim3(1, 1, 1), (4 * bs)>>>(x);";
        let error = shared_mem_bytes(launch).unwrap_err();
        assert!(
            error.contains("not a constant integer expression"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_missing_launch_is_an_error() {
        assert!(shared_bytes_expr("__host__ auto dummy_main() -> void {}").is_err());
    }

    #[test]
    fn a_launch_with_two_arguments_is_an_error() {
        let launch = "k<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(x);";
        let error = shared_bytes_expr(launch).unwrap_err();
        assert!(error.contains("expected 3 launch arguments"), "{error}");
    }
}
