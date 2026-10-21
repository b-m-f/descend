use crate::artifact::Ptx;
use crate::diag::MacroError;
use crate::nvrtc;

pub fn assemble(cpp: &str, kernel_name: &str) -> Result<Ptx, MacroError> {
    let toolchain = nvrtc::toolchain()?;
    let ptx_text = toolchain.compile(cpp, descend::DESCEND_CUH)?;
    let entry = find_entry_symbol(&ptx_text, kernel_name)?;

    Ok(Ptx {
        text: ptx_text,
        entry,
    })
}

fn declared_entry_symbols(ptx_text: &str) -> Vec<String> {
    ptx_text
        .lines()
        .filter_map(|line| {
            let after_entry_directive = line.split_once(".entry ")?.1;
            let symbol = after_entry_directive
                .trim_start()
                .split(['(', ' ', '\t'])
                .next()
                .unwrap_or("");
            (!symbol.is_empty()).then(|| symbol.to_string())
        })
        .collect()
}

fn find_entry_symbol(ptx_text: &str, kernel_name: &str) -> Result<String, MacroError> {
    let declared = declared_entry_symbols(ptx_text);
    if declared.iter().any(|symbol| symbol == kernel_name) {
        return Ok(kernel_name.to_string());
    }

    let what_nvrtc_produced = match declared.as_slice() {
        [] => "it declares no entry symbol at all".to_string(),
        symbols => format!("it declares {}", symbols.join(", ")),
    };
    Err(MacroError::from(format!(
        "the PTX NVRTC produced has no entry symbol named `{kernel_name}`: {what_nvrtc_produced}. \
         A kernel is compiled `extern \"C\"`, so its entry symbol is its own name; a missing one \
         means the rename in `translate::generate_device_code` did not match what Descend's code \
         generator emitted"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PTX_WITH_TWO_ENTRIES: &str = "\
.visible .entry gpu_vec_add(
    .param .u64 gpu_vec_add_param_0
)
{
    ret;
}

.visible .entry gpu_scale(
    .param .u64 gpu_scale_param_0
)
{
    ret;
}
";

    #[test]
    fn the_entry_symbol_is_the_kernels_own_name() {
        assert_eq!(
            find_entry_symbol(PTX_WITH_TWO_ENTRIES, "gpu_vec_add").unwrap(),
            "gpu_vec_add"
        );
        assert_eq!(
            find_entry_symbol(PTX_WITH_TWO_ENTRIES, "gpu_scale").unwrap(),
            "gpu_scale"
        );
    }

    #[test]
    fn a_kernel_name_that_is_a_substring_of_another_entry_does_not_match_it() {
        let ptx = ".visible .entry padded(\n    .param .u64 padded_param_0\n)\n{\n    ret;\n}\n";
        let error = find_entry_symbol(ptx, "add").unwrap_err();
        assert!(
            error.message.contains("no entry symbol named `add`"),
            "{error:?}"
        );
        assert!(
            error.message.contains("padded"),
            "the message should say what was there: {error:?}"
        );
    }

    #[test]
    fn a_missing_entry_is_a_compile_time_error_that_says_what_was_found() {
        let error = find_entry_symbol(PTX_WITH_TWO_ENTRIES, "gpu_never_compiled").unwrap_err();
        assert!(
            error.message.contains("gpu_vec_add, gpu_scale"),
            "the message should list the entries NVRTC did produce: {error:?}"
        );
    }

    #[test]
    fn ptx_with_no_entry_at_all_is_reported_rather_than_passed_through() {
        let error = find_entry_symbol(".version 8.3\n.target sm_86\n", "gpu_vec_add").unwrap_err();
        assert!(
            error.message.contains("no entry symbol at all"),
            "{error:?}"
        );
    }

    #[test]
    fn a_device_function_is_not_an_entry() {
        let ptx = ".visible .func helper(\n    .param .u64 helper_param_0\n)\n{\n    ret;\n}\n";
        assert!(declared_entry_symbols(ptx).is_empty());
        assert!(find_entry_symbol(ptx, "helper").is_err());
    }
}
