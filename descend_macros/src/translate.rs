use crate::diag::MacroError;

/// The CUDA NVRTC is given, and the host-side `main` that was taken out of it.
///
/// `host_code` is not part of the pipeline: the launch it describes is only read by
/// `launch_scrape`, so a test can confirm the shared-memory count Descend's own code
/// generator emits agrees with the one `check::shared_memory_bytes` derives from the
/// parameter types.
#[derive(Debug)]
pub struct CppSource {
    pub device_code: String,
    pub host_code: String,
}

pub fn translate(program: &str, kernel_name: &str) -> Result<CppSource, MacroError> {
    debug!("FULL DESCEND:\n{}\nEND FULL DESCEND", program);
    let full_source = descend::parser::SourceCode::new(program.to_string());
    let mut compilation_unit = descend::parser::parse(&full_source).map_err(|_| {
        MacroError::from(format!(
            "the synthesized Descend program did not parse; the Descend compiler reported the \
             reason above. Source was:\n{}",
            program
        ))
    })?;

    let (checked, captured) =
        crate::capture::with_captured_output(|| descend::ty_check::ty_check(&mut compilation_unit));
    if checked.is_err() {
        return Err(crate::capture::descend_message(&captured)
            .unwrap_or_else(|| {
                "Descend type checking failed; the Descend compiler reported the reason above"
                    .to_string()
            })
            .into());
    }

    let cpp_source = generate_cpp_source(&mut compilation_unit, kernel_name);

    debug!("========== GENERATED CUDA/C++ CODE for {} ==========\n{}\n=====================================================", kernel_name, cpp_source.device_code);

    Ok(cpp_source)
}

/// Turning Descend's generated CUDA into a translation unit NVRTC will accept as a
/// device-only module.
///
/// Each replacement below undoes one thing `descend::codegen` emits that is right for
/// a whole program and wrong for a PTX module. They are string replacements rather than
/// changes to the code generator because `descend/src/` is not this crate's to edit; the
/// site that produces each one is named so a change there is traceable to here.
fn generate_cpp_source(
    compilation_unit: &mut descend::ast::CompilUnit,
    kernel_name: &str,
) -> CppSource {
    let mut device_code = descend::codegen::gen(compilation_unit);

    // The launcher looks the kernel up by name through `cuModuleGetFunction`, which
    // needs an unmangled symbol. `assemble::find_entry_symbol` depends on this landing.
    device_code = device_code.replace(
        &format!("__global__ auto {}(", kernel_name),
        &format!("extern \"C\" __global__ auto {}(", kernel_name),
    );

    let CppSource {
        device_code,
        host_code,
    } = split_off_host_side_main_function(&device_code, HOST_SIDE_MAIN_MARKER);

    // The code generator names the dynamic shared-memory backing array `$buffer`
    // (`src/codegen/mod.rs:794`). nvcc accepts `$` in an identifier, NVRTC reports
    // `unrecognized token`, and it cannot be spelled in Descend source at all.
    let device_code = device_code.replace("$buffer", "_buffer");

    CppSource {
        device_code,
        host_code,
    }
}

// The open parenthesis is what keeps the marker from matching a `__host__` function whose
// name merely starts with `main`. `synthesize::synthesize_main_function` writes the `main`
// it marks, purely so Descend's type checker has something to check the kernel against.
const HOST_SIDE_MAIN_MARKER: &str = "__host__ auto main(";

fn split_off_host_side_main_function(cuda_source: &str, marker: &str) -> CppSource {
    let mut device_code = String::new();
    let mut host_code = String::new();
    let mut depth = 0i32;
    let mut in_host_side_main_function = false;
    let mut header_seen = false;

    for line in cuda_source.lines() {
        if !in_host_side_main_function && line.contains(marker) {
            in_host_side_main_function = true;
            header_seen = false;
            depth = 0;
        }

        if in_host_side_main_function {
            host_code.push_str(line);
            host_code.push('\n');

            let braces_opened = line.matches('{').count() as i32;
            let braces_closed = line.matches('}').count() as i32;
            depth += braces_opened - braces_closed;
            header_seen |= braces_opened > 0;

            // A definition ends where its braces balance. A forward declaration never
            // opens one, so it ends at its semicolon instead; `header_seen` tells the
            // two apart.
            let definition_ended = header_seen && depth <= 0;
            let declaration_ended = !header_seen && line.trim_end().ends_with(';');
            if definition_ended || declaration_ended {
                in_host_side_main_function = false;
            }
            continue;
        }

        device_code.push_str(line);
        device_code.push('\n');
    }

    CppSource {
        device_code,
        host_code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch_scrape::scraped_shared_memory_bytes;

    #[test]
    fn the_host_side_main_function_is_removed_whole() {
        let cpp = "\
__global__ auto kernel(int *a) -> void { *a = 1; }
__host__ auto main() -> void {
  { int nested = 0; }
  kernel<<<dim3(1, 1, 1), dim3(2, 1, 1), 0>>>(a);
}
__global__ auto after(int *b) -> void { *b = 2; }
";
        let separated = split_off_host_side_main_function(cpp, HOST_SIDE_MAIN_MARKER);
        assert!(
            !separated.device_code.contains("auto main("),
            "{}",
            separated.device_code
        );
        assert!(
            separated.device_code.contains("__global__ auto after"),
            "code after the host function was swallowed: {}",
            separated.device_code
        );
        assert!(
            separated.host_code.contains("<<<"),
            "{}",
            separated.host_code
        );
        assert!(
            separated.host_code.contains("nested"),
            "the nested block belongs to the host function: {}",
            separated.host_code
        );
    }

    #[test]
    fn a_forward_declaration_ends_at_its_semicolon() {
        let cpp = "\
__host__ auto main() -> void;
__global__ auto kernel(int *a) -> void { *a = 1; }
";
        let separated = split_off_host_side_main_function(cpp, HOST_SIDE_MAIN_MARKER);
        assert!(
            !separated.device_code.contains("auto main("),
            "{}",
            separated.device_code
        );
        assert!(
            separated.device_code.contains("__global__ auto kernel"),
            "{}",
            separated.device_code
        );
    }

    /// The regression the marker's open parenthesis prevents: without it, a `__host__`
    /// function whose name merely starts with `main` is swallowed along with the real one.
    #[test]
    fn a_host_function_named_after_main_survives_the_split() {
        let cpp = "\
__host__ auto main() -> void { }
__host__ auto main_loop(int *a) -> void { *a = 1; }
";
        let separated = split_off_host_side_main_function(cpp, HOST_SIDE_MAIN_MARKER);
        assert!(
            !separated.device_code.contains("auto main("),
            "{}",
            separated.device_code
        );
        assert!(
            separated.device_code.contains("__host__ auto main_loop"),
            "a function whose name starts with `main` was swallowed: {}",
            separated.device_code
        );
    }

    #[test]
    fn the_split_matches_what_the_code_generator_emits() {
        let source = descend::parser::SourceCode::new(
            "fn gpu_add(a_array: &r uniq gpu.global [i32; 64]) -[grid: gpu.grid<X<1>, X<64>>]-> () {
                 sched block in grid {
                     sched thread in block {
                         let a_ref = &uniq (*a_array).to_view.grp::<64>[[block]][[thread]];
                         *a_ref = *a_ref + 1
                     }
                 }
             }
             fn main() -[t: cpu.thread]-> () <'a> {
                 let mut gpu = gpu_device(0);
                 let h: &'a shrd cpu.mem [i32; 64] = unsafe _;
                 let mut a_gpu = gpu_alloc_copy(&uniq gpu, h);
                 gpu_add::<<<X<1>, X<64>>>>(&uniq a_gpu);
                 ()
             }"
                .to_string(),
        );
        let mut compilation_unit =
            descend::parser::parse(&source).expect("test program must parse");
        descend::ty_check::ty_check(&mut compilation_unit).expect("test program must type check");

        let separated = generate_cpp_source(&mut compilation_unit, "gpu_add");

        assert!(
            !separated.device_code.contains("auto main("),
            "the synthesized host function reached the device code:\n{}",
            separated.device_code
        );
        assert!(
            separated
                .device_code
                .contains("extern \"C\" __global__ auto gpu_add"),
            "the kernel is missing or unmarked:\n{}",
            separated.device_code
        );
        assert!(
            separated.host_code.contains("<<<"),
            "the launch was not captured, so no byte count could be read:\n{}",
            separated.host_code
        );
        assert_eq!(
            scraped_shared_memory_bytes(&separated.host_code),
            Ok(0),
            "a kernel with no shared parameters requests no dynamic shared memory"
        );
    }
}
