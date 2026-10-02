pub struct DeviceCpp {
    pub code: String,
    pub host_function: String,
}

pub fn translate(program: &str, kernel_name: &str) -> Result<DeviceCpp, String> {
    debug!("FULL DESCEND:\n{}\nEND FULL DESCEND", program);
    let full_source = descend::parser::SourceCode::new(program.to_string());
    let mut compilation_unit = descend::parser::parse(&full_source).map_err(|_| {
        format!(
            "the synthesized Descend program did not parse; the Descend compiler reported the \
             reason above. Source was:\n{}",
            program
        )
    })?;

    let (checked, captured) =
        crate::capture::with_captured_output(|| descend::ty_check::ty_check(&mut compilation_unit));
    if checked.is_err() {
        return Err(
            crate::capture::descend_message(&captured).unwrap_or_else(|| {
                "Descend type checking failed; the Descend compiler reported the reason above"
                    .to_string()
            }),
        );
    }

    let device = generate_device_cpp(&mut compilation_unit, kernel_name);

    debug!("========== GENERATED CUDA/C++ CODE for {} ==========\n{}\n=====================================================", kernel_name, device.code);

    Ok(device)
}

fn generate_device_cpp(
    compilation_unit: &mut descend::ast::CompilUnit,
    kernel_name: &str,
) -> DeviceCpp {
    let mut device_code = descend::codegen::gen(compilation_unit);
    device_code = device_code.replace("std::size_t", "size_t");
    device_code = device_code.replace("auto main(", "auto dummy_main(");
    device_code = device_code.replace(
        &format!("__global__ auto {}(", kernel_name),
        &format!("extern \"C\" __global__ auto {}(", kernel_name),
    );

    let DeviceCpp {
        code,
        host_function,
    } = split_off_host_function(&device_code, HOST_FUNCTION_MARKER);
    let code = code.replace("$buffer", "_buffer");

    DeviceCpp {
        code: format!(
            "{}\n{}",
            descend::DESCEND_CUH,
            code.replace("#include \"descend.cuh\"", "")
        ),
        host_function,
    }
}

const HOST_FUNCTION_MARKER: &str = "__host__ auto dummy_main";

fn split_off_host_function(cuda_source: &str, marker: &str) -> DeviceCpp {
    let mut code = String::new();
    let mut host_function = String::new();
    let mut depth = 0i32;
    let mut in_host_function = false;
    let mut header_seen = false;

    for line in cuda_source.lines() {
        if !in_host_function && line.contains(marker) {
            in_host_function = true;
            header_seen = false;
            depth = 0;
        }

        if in_host_function {
            host_function.push_str(line);
            host_function.push('\n');

            let braces_opened = line.matches('{').count() as i32;
            let braces_closed = line.matches('}').count() as i32;
            depth += braces_opened - braces_closed;
            header_seen |= braces_opened > 0;

            if header_seen && depth <= 0 {
                in_host_function = false;
            } else if !header_seen && line.trim_end().ends_with(';') {
                in_host_function = false;
            }
            continue;
        }

        code.push_str(line);
        code.push('\n');
    }

    DeviceCpp {
        code,
        host_function,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch_scrape::scraped_shared_memory_bytes;

    #[test]
    fn the_host_function_is_removed_whole() {
        let cpp = "\
__global__ auto kernel(int *a) -> void { *a = 1; }
__host__ auto dummy_main() -> void {
  { int nested = 0; }
  kernel<<<dim3(1, 1, 1), dim3(2, 1, 1), 0>>>(a);
}
__global__ auto after(int *b) -> void { *b = 2; }
";
        let separated = split_off_host_function(cpp, HOST_FUNCTION_MARKER);
        assert!(!separated.code.contains("dummy_main"), "{}", separated.code);
        assert!(
            separated.code.contains("__global__ auto after"),
            "code after the host function was swallowed: {}",
            separated.code
        );
        assert!(
            separated.host_function.contains("<<<"),
            "{}",
            separated.host_function
        );
        assert!(
            separated.host_function.contains("nested"),
            "the nested block belongs to the host function: {}",
            separated.host_function
        );
    }

    #[test]
    fn a_forward_declaration_ends_at_its_semicolon() {
        let cpp = "\
__host__ auto dummy_main() -> void;
__global__ auto kernel(int *a) -> void { *a = 1; }
";
        let separated = split_off_host_function(cpp, HOST_FUNCTION_MARKER);
        assert!(!separated.code.contains("dummy_main"), "{}", separated.code);
        assert!(
            separated.code.contains("__global__ auto kernel"),
            "{}",
            separated.code
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

        let separated = generate_device_cpp(&mut compilation_unit, "gpu_add");

        assert!(
            !separated.code.contains("dummy_main"),
            "the synthesized host function reached the device code:\n{}",
            separated.code
        );
        assert!(
            separated
                .code
                .contains("extern \"C\" __global__ auto gpu_add"),
            "the kernel is missing or unmarked:\n{}",
            separated.code
        );
        assert!(
            separated.host_function.contains("<<<"),
            "the launch was not captured, so no byte count could be read:\n{}",
            separated.host_function
        );
        assert_eq!(
            scraped_shared_memory_bytes(&separated.host_function),
            Ok(0),
            "a kernel with no shared parameters requests no dynamic shared memory"
        );
    }
}
