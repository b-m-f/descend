pub struct DeviceCpp {
    pub code: String,
    pub host_fn: String,
}

pub fn translate(program: &str, fn_name: &str) -> Result<DeviceCpp, String> {
    debug!("FULL DESCEND:\n{}\nEND FULL DESCEND", program);
    let full_source = descend::parser::SourceCode::new(program.to_string());
    let mut compil_unit = descend::parser::parse(&full_source).map_err(|_| {
        format!(
            "the synthesized Descend program did not parse; the Descend compiler reported the \
             reason above. Source was:\n{}",
            program
        )
    })?;

    let (checked, captured) =
        crate::capture::with_captured_output(|| descend::ty_check::ty_check(&mut compil_unit));
    if checked.is_err() {
        return Err(crate::capture::descend_message(&captured).unwrap_or_else(|| {
            "Descend type checking failed; the Descend compiler reported the reason above"
                .to_string()
        }));
    }

    let device = generate_device_cpp(&mut compil_unit, fn_name);

    debug!("========== GENERATED CUDA/C++ CODE for {} ==========\n{}\n=====================================================", fn_name, device.code);

    Ok(device)
}

fn generate_device_cpp(compil_unit: &mut descend::ast::CompilUnit, fn_name: &str) -> DeviceCpp {
    let mut cpp_code = descend::codegen::gen(compil_unit);
    cpp_code = cpp_code.replace("std::size_t", "size_t");
    cpp_code = cpp_code.replace("auto main(", "auto dummy_main(");
    cpp_code = cpp_code.replace(
        &format!("__global__ auto {}(", fn_name),
        &format!("extern \"C\" __global__ auto {}(", fn_name),
    );

    let DeviceCpp { code, host_fn } = split_off_host_fn(&cpp_code, HOST_FN_MARKER);
    let code = code.replace("$buffer", "_buffer");

    DeviceCpp {
        code: format!(
            "{}\n{}",
            descend::DESCEND_CUH,
            code.replace("#include \"descend.cuh\"", "")
        ),
        host_fn,
    }
}

const HOST_FN_MARKER: &str = "__host__ auto dummy_main";

fn split_off_host_fn(cpp: &str, marker: &str) -> DeviceCpp {
    let mut code = String::new();
    let mut host_fn = String::new();
    let mut depth = 0i32;
    let mut in_host_fn = false;
    let mut header_seen = false;

    for line in cpp.lines() {
        if !in_host_fn && line.contains(marker) {
            in_host_fn = true;
            header_seen = false;
            depth = 0;
        }

        if in_host_fn {
            host_fn.push_str(line);
            host_fn.push('\n');

            let opened = line.matches('{').count() as i32;
            let closed = line.matches('}').count() as i32;
            depth += opened - closed;
            header_seen |= opened > 0;

            if header_seen && depth <= 0 {
                in_host_fn = false;
            } else if !header_seen && line.trim_end().ends_with(';') {
                in_host_fn = false;
            }
            continue;
        }

        code.push_str(line);
        code.push('\n');
    }

    DeviceCpp { code, host_fn }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch_scrape::scraped_shared_mem_bytes;

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
        let split = split_off_host_fn(cpp, HOST_FN_MARKER);
        assert!(!split.code.contains("dummy_main"), "{}", split.code);
        assert!(
            split.code.contains("__global__ auto after"),
            "code after the host function was swallowed: {}",
            split.code
        );
        assert!(split.host_fn.contains("<<<"), "{}", split.host_fn);
        assert!(
            split.host_fn.contains("nested"),
            "the nested block belongs to the host function: {}",
            split.host_fn
        );
    }

    #[test]
    fn a_forward_declaration_ends_at_its_semicolon() {
        let cpp = "\
__host__ auto dummy_main() -> void;
__global__ auto kernel(int *a) -> void { *a = 1; }
";
        let split = split_off_host_fn(cpp, HOST_FN_MARKER);
        assert!(!split.code.contains("dummy_main"), "{}", split.code);
        assert!(
            split.code.contains("__global__ auto kernel"),
            "{}",
            split.code
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
        let mut compil_unit = descend::parser::parse(&source).expect("test program must parse");
        descend::ty_check::ty_check(&mut compil_unit).expect("test program must type check");

        let split = generate_device_cpp(&mut compil_unit, "gpu_add");

        assert!(
            !split.code.contains("dummy_main"),
            "the synthesized host function reached the device code:\n{}",
            split.code
        );
        assert!(
            split.code.contains("extern \"C\" __global__ auto gpu_add"),
            "the kernel is missing or unmarked:\n{}",
            split.code
        );
        assert!(
            split.host_fn.contains("<<<"),
            "the launch was not captured, so no byte count could be read:\n{}",
            split.host_fn
        );
        assert_eq!(
            scraped_shared_mem_bytes(&split.host_fn),
            Ok(0),
            "a kernel with no shared parameters requests no dynamic shared memory"
        );
    }
}
