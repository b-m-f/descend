use std::io::Write;
use std::process::Command;

/// One compiled kernel: the PTX payload, the entry symbol to look up in it, and
/// the dynamic shared memory its launch must request.
pub struct CompiledKernel {
    pub ptx: String,
    pub mangled_name: String,
    pub shared_mem_bytes: u64,
}

/// Compiles synthesized Descend source code down to PTX bytecode via descend library + nvcc.
///
/// What it does: Runs AST type checking and converts `full_descend` logic via NVCC into a raw PTX payload.
/// Why it is needed: Necessary to provide the executable payload needed by the host orchestrator at runtime.
pub fn compile_to_ptx(full_descend: &str, fn_name: &str) -> Result<CompiledKernel, String> {
    debug!("FULL DESCEND:\n{}\nEND FULL DESCEND", full_descend);
    let full_source = descend::parser::SourceCode::new(full_descend.to_string());
    let mut compil_unit = descend::parser::parse(&full_source).map_err(|e| {
        format!(
            "Failed to parse synthesized descend program: {:?}\nSource was:\n{}",
            e, full_descend
        )
    })?;

    if let Err(e) = descend::ty_check::ty_check(&mut compil_unit) {
        return Err(format!("Descend type checking failed: {:?}", e));
    }

    let DeviceCpp {
        code: final_cpp_code,
        host_fn,
    } = generate_device_cpp(&mut compil_unit, fn_name);

    // Read the launch's shared-memory argument before spawning nvcc, so a size
    // the generator could not resolve is reported without the compile round trip.
    let shared_mem_bytes = crate::shared_mem::shared_mem_bytes(&host_fn)?;

    debug!("========== GENERATED CUDA/C++ CODE for {} ==========\n{}\n=====================================================", fn_name, final_cpp_code);

    let temp_dir = std::env::temp_dir();
    let cu_path = temp_dir.join(format!("descend_kernel_{}.cu", uuid::Uuid::new_v4()));
    let ptx_path = temp_dir.join(format!("descend_kernel_{}.ptx", uuid::Uuid::new_v4()));

    let mut file = std::fs::File::create(&cu_path)
        .map_err(|e| format!("Failed to create temporary .cu file: {}", e))?;
    file.write_all(final_cpp_code.as_bytes())
        .map_err(|e| format!("Failed to write to .cu file: {}", e))?;

    let nvcc_args = resolve_nvcc_options()?;

    let mut cmd = Command::new("nvcc");
    cmd.arg("-ptx").arg(&cu_path).arg("-o").arg(&ptx_path);
    for arg in nvcc_args {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to invoke nvcc: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NVCC Compilation failed:\n{}", stderr));
    }

    let ptx_string = std::fs::read_to_string(&ptx_path)
        .map_err(|e| format!("Failed to read generated PTX file: {}", e))?;

    // Cleanup temp files
    let _ = std::fs::remove_file(cu_path);
    let _ = std::fs::remove_file(ptx_path);

    let mangled_name = extract_mangled_name(&ptx_string, fn_name);

    Ok(CompiledKernel {
        ptx: ptx_string,
        mangled_name,
        shared_mem_bytes,
    })
}

/// The generated CUDA, split into the device code that goes to nvcc and the
/// synthesized host function that is stripped from it.
struct DeviceCpp {
    code: String,
    host_fn: String,
}

fn generate_device_cpp(compil_unit: &mut descend::ast::CompilUnit, fn_name: &str) -> DeviceCpp {
    let mut cpp_code = descend::codegen::gen(compil_unit);
    cpp_code = cpp_code.replace("std::size_t", "size_t");
    cpp_code = cpp_code.replace("auto main(", "auto dummy_main(");
    cpp_code = cpp_code.replace(
        &format!("__global__ auto {}(", fn_name),
        &format!("extern \"C\" __global__ auto {}(", fn_name),
    );

    let mut device_cpp_code = String::new();
    let mut host_fn = String::new();
    let mut in_dummy_main = false;
    let mut brace_count = 0;
    let mut is_decl = false;

    for line in cpp_code.lines() {
        if line.contains("__host__ auto dummy_main") {
            in_dummy_main = true;
            brace_count = 0;
            is_decl = !line.contains("{");
        }

        if in_dummy_main {
            if line.contains("{") {
                brace_count += line.matches('{').count() as i32;
                is_decl = false;
            }
            if line.contains("}") {
                brace_count -= line.matches('}').count() as i32;
                if brace_count <= 0 {
                    in_dummy_main = false;
                }
            } else if is_decl && line.trim().ends_with(";") {
                in_dummy_main = false;
            }
            // The synthesized host function does not belong in the device code,
            // but its launch carries the shared-memory byte count.
            host_fn.push_str(line);
            host_fn.push('\n');
            continue;
        }

        device_cpp_code.push_str(line);
        device_cpp_code.push('\n');
    }

    let device_cpp_code = device_cpp_code.replace("$buffer", "_buffer");

    DeviceCpp {
        code: format!(
            "{}\n{}",
            descend::DESCEND_CUH,
            device_cpp_code.replace("#include \"descend.cuh\"", "")
        ),
        host_fn,
    }
}

fn resolve_nvcc_options() -> Result<Vec<String>, String> {
    let flags_env = std::env::var("DESCEND_NVCC_FLAGS").unwrap_or_default();
    let mut options: Vec<String> = flags_env
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    // Default to the architecture of the current machine if possible
    if !options
        .iter()
        .any(|opt| opt.starts_with("-arch=") || opt.starts_with("--gpu-architecture="))
    {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .map_err(|error| format!(
                "Could not detect a CUDA architecture with nvidia-smi ({error}). Set DESCEND_NVCC_FLAGS=-arch=compute_<capability>."
            ))?;
        if !output.status.success() {
            return Err(format!(
                "nvidia-smi could not detect a CUDA architecture. Set DESCEND_NVCC_FLAGS=-arch=compute_<capability>. Details: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let capability = String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_owned();
        let mut parts = capability.split('.');
        let major = parts.next().unwrap_or("");
        let minor = parts.next().unwrap_or("");
        if parts.next().is_some()
            || major.is_empty()
            || minor.is_empty()
            || !major.chars().all(|c| c.is_ascii_digit())
            || !minor.chars().all(|c| c.is_ascii_digit())
        {
            return Err(format!(
                "nvidia-smi returned invalid compute capability {capability:?}. Set DESCEND_NVCC_FLAGS=-arch=compute_<capability>."
            ));
        }
        options.push(format!("-arch=compute_{major}{minor}"));
    }
    Ok(options)
}

fn extract_mangled_name(ptx_string: &str, original_fn_name: &str) -> String {
    let mut mangled_name = original_fn_name.to_string();
    for line in ptx_string.lines() {
        if line.starts_with(".visible .entry ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let name = parts[2].trim_end_matches('(');
                if name.contains(original_fn_name) {
                    mangled_name = name.to_string();
                    break;
                }
            }
        }
    }
    mangled_name
}
