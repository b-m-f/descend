
use std::io::Write;
use std::process::Command;

/// Compiles synthesized Descend source code down to PTX bytecode via descend library + nvcc.
/// Returns the raw PTX string and the mangled C++ entry function name.
/// 
/// What it does: Runs AST type checking and converts `full_descend` logic via NVCC into a raw PTX payload.
/// Why it is needed: Necessary to provide the executable payload needed by the host orchestrator at runtime.
pub fn compile_to_ptx(full_descend: &str, fn_name: &str) -> Result<(String, String), String> {
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

    let final_cpp_code = generate_device_cpp(&mut compil_unit, fn_name);
    
    debug!("========== GENERATED CUDA/C++ CODE for {} ==========\n{}\n=====================================================", fn_name, final_cpp_code);

    let temp_dir = std::env::temp_dir();
    let cu_path = temp_dir.join(format!("descend_kernel_{}.cu", uuid::Uuid::new_v4()));
    let ptx_path = temp_dir.join(format!("descend_kernel_{}.ptx", uuid::Uuid::new_v4()));

    let mut file = std::fs::File::create(&cu_path).map_err(|e| format!("Failed to create temporary .cu file: {}", e))?;
    file.write_all(final_cpp_code.as_bytes()).map_err(|e| format!("Failed to write to .cu file: {}", e))?;

    let nvcc_args = resolve_nvcc_options();
    
    let mut cmd = Command::new("nvcc");
    cmd.arg("-ptx").arg(&cu_path).arg("-o").arg(&ptx_path);
    for arg in nvcc_args {
        cmd.arg(arg);
    }

    let output = cmd.output().map_err(|e| format!("Failed to invoke nvcc: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NVCC Compilation failed:\n{}", stderr));
    }

    let ptx_string = std::fs::read_to_string(&ptx_path).map_err(|e| format!("Failed to read generated PTX file: {}", e))?;

    // Cleanup temp files
    let _ = std::fs::remove_file(cu_path);
    let _ = std::fs::remove_file(ptx_path);

    let mangled_name = extract_mangled_name(&ptx_string, fn_name);

    Ok((ptx_string, mangled_name))
}

fn generate_device_cpp(compil_unit: &mut descend::ast::CompilUnit, fn_name: &str) -> String {
    let mut cpp_code = descend::codegen::gen(compil_unit);
    cpp_code = cpp_code.replace("std::size_t", "size_t");
    cpp_code = cpp_code.replace("auto main(", "auto dummy_main(");
    cpp_code = cpp_code.replace(
        &format!("__global__ auto {}(", fn_name),
        &format!("extern \"C\" __global__ auto {}(", fn_name),
    );

    let mut device_cpp_code = String::new();
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
            continue;
        }

        device_cpp_code.push_str(line);
        device_cpp_code.push('\n');
    }

    let device_cpp_code = device_cpp_code.replace("$buffer", "_buffer");

    format!(
        "{}\n{}",
        descend::DESCEND_CUH,
        device_cpp_code.replace("#include \"descend.cuh\"", "")
    )
}

fn resolve_nvcc_options() -> Vec<String> {
    let flags_env = std::env::var("DESCEND_NVCC_FLAGS").unwrap_or_default();
    let mut options: Vec<String> = flags_env.split_whitespace().map(|s| s.to_string()).collect();

    // Default to the architecture of the current machine if possible
    if !options.iter().any(|opt| opt.starts_with("-arch=")) {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let cap = stdout.lines().next().unwrap_or("").trim().replace('.', "");
            if !cap.is_empty() {
                options.push(format!("-arch=compute_{}", cap));
            }
        }
    }
    options
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
