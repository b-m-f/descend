use cudarc::nvrtc::compile_ptx_with_opts;

/// Compiles synthesized Descend source code down to PTX bytecode via descend library + nvrtc.
/// Returns the raw PTX string and the mangled C++ entry function name.
/// Compiles synthesized Descend source code down to PTX bytecode via descend library + nvrtc.
/// 
/// What it does: Runs AST type checking and converts `full_descend` logic via NVCC configuration into raw PTX payload.
/// Why it is needed: Necessary to provide the executable payload needed by the host orchestrator (`cudarc`) at runtime.
pub fn compile_to_ptx(full_descend: &str, fn_name: &str) -> Result<(String, String), String> {
    eprintln!("FULL DESCEND:\n{}\nEND FULL DESCEND", full_descend);
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
    let mut opts = resolve_nvrtc_options()?;

    if let Ok(nvcc_path) = std::process::Command::new("which").arg("nvcc").output() {
        let nvcc_path_str = String::from_utf8_lossy(&nvcc_path.stdout);
        let nvcc_path = std::path::PathBuf::from(nvcc_path_str.trim());
        if let Some(bin_dir) = nvcc_path.parent() {
            if let Some(cuda_root) = bin_dir.parent() {
                let include_dir = cuda_root.join("include");
                opts.include_paths
                    .push(include_dir.to_string_lossy().to_string());
            }
        }
    }

    let ptx_result = compile_ptx_with_opts(final_cpp_code, opts);

    let ptx_string = match ptx_result {
        Ok(ptx) => ptx.to_src(),
        Err(e) => {
            return Err(format!("NVRTC Compilation failed:\n{}", e));
        }
    };

    let mangled_name = extract_mangled_name(&ptx_string, fn_name);

    Ok((ptx_string, mangled_name))
}

/// Generates intermediate C++ from a compiled AST unit.
/// 
/// What it does: Calls the descend code generator, sanitizes the payload, and masks out non-device artifacts.
/// Why it is needed: NVRTC expects valid pure GPU-bound C++ syntax; the host code embedded via CPU orchestrator crashes standard compilation.
fn generate_device_cpp(compil_unit: &mut descend::ast::CompilUnit, fn_name: &str) -> String {
    let mut cpp_code = descend::codegen::gen(compil_unit);
    // WARNING: TODO: Descend compiler outputs std::size_t which breaks nvrtc.
    // This is a temporary string replacement hack until the Descend compiler is updated.
    cpp_code = cpp_code.replace("std::size_t", "size_t");
    cpp_code = cpp_code.replace("auto main(", "auto dummy_main(");
    cpp_code = cpp_code.replace(
        &format!("__global__ auto {}(", fn_name),
        &format!("extern \"C\" __global__ auto {}(", fn_name),
    );

    // Strip host orchestration code (dummy_main) from the generated C++ since NVRTC is device-only
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

    // Inline the descend.cuh header because nvrtc CompileOptions in cudarc doesn't support virtual files directly
    format!(
        "{}\n{}",
        descend::DESCEND_CUH,
        device_cpp_code.replace("#include \"descend.cuh\"", "")
    )
}

/// Captures dynamic compilation flags passed into the system environment.
/// 
/// What it does: Interrogates `DESCEND_NVRTC_FLAGS` or auto-detects GPU computation capability.
/// Why it is needed: NVCC requires an accurate architecture flag (`compute_86`, etc.) in order to emit compatible PTX binaries.
fn resolve_nvrtc_options() -> Result<cudarc::nvrtc::CompileOptions, String> {
    // Read user-provided flags, defaulting to an empty string if not set
    let flags_env = std::env::var("DESCEND_NVRTC_FLAGS").unwrap_or_default();

    let mut options: Vec<String> = flags_env
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    // Enforce internal requirement for our C++ generation (unannotated functions compile as device)
    options.push("-default-device".to_string());

    // Check if the user explicitly provided an architecture flag
    let has_arch = options.iter().any(|opt| opt.starts_with("-arch="));

    if !has_arch {
        // Attempt auto-detection
        let detected_arch = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let cap = stdout.lines().next().unwrap_or("").trim().replace('.', "");
                if !cap.is_empty() {
                    Some(cap)
                } else {
                    None
                }
            });

        match detected_arch {
            Some(cap) => {
                options.push(format!("-arch=compute_{}", cap));
            }
            None => {
                return Err(
                    "CUDA Architecture not specified and auto-detection via `nvidia-smi` failed.\n\
                     Please specify the target architecture explicitly by setting the \
                     `DESCEND_NVRTC_FLAGS` environment variable (e.g., in `.cargo/config.toml`):\n\
                     \n\
                     [env]\n\
                     DESCEND_NVRTC_FLAGS = \"-arch=compute_86\"\n\
                    "
                    .to_string(),
                );
            }
        }
    }

    let mut opts = cudarc::nvrtc::CompileOptions::default();
    opts.options = options;
    Ok(opts)
}

/// Examines raw PTX strings to extract generated mangled kernel symbols.
/// 
/// What it does: String-matches standard `.visible .entry` conventions against the user-supplied function name.
/// Why it is needed: NVCC compilation obfuscates C++ functions; `cudarc` needs the exact mangled name string to launch the compiled bytecode payload securely.
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
