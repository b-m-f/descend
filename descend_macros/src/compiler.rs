use std::path::PathBuf;

/// Compiles synthesized Descend source code down to PTX bytecode via descend library + nvcc.
/// Returns the path string to the generated PTX file for embedding via include_bytes!.
pub fn compile_to_ptx(full_descend: &str, fn_name: &str) -> Result<(String, String), String> {
    // CARGO_MANIFEST_DIR is set to the caller's directory. For trybuild, it is in target/tests/trybuild/descend.
    // However, the `descend_macros` crate itself was compiled from `descend/descend_macros`.
    // We can use env!("CARGO_MANIFEST_DIR") which is evaluated when `descend_macros` is compiled.
    let macro_crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = macro_crate_dir.parent().unwrap().to_path_buf();
    
    std::fs::create_dir_all(workspace_root.join("output"))
        .map_err(|e| format!("Failed to create output dir: {}", e))?;

    // Call descend library directly to compile the kernel
    let full_source = descend::parser::SourceCode::new(full_descend.to_string());
    let mut full_compil_unit =
        descend::parser::parse(&full_source).map_err(|e| format!("Failed to parse synthesized descend program: {:?}\nSource was:\n{}", e, full_descend))?;

    if let Err(e) = descend::ty_check::ty_check(&mut full_compil_unit) {
        return Err(format!("Descend type checking failed: {:?}", e));
    }

    let mut cpp_code = descend::codegen::gen(&full_compil_unit);

    // Rename main to dummy_main to avoid nvcc return type constraints for C++ main
    cpp_code = cpp_code.replace("auto main(", "auto dummy_main(");

    // Add extern "C" to prevent C++ name mangling so cuModuleGetFunction can find it by exact name
    cpp_code = cpp_code.replace(
        &format!("__global__ auto {}(", fn_name),
        &format!("extern \"C\" __global__ auto {}(", fn_name),
    );

    // Write C++ code to .cu file
    let cu_file_name = format!("output/{}.cu", fn_name);
    let macro_cu_path = workspace_root.join(&cu_file_name);
    std::fs::write(&macro_cu_path, cpp_code)
        .map_err(|e| format!("Failed to write .cu file: {}", e))?;

    // Compile to PTX using nvcc
    let ptx_file_name = format!("output/{}.ptx", fn_name);
    let macro_ptx_path = workspace_root.join(&ptx_file_name);
    let ptx_lib_path_str = macro_ptx_path.to_string_lossy().into_owned();

    let ptx_output = std::process::Command::new("nix")
        .arg("shell")
        .arg("--impure")
        .arg("nixpkgs#cudaPackages.cudatoolkit")
        .arg("--command")
        .arg("sh")
        .arg("-c")
        .arg(&format!(
            "nvcc -ptx -arch=compute_75 -I$(dirname $(which nvcc))/../include -I{}/cuda-examples {} -o {}",
            workspace_root.display(), cu_file_name, ptx_file_name
        ))
        .env("NIXPKGS_ALLOW_UNFREE", "1")
        .current_dir(&workspace_root)
        .output()
        .map_err(|e| format!("Failed to run nvcc compilation: {}", e))?;

    if !ptx_output.status.success() {
        let stderr = String::from_utf8_lossy(&ptx_output.stderr);
        let stdout = String::from_utf8_lossy(&ptx_output.stdout);
        return Err(format!("nvcc PTX compilation failed. stdout: {} stderr: {}", stdout, stderr));
    }

    // Downgrade PTX version string to bypass CUDA_ERROR_UNSUPPORTED_PTX_VERSION (218)
    // on older drivers while using newer NixOS CUDA toolkits.
    let ptx_content = std::fs::read_to_string(&macro_ptx_path).unwrap();
    let mut downgraded_ptx: String = ptx_content
        .lines()
        .map(|line| if line.starts_with(".version") { ".version 7.0" } else { line })
        .collect::<Vec<_>>()
        .join("\n");
    downgraded_ptx.push('\0');
    
    let mut mangled_name = fn_name.to_string();
    for line in downgraded_ptx.lines() {
        if line.starts_with(".visible .entry ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let name = parts[2].trim_end_matches('(');
                if name.contains(fn_name) {
                    mangled_name = name.to_string();
                    break;
                }
            }
        }
    }

    std::fs::write(&macro_ptx_path, downgraded_ptx).unwrap();

    Ok((ptx_lib_path_str, mangled_name))
}
