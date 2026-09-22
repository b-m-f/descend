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

    let nvcc_args = resolve_nvcc_options()?;
    let ptx_string = match cached_ptx(&final_cpp_code, nvcc_args) {
        Some(cached) => {
            debug!("PTX cache hit for {}", fn_name);
            cached
        }
        None => {
            let ptx = run_nvcc(&final_cpp_code, nvcc_args)?;
            store_ptx(&final_cpp_code, nvcc_args, &ptx);
            ptx
        }
    };

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

/// Compiles `cuda` to PTX with `nvcc`.
///
/// The sources live in a `tempfile`, so a failure between writing and reading
/// cannot leave them behind.
fn run_nvcc(cuda: &str, nvcc_args: &'static [String]) -> Result<String, String> {
    let mut cu_file = tempfile::Builder::new()
        .prefix("descend_kernel_")
        .suffix(".cu")
        .tempfile()
        .map_err(|e| format!("Failed to create temporary .cu file: {e}"))?;
    cu_file
        .write_all(cuda.as_bytes())
        .map_err(|e| format!("Failed to write to .cu file: {e}"))?;

    let ptx_file = tempfile::Builder::new()
        .prefix("descend_kernel_")
        .suffix(".ptx")
        .tempfile()
        .map_err(|e| format!("Failed to create temporary .ptx file: {e}"))?;

    let mut cmd = Command::new("nvcc");
    cmd.arg("-ptx")
        .arg(cu_file.path())
        .arg("-o")
        .arg(ptx_file.path());
    for arg in nvcc_args {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to invoke nvcc: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NVCC Compilation failed:\n{}", stderr));
    }

    std::fs::read_to_string(ptx_file.path())
        .map_err(|e| format!("Failed to read generated PTX file: {e}"))
}

/// Where a compilation of `cuda` with `nvcc_args` would be cached.
///
/// The key covers the generated CUDA rather than the Descend source, so a change
/// in the Descend compiler invalidates the entry too, and the nvcc options and
/// version, since either changes the output. `DESCEND_NO_CACHE` disables the
/// cache entirely.
fn cache_path(cuda: &str, nvcc_args: &[String]) -> Option<std::path::PathBuf> {
    use std::hash::{DefaultHasher, Hash, Hasher};

    if std::env::var_os("DESCEND_NO_CACHE").is_some() {
        return None;
    }

    let mut hasher = DefaultHasher::new();
    cuda.hash(&mut hasher);
    nvcc_args.hash(&mut hasher);
    nvcc_version().hash(&mut hasher);

    Some(
        std::env::temp_dir()
            .join("descend-ptx-cache")
            .join(format!("{:016x}.ptx", hasher.finish())),
    )
}

fn cached_ptx(cuda: &str, nvcc_args: &[String]) -> Option<String> {
    std::fs::read_to_string(cache_path(cuda, nvcc_args)?).ok()
}

fn store_ptx(cuda: &str, nvcc_args: &[String], ptx: &str) {
    let Some(path) = cache_path(cuda, nvcc_args) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // A failed write only costs the next compilation a cache miss.
    let _ = std::fs::write(path, ptx);
}

/// The `nvcc` banner, read once per process.
fn nvcc_version() -> &'static str {
    static VERSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    VERSION.get_or_init(|| {
        Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
            .unwrap_or_default()
    })
}

/// The header of the host function the macro synthesized and does not want.
const HOST_FN_MARKER: &str = "__host__ auto dummy_main";

/// Splits generated CUDA into the device code and the synthesized host function.
///
/// The function is found by its header and ended by brace balance rather than by
/// the first closing brace, so a nested block inside it cannot end it early. A
/// header that is only a declaration ends at its semicolon. The host function is
/// returned rather than dropped, because the dynamic shared-memory byte count
/// lives in the launch it contains.
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
                // A forward declaration, with no body to skip.
                in_host_fn = false;
            }
            continue;
        }

        code.push_str(line);
        code.push('\n');
    }

    DeviceCpp { code, host_fn }
}

/// The nvcc options for this process, resolved once.
///
/// Detecting the architecture spawns `nvidia-smi`, which is far too expensive to
/// repeat for every kernel in a build.
fn resolve_nvcc_options() -> Result<&'static [String], String> {
    static OPTIONS: std::sync::OnceLock<Result<Vec<String>, String>> = std::sync::OnceLock::new();
    OPTIONS
        .get_or_init(probe_nvcc_options)
        .as_ref()
        .map(|options| options.as_slice())
        .map_err(|error| error.clone())
}

fn probe_nvcc_options() -> Result<Vec<String>, String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Runs the real code generator, so the split keeps working if the shape of
    /// its output drifts. Needs no GPU and no `nvcc`.
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
            crate::shared_mem::shared_mem_bytes(&split.host_fn),
            Ok(0),
            "a kernel with no shared parameters requests no dynamic shared memory"
        );
    }
}
