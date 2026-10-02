use std::io::Write;
use std::process::Command;

pub fn run_nvcc(cuda_source: &str, nvcc_arguments: &'static [String]) -> Result<String, String> {
    let mut cuda_file = tempfile::Builder::new()
        .prefix("descend_kernel_")
        .suffix(".cu")
        .tempfile()
        .map_err(|error| format!("Failed to create temporary .cu file: {error}"))?;
    cuda_file
        .write_all(cuda_source.as_bytes())
        .map_err(|error| format!("Failed to write to .cu file: {error}"))?;

    let ptx_file = tempfile::Builder::new()
        .prefix("descend_kernel_")
        .suffix(".ptx")
        .tempfile()
        .map_err(|error| format!("Failed to create temporary .ptx file: {error}"))?;

    let mut command = Command::new("nvcc");
    command
        .arg("-ptx")
        .arg(cuda_file.path())
        .arg("-o")
        .arg(ptx_file.path());
    for argument in nvcc_arguments {
        command.arg(argument);
    }

    let output = command
        .output()
        .map_err(|error| format!("Failed to invoke nvcc: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NVCC Compilation failed:\n{}", stderr));
    }

    std::fs::read_to_string(ptx_file.path())
        .map_err(|error| format!("Failed to read generated PTX file: {error}"))
}

fn cache_path(cuda_source: &str, nvcc_arguments: &[String]) -> Option<std::path::PathBuf> {
    use std::hash::{DefaultHasher, Hash, Hasher};

    if std::env::var_os("DESCEND_NO_CACHE").is_some() {
        return None;
    }

    let mut hasher = DefaultHasher::new();
    cuda_source.hash(&mut hasher);
    nvcc_arguments.hash(&mut hasher);
    nvcc_version().hash(&mut hasher);

    Some(
        std::env::temp_dir()
            .join("descend-ptx-cache")
            .join(format!("{:016x}.ptx", hasher.finish())),
    )
}

pub fn cached_ptx(cuda_source: &str, nvcc_arguments: &[String]) -> Option<String> {
    std::fs::read_to_string(cache_path(cuda_source, nvcc_arguments)?).ok()
}

pub fn store_ptx(cuda_source: &str, nvcc_arguments: &[String], ptx: &str) {
    let Some(path) = cache_path(cuda_source, nvcc_arguments) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, ptx);
}

fn nvcc_version() -> &'static str {
    static VERSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    VERSION.get_or_init(|| {
        Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|invocation| String::from_utf8_lossy(&invocation.stdout).into_owned())
            .unwrap_or_default()
    })
}
pub fn resolve_nvcc_options() -> Result<&'static [String], String> {
    static OPTIONS: std::sync::OnceLock<Result<Vec<String>, String>> = std::sync::OnceLock::new();
    OPTIONS
        .get_or_init(probe_nvcc_options)
        .as_ref()
        .map(|options| options.as_slice())
        .map_err(|error| error.clone())
}

fn probe_nvcc_options() -> Result<Vec<String>, String> {
    let flags_from_environment = std::env::var("DESCEND_NVCC_FLAGS").unwrap_or_default();
    let mut options: Vec<String> = flags_from_environment
        .split_whitespace()
        .map(|argument| argument.to_string())
        .collect();

    // TODO: only needed because offline nvcc leaves __CUDACC_RTC__ undefined, so
    // descend::array resolves to std::array, whose operator[] is host-only
    // constexpr, and indexing a by-value array argument in device code is
    // rejected. Compiling through NVRTC instead selects the __CUDACC_RTC__ branch
    // of descend.cuh, which already defines array with a __host__ __device__
    // operator[]; that was verified to compile this kernel unmodified. Drop this
    // flag once the PTX path moves to NVRTC.
    if !options
        .iter()
        .any(|option| option == "--expt-relaxed-constexpr" || option == "-expt-relaxed-constexpr")
    {
        options.push("--expt-relaxed-constexpr".to_string());
    }

    if !options
        .iter()
        .any(|option| option.starts_with("-arch=") || option.starts_with("--gpu-architecture="))
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
