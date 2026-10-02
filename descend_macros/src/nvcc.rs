use crate::diag::MacroError;
use std::io::Write;
use std::process::Command;

// Every failure in this module is a toolchain or filesystem problem, not something
// attributable to one place in the user's source, so none of them carries a span.
pub fn run_nvcc(cuda_source: &str, nvcc_arguments: &[String]) -> Result<String, MacroError> {
    let mut cuda_file = tempfile::Builder::new()
        .prefix("descend_kernel_")
        .suffix(".cu")
        .tempfile()
        .map_err(|error| format!("failed to create a temporary .cu file: {error}"))?;
    cuda_file
        .write_all(cuda_source.as_bytes())
        .map_err(|error| format!("failed to write to a temporary .cu file: {error}"))?;

    let ptx_file = tempfile::Builder::new()
        .prefix("descend_kernel_")
        .suffix(".ptx")
        .tempfile()
        .map_err(|error| format!("failed to create a temporary .ptx file: {error}"))?;

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
        .map_err(|error| format!("failed to invoke nvcc: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("nvcc compilation failed:\n{}", stderr).into());
    }

    std::fs::read_to_string(ptx_file.path())
        .map_err(|error| format!("failed to read the generated PTX file: {error}").into())
}

/// Where compiled PTX is kept between builds.
///
/// Deliberately not `/tmp`. A cache in a world-writable directory under a predictable
/// name is an invitation, and `create_dir_all` succeeds quietly when the directory is
/// already there -- whoever made it first owns it. From that position another local user
/// can either plant a `.ptx` at the hash this crate computes, which the macro then
/// embeds in the binary and the GPU runs, or plant a symlink, which turns a cache write
/// into a truncation of a file the reader owns.
///
/// Under `CARGO_TARGET_DIR` or the user's cache home the directory is not shared with
/// anyone, and [`verified_cache_directory`] checks that it still is not before either
/// reading from it or writing to it.
fn cache_directory() -> Option<std::path::PathBuf> {
    if let Some(configured) = std::env::var_os("DESCEND_PTX_CACHE") {
        return Some(std::path::PathBuf::from(configured));
    }
    if let Some(target_directory) = std::env::var_os("CARGO_TARGET_DIR") {
        return Some(std::path::PathBuf::from(target_directory).join("descend-ptx-cache"));
    }
    if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(
            std::path::PathBuf::from(cache_home)
                .join("descend")
                .join("ptx"),
        );
    }
    std::env::var_os("HOME").map(|home| {
        std::path::PathBuf::from(home)
            .join(".cache")
            .join("descend")
            .join("ptx")
    })
}

/// The cache directory, created if it is missing and verified to be this user's.
///
/// An existing directory is not assumed to be safe, which is the part `create_dir_all`
/// cannot do: mode `0o700` protects only a directory this process created, so one that
/// was already there has to be checked for a symlink, for another owner, and for group
/// or world write permission. `None` means no caching, in either direction.
fn verified_cache_directory() -> Option<std::path::PathBuf> {
    let directory = cache_directory()?;

    match std::fs::symlink_metadata(&directory) {
        Ok(metadata) => {
            if !is_private_directory(&metadata) {
                return None;
            }
        }
        // `NotFound` is the ordinary first run; anything else (a permissions problem, a
        // dangling component) is a reason to compile without a cache rather than guess.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(&directory).ok()?;
            // The leaf is what holds the PTX. Intermediates were created under a
            // directory the user already owns and are covered by the process umask.
            restrict_to_owner(&directory);
            if !std::fs::symlink_metadata(&directory)
                .is_ok_and(|metadata| is_private_directory(&metadata))
            {
                return None;
            }
        }
        Err(_) => return None,
    }

    Some(directory)
}

/// Whether a directory is this process's own and writable by nobody else.
#[cfg(unix)]
fn is_private_directory(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    metadata.is_dir()
        && metadata.mode() & 0o077 == 0
        && metadata.uid() == unsafe { libc::geteuid() }
}

#[cfg(not(unix))]
fn is_private_directory(metadata: &std::fs::Metadata) -> bool {
    metadata.is_dir()
}

#[cfg(unix)]
fn restrict_to_owner(directory: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;

    let _ = std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn restrict_to_owner(_directory: &std::path::Path) {}

fn cache_path(cuda_source: &str, nvcc_arguments: &[String]) -> Option<std::path::PathBuf> {
    if std::env::var_os("DESCEND_NO_CACHE").is_some() {
        return None;
    }

    Some(verified_cache_directory()?.join(format!(
        "{:016x}.ptx",
        cache_key(cuda_source, nvcc_arguments)
    )))
}

/// FNV-1a over the CUDA source, the nvcc arguments, and the nvcc version.
///
/// `DefaultHasher` would be one line, but it is documented as unstable across Rust
/// releases, so a toolchain bump silently invalidates every cached kernel and the first
/// build afterwards pays for nvcc all over again. A hash spelled out here is stable by
/// construction, which is the only property the value needs: it names a file, and is not
/// a security boundary -- [`verified_cache_directory`] is what makes the file trustworthy.
fn cache_key(cuda_source: &str, nvcc_arguments: &[String]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn mix(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(PRIME);
        }
    }

    let mut hash = OFFSET_BASIS;
    mix(&mut hash, cuda_source.as_bytes());
    for argument in nvcc_arguments {
        // A separator, so `["-a", "-b"]` and `["-a-b"]` are different keys.
        mix(&mut hash, &[0]);
        mix(&mut hash, argument.as_bytes());
    }
    mix(&mut hash, &[0]);
    mix(&mut hash, nvcc_version().as_bytes());
    hash
}

pub fn cached_ptx(cuda_source: &str, nvcc_arguments: &[String]) -> Option<String> {
    std::fs::read_to_string(cache_path(cuda_source, nvcc_arguments)?).ok()
}

pub fn store_ptx(cuda_source: &str, nvcc_arguments: &[String], ptx: &str) {
    if let Some(path) = cache_path(cuda_source, nvcc_arguments) {
        write_ptx_atomically(&path, ptx);
    }
}

/// Staged in the cache directory and renamed into place.
///
/// `fs::write` would open the destination with `O_CREAT|O_TRUNC`, which follows a
/// symlink; `persist` will not replace a path that is already there, so a file planted
/// at the hash is left alone rather than overwritten -- and the next build simply reads
/// it as a cache hit. Failing to write is not an error either: the caller already has
/// the PTX, and the next build compiles again.
fn write_ptx_atomically(path: &std::path::Path, ptx: &str) {
    let Some(directory) = path.parent() else {
        return;
    };
    let Ok(mut staged) = tempfile::Builder::new()
        .prefix(".descend-ptx-")
        .suffix(".tmp")
        .tempfile_in(directory)
    else {
        return;
    };
    if staged.write_all(ptx.as_bytes()).is_err() || staged.flush().is_err() {
        return;
    }
    let _ = staged.persist(path);
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

pub fn resolve_nvcc_options() -> Result<&'static [String], MacroError> {
    static OPTIONS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    if let Some(options) = OPTIONS.get() {
        return Ok(options.as_slice());
    }

    // Only a success is stored. `probe_nvcc_options` shells out to nvidia-smi, which
    // can fail transiently, and a cached error would outlive its cause for the rest of
    // the process. Two threads probing at once is fine: they compute the same answer,
    // and whichever `set` wins is the one both then read.
    let probed = probe_nvcc_options()?;
    let _ = OPTIONS.set(probed);
    Ok(OPTIONS
        .get()
        .expect("the options were stored just above")
        .as_slice())
}

fn probe_nvcc_options() -> Result<Vec<String>, MacroError> {
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
            .map_err(|error| {
                format!(
                    "could not detect a CUDA architecture with nvidia-smi ({error}), so set \
                 DESCEND_NVCC_FLAGS=-arch=compute_<capability>"
                )
            })?;
        if !output.status.success() {
            return Err(format!(
                "nvidia-smi could not detect a CUDA architecture, so set \
                 DESCEND_NVCC_FLAGS=-arch=compute_<capability>; nvidia-smi said: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )
            .into());
        }
        let capability = String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_owned();
        let mut components = capability.split('.');
        let major = components.next().unwrap_or("");
        let minor = components.next().unwrap_or("");
        if components.next().is_some()
            || major.is_empty()
            || minor.is_empty()
            || !major.chars().all(|character| character.is_ascii_digit())
            || !minor.chars().all(|character| character.is_ascii_digit())
        {
            return Err(format!(
                "nvidia-smi returned an invalid compute capability {capability:?}, so set \
                 DESCEND_NVCC_FLAGS=-arch=compute_<capability>"
            )
            .into());
        }
        options.push(format!("-arch=compute_{major}{minor}"));
    }
    Ok(options)
}
