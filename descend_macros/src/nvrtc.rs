//! Compiling device code to PTX through NVRTC.
//!
//! NVRTC compiles from memory: the source and `descend.cuh` are handed to it as strings,
//! so nothing is written to disk and no subprocess is spawned.

use crate::diag::MacroError;
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const SUCCESS: c_int = 0;
const FLAGS_ENVIRONMENT_VARIABLE: &str = "DESCEND_NVRTC_FLAGS";
const HEADER_NAME: &str = "descend.cuh";
const PROGRAM_NAME: &str = "descend_kernel.cu";

/// The toolkit root, in precedence order.
///
/// `CUDA_PATH` is an ecosystem convention rather than a documented guarantee: NVIDIA's
/// Linux installation guide for 13.2 names the `/usr/local/cuda` symbolic link as the
/// toolkit location, reserves `CUDAToolkit_ROOT` for CMake, and defines no variable for
/// the root itself. That gap is why this crate has one of its own.
const TOOLKIT_ENVIRONMENT_VARIABLES: &[&str] = &["DESCEND_CUDA_PATH", "CUDA_PATH"];

const LIBRARY_DIRECTORIES: &[&str] = &["lib", "lib64"];
const LIBRARY_NAMES: &[&str] = &["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"];

type ResultCode = c_int;
type Program = *mut c_void;

type VersionFn = unsafe extern "C" fn(*mut c_int, *mut c_int) -> ResultCode;
type CreateProgramFn = unsafe extern "C" fn(
    *mut Program,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> ResultCode;
type CompileProgramFn = unsafe extern "C" fn(Program, c_int, *const *const c_char) -> ResultCode;
type GetPtxSizeFn = unsafe extern "C" fn(Program, *mut usize) -> ResultCode;
type GetPtxFn = unsafe extern "C" fn(Program, *mut c_char) -> ResultCode;
type GetLogSizeFn = unsafe extern "C" fn(Program, *mut usize) -> ResultCode;
type GetLogFn = unsafe extern "C" fn(Program, *mut c_char) -> ResultCode;
type DestroyProgramFn = unsafe extern "C" fn(*mut Program) -> ResultCode;
type GetErrorStringFn = unsafe extern "C" fn(ResultCode) -> *const c_char;

/// Everything a compilation needs, resolved once per expansion.
pub struct Toolchain {
    library_path: PathBuf,
    options: Vec<String>,
    version: String,
}

impl Toolchain {
    pub fn options(&self) -> &[String] {
        &self.options
    }

    /// The compiler version, which belongs in the PTX cache key: the same source under a
    /// different NVRTC is different PTX.
    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn compile(&self, device_code: &str, header: &str) -> Result<String, MacroError> {
        let nvrtc = Nvrtc::load(&self.library_path)?;
        nvrtc.compile(device_code, HEADER_NAME, header, &self.options)
    }
}

/// The toolchain, resolved at most once successfully per process.
///
/// Only a success is stored, so a transient `nvidia-smi` failure is not cached for the
/// rest of the expansion. Two threads resolving at once compute the same answer and
/// whichever `set` wins is the one both then read.
pub fn toolchain() -> Result<&'static Toolchain, MacroError> {
    static TOOLCHAIN: OnceLock<Toolchain> = OnceLock::new();
    if let Some(resolved) = TOOLCHAIN.get() {
        return Ok(resolved);
    }

    let discovered = discover()?;
    let _ = TOOLCHAIN.set(discovered);
    Ok(TOOLCHAIN
        .get()
        .expect("the toolchain was stored just above"))
}

fn discover() -> Result<Toolchain, MacroError> {
    let (root, named_by) = toolkit_root(|name| std::env::var_os(name))?;

    let library_path = find_library(&root)?;
    let include_path = root.join("include");
    if !include_path.is_dir() {
        return Err(format!(
            "{named_by} points at {}, which has no include/ directory, so NVRTC could not \
             resolve the CUDA headers descend.cuh includes",
            root.display()
        )
        .into());
    }

    let version = Nvrtc::load(&library_path)?.version()?;
    let options = resolve_options(&include_path)?;

    Ok(Toolchain {
        library_path,
        options,
        version,
    })
}

/// The toolkit root the environment names, and which variable named it.
///
/// The lookup is a parameter so the precedence can be tested without touching the
/// process environment, which the rest of the test run reads.
fn toolkit_root(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<(PathBuf, &'static str), MacroError> {
    for name in TOOLKIT_ENVIRONMENT_VARIABLES {
        if let Some(value) = lookup(name).filter(|value| !value.is_empty()) {
            return Ok((PathBuf::from(value), name));
        }
    }

    Err(format!(
        "neither {} is set, so the CUDA toolkit cannot be located. Set one of them to the \
         toolkit root -- the directory holding both lib/libnvrtc.so and include/ -- as in \
         DESCEND_CUDA_PATH=/usr/local/cuda",
        TOOLKIT_ENVIRONMENT_VARIABLES.join(", ")
    )
    .into())
}

fn find_library(root: &Path) -> Result<PathBuf, MacroError> {
    let mut tried = Vec::new();
    for directory in LIBRARY_DIRECTORIES {
        for name in LIBRARY_NAMES {
            let candidate = root.join(directory).join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
            tried.push(candidate);
        }
    }

    Err(format!(
        "no NVRTC library under {}, looked for: {}",
        root.display(),
        tried
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .into())
}

fn resolve_options(include_path: &Path) -> Result<Vec<String>, MacroError> {
    let flags_from_environment = std::env::var(FLAGS_ENVIRONMENT_VARIABLE).unwrap_or_default();
    let mut options: Vec<String> = flags_from_environment
        .split_whitespace()
        .map(|option| option.to_string())
        .collect();

    options.push(format!(
        "-I{}",
        include_path
            .to_str()
            .ok_or_else(|| {
                format!(
                    "the CUDA include directory {} is not valid UTF-8, so it cannot be passed \
                     to NVRTC as an -I option",
                    include_path.display()
                )
            })?
            .replace('\\', "\\\\")
    ));

    if !options.iter().any(|option| option.starts_with("-arch=")) {
        options.push(probe_architecture()?);
    }
    Ok(options)
}

/// The compute capability of the first GPU, as an NVRTC `-arch=` option.
fn probe_architecture() -> Result<String, MacroError> {
    const OVERRIDE_HINT: &str = "set DESCEND_NVRTC_FLAGS=-arch=compute_<capability>";

    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .map_err(|error| {
            format!("could not detect a CUDA architecture with nvidia-smi ({error}), so {OVERRIDE_HINT}")
        })?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi could not detect a CUDA architecture, so {OVERRIDE_HINT}; nvidia-smi \
             said: {}",
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
            "nvidia-smi returned an invalid compute capability {capability:?}, so \
             {OVERRIDE_HINT}"
        )
        .into());
    }

    Ok(format!("-arch=compute_{major}{minor}"))
}

struct Nvrtc {
    version: VersionFn,
    create_program: CreateProgramFn,
    compile_program: CompileProgramFn,
    get_ptx_size: GetPtxSizeFn,
    get_ptx: GetPtxFn,
    get_log_size: GetLogSizeFn,
    get_log: GetLogFn,
    destroy_program: DestroyProgramFn,
    get_error_string: GetErrorStringFn,
    // Held for its `Drop` alone, and declared last so it outlives every pointer above.
    _library: Library,
}

impl Nvrtc {
    fn load(path: &Path) -> Result<Nvrtc, MacroError> {
        // SAFETY: `Library::new` is `dlopen`, which runs no code from the library beyond
        // its initialisers. The path comes from `find_library`, which accepted it only
        // because it is a regular file under the toolkit root the reader named.
        let library = unsafe { Library::new(path) }
            .map_err(|error| format!("could not load {}: {error}", path.display()))?;

        Ok(Nvrtc {
            version: bind(&library, path, "nvrtcVersion")?,
            create_program: bind(&library, path, "nvrtcCreateProgram")?,
            compile_program: bind(&library, path, "nvrtcCompileProgram")?,
            get_ptx_size: bind(&library, path, "nvrtcGetPTXSize")?,
            get_ptx: bind(&library, path, "nvrtcGetPTX")?,
            get_log_size: bind(&library, path, "nvrtcGetProgramLogSize")?,
            get_log: bind(&library, path, "nvrtcGetProgramLog")?,
            destroy_program: bind(&library, path, "nvrtcDestroyProgram")?,
            get_error_string: bind(&library, path, "nvrtcGetErrorString")?,
            _library: library,
        })
    }

    fn version(&self) -> Result<String, MacroError> {
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: two out-parameters, written on success and read only then.
        let result = unsafe { (self.version)(&mut major, &mut minor) };
        if result != SUCCESS {
            return Err(self.failure("nvrtcVersion", result));
        }
        Ok(format!("{major}.{minor}"))
    }

    fn compile(
        &self,
        device_code: &str,
        header_name: &str,
        header: &str,
        options: &[String],
    ) -> Result<String, MacroError> {
        let source = c_string("the generated CUDA", device_code)?;
        let program_name = c_string(
            "the file name NVRTC reports diagnostics against",
            PROGRAM_NAME,
        )?;
        let header_text = c_string("descend.cuh", header)?;
        let header_file_name = c_string("the descend.cuh file name", header_name)?;

        let mut owned_options = Vec::with_capacity(options.len());
        for option in options {
            owned_options.push(c_string("an NVRTC option", option)?);
        }
        let option_pointers: Vec<*const c_char> =
            owned_options.iter().map(|option| option.as_ptr()).collect();

        let mut program: Program = std::ptr::null_mut();
        // SAFETY: the out-parameter is written on success, which is checked before the
        // program handle is used or destroyed. Every pointer argument outlives the call,
        // because the `CString`s and the two vectors are held in this scope.
        let created = unsafe {
            (self.create_program)(
                &mut program,
                source.as_ptr(),
                program_name.as_ptr(),
                1,
                [header_text.as_ptr()].as_ptr(),
                [header_file_name.as_ptr()].as_ptr(),
            )
        };
        if created != SUCCESS {
            return Err(self.failure("nvrtcCreateProgram", created));
        }
        let owned = OwnedProgram {
            handle: program,
            destroy: self.destroy_program,
        };

        // SAFETY: `option_pointers` is a contiguous array of `option_pointers.len()`
        // valid C strings, all alive until the end of this scope.
        let compiled = unsafe {
            (self.compile_program)(
                owned.handle,
                option_pointers.len() as c_int,
                option_pointers.as_ptr(),
            )
        };
        let log = self.program_log(owned.handle);
        if compiled != SUCCESS {
            return Err(format!(
                "NVRTC rejected the generated device code ({}):\n{}",
                self.describe(compiled),
                log.trim()
            )
            .into());
        }

        self.ptx(owned.handle).map_err(|reason| {
            format!(
                "NVRTC compiled the kernel but its PTX could not be read: {reason}\n{}",
                log.trim()
            )
            .into()
        })
    }

    fn ptx(&self, program: Program) -> Result<String, String> {
        let mut size = 0usize;
        // SAFETY: one out-parameter, read only after a success is checked.
        let sized = unsafe { (self.get_ptx_size)(program, &mut size) };
        if sized != SUCCESS {
            return Err(self.describe(sized));
        }
        if size == 0 {
            return Err("NVRTC reported a PTX size of zero".to_string());
        }

        let mut buffer = vec![0u8; size];
        // SAFETY: `buffer` holds exactly the `size` bytes the call above reported, which
        // NVRTC documents as including the terminating NUL.
        let written = unsafe { (self.get_ptx)(program, buffer.as_mut_ptr().cast::<c_char>()) };
        if written != SUCCESS {
            return Err(self.describe(written));
        }

        let terminated = buffer.split(|byte| *byte == 0).next().unwrap_or(&buffer);
        String::from_utf8(terminated.to_vec()).map_err(|error| error.to_string())
    }

    fn program_log(&self, program: Program) -> String {
        let mut size = 0usize;
        // SAFETY: one out-parameter; on failure the size stays 0 and the log is empty.
        let sized = unsafe { (self.get_log_size)(program, &mut size) };
        if sized != SUCCESS || size == 0 {
            return String::new();
        }

        let mut buffer = vec![0u8; size];
        // SAFETY: `buffer` holds exactly the `size` bytes reported above.
        let written = unsafe { (self.get_log)(program, buffer.as_mut_ptr().cast::<c_char>()) };
        if written != SUCCESS {
            return String::new();
        }

        let terminated = buffer.split(|byte| *byte == 0).next().unwrap_or(&buffer);
        String::from_utf8_lossy(terminated).into_owned()
    }

    fn describe(&self, result: ResultCode) -> String {
        // SAFETY: the returned pointer is to a static string owned by the library, which
        // is alive for as long as `self` is.
        let message = unsafe { (self.get_error_string)(result) };
        if message.is_null() {
            return format!("NVRTC error {result}");
        }
        // SAFETY: NVRTC documents the string as NUL-terminated ASCII.
        let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
        format!("{text} ({result})")
    }

    fn failure(&self, what: &str, result: ResultCode) -> MacroError {
        format!("{what} failed: {}", self.describe(result)).into()
    }
}

/// A program handle that is destroyed whichever way this function returns.
struct OwnedProgram {
    handle: Program,
    destroy: DestroyProgramFn,
}

impl Drop for OwnedProgram {
    fn drop(&mut self) {
        // SAFETY: `handle` came from a successful `nvrtcCreateProgram` and is destroyed
        // exactly once, because `OwnedProgram` is not cloned and owns it alone.
        unsafe { (self.destroy)(&mut self.handle) };
    }
}

fn c_string(what: &str, text: &str) -> Result<CString, MacroError> {
    CString::new(text).map_err(|_| format!("{what} contains an interior NUL byte").into())
}

/// One symbol out of the loaded library.
///
/// `libloading` wants the name NUL-terminated, and the terminator is added here so no
/// call site has to spell it.
fn bind<Signature: Copy>(
    library: &Library,
    path: &Path,
    name: &str,
) -> Result<Signature, MacroError> {
    let mut terminated = name.as_bytes().to_vec();
    terminated.push(0);

    // SAFETY: the symbol is cast to the signature the NVRTC header declares for it. A
    // mismatch is undefined behaviour, and the signatures above are transcribed from
    // `nvrtc.h` in the toolkit this crate is compiled against.
    let symbol: Symbol<Signature> = unsafe { library.get(&terminated) }.map_err(|error| {
        format!(
            "`{name}` is missing from {}: {error}. A toolkit older than CUDA 12 does not \
             provide the NVRTC API this crate uses",
            path.display()
        )
    })?;
    Ok(*symbol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn empty_toolkit_root() -> tempfile::TempDir {
        tempfile::tempdir().expect("a temporary directory must be creatable")
    }

    #[test]
    fn the_library_is_found_in_either_library_directory() {
        for directory in LIBRARY_DIRECTORIES {
            let root = empty_toolkit_root();
            let library_directory = root.path().join(directory);
            std::fs::create_dir_all(&library_directory).expect("the directory must be creatable");
            let library = library_directory.join("libnvrtc.so");
            std::fs::write(&library, []).expect("the file must be writable");

            assert_eq!(
                find_library(root.path()).expect("the library must be found"),
                library,
                "{directory}/ was not searched"
            );
        }
    }

    #[test]
    fn a_versioned_library_is_found_when_there_is_no_unversioned_one() {
        let root = empty_toolkit_root();
        let library_directory = root.path().join("lib");
        std::fs::create_dir_all(&library_directory).expect("the directory must be creatable");
        let library = library_directory.join("libnvrtc.so.13");
        std::fs::write(&library, []).expect("the file must be writable");

        assert_eq!(
            find_library(root.path()).expect("the library must be found"),
            library
        );
    }

    #[test]
    fn a_directory_shaped_like_the_library_is_not_one() {
        let root = empty_toolkit_root();
        std::fs::create_dir_all(root.path().join("lib").join("libnvrtc.so"))
            .expect("the directory must be creatable");

        assert!(find_library(root.path()).is_err());
    }

    /// A missing toolkit is the failure a reader hits first, so the message has to say
    /// what was looked for rather than only that nothing was found.
    #[test]
    fn a_toolkit_without_a_library_names_every_path_it_tried() {
        let root = empty_toolkit_root();
        let error = find_library(root.path()).unwrap_err();

        for directory in LIBRARY_DIRECTORIES {
            assert!(
                error.message.contains(&format!("{directory}/libnvrtc.so")),
                "the {directory}/ candidates are missing from: {}",
                error.message
            );
        }
        assert!(
            error.message.contains(&root.path().display().to_string()),
            "the root it searched under is missing from: {}",
            error.message
        );
    }

    /// A stand-in for the process environment, so the precedence can be tested without
    /// setting variables the rest of the test run reads.
    fn environment(entries: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let owned: Vec<(String, OsString)> = entries
            .iter()
            .map(|(name, value)| ((*name).to_string(), OsString::from(*value)))
            .collect();

        move |name: &str| {
            owned
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        }
    }

    #[test]
    fn the_project_variable_wins_over_the_ecosystem_one() {
        let (root, named_by) = toolkit_root(environment(&[
            ("DESCEND_CUDA_PATH", "/chosen"),
            ("CUDA_PATH", "/ignored"),
        ]))
        .expect("a toolkit must be found");

        assert_eq!(root, PathBuf::from("/chosen"));
        assert_eq!(named_by, "DESCEND_CUDA_PATH");
    }

    #[test]
    fn the_ecosystem_variable_is_used_when_the_project_one_is_absent() {
        let (root, named_by) = toolkit_root(environment(&[("CUDA_PATH", "/usr/local/cuda")]))
            .expect("a toolkit must be found");

        assert_eq!(root, PathBuf::from("/usr/local/cuda"));
        assert_eq!(named_by, "CUDA_PATH");
    }

    #[test]
    fn an_empty_value_counts_as_absent() {
        let (root, named_by) = toolkit_root(environment(&[
            ("DESCEND_CUDA_PATH", ""),
            ("CUDA_PATH", "/usr/local/cuda"),
        ]))
        .expect("a toolkit must be found");

        assert_eq!(root, PathBuf::from("/usr/local/cuda"));
        assert_eq!(named_by, "CUDA_PATH");
    }

    /// A reader who has set neither has to be told which names would have worked.
    #[test]
    fn no_toolkit_variable_names_every_one_that_was_tried() {
        let error = toolkit_root(environment(&[])).unwrap_err();

        for name in TOOLKIT_ENVIRONMENT_VARIABLES {
            assert!(
                error.message.contains(name),
                "`{name}` is missing from: {}",
                error.message
            );
        }
    }
}
