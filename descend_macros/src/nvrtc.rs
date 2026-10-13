use crate::diag::MacroError;
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const SUCCESS: c_int = 0;
const FLAGS_ENVIRONMENT_VARIABLE: &str = "DESCEND_NVRTC_FLAGS";
const HEADER_NAME: &str = "descend.cuh";
const PROGRAM_NAME: &str = "descend_kernel.cu";

const TOOLKIT_ENVIRONMENT_VARIABLES: [&str; 2] = ["CUDA_TOOLKIT_PATH", "CUDA_HOME"];
const MINIMUM_CUDA_VERSION: u32 = 13000;

const LIBRARY_DIRECTORIES: &[&str] = &["lib", "lib64"];

type ResultCode = c_int;
type Program = *mut c_void;

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

pub struct Toolchain {
    nvrtc: Nvrtc,
    options: Vec<String>,
}

impl Toolchain {
    pub fn compile(&self, device_code: &str, header: &str) -> Result<String, MacroError> {
        self.nvrtc
            .compile(device_code, HEADER_NAME, header, &self.options)
    }
}

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
    let root = toolkit_root(|name| std::env::var_os(name))?;

    let include_path = root.join("include");
    let cuda_version = require_supported_cuda_version(&include_path)?;

    let options = resolve_options(|name| std::env::var_os(name), &include_path)?;
    let library_path = find_library(&root, &library_names(cuda_version))?;
    let nvrtc = Nvrtc::load(&library_path)?;

    Ok(Toolchain { nvrtc, options })
}

fn toolkit_root(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<PathBuf, MacroError> {
    for name in TOOLKIT_ENVIRONMENT_VARIABLES {
        let Some(value) = lookup(name) else {
            continue;
        };
        if value.is_empty() {
            return Err(format!(
                "{name} is set but empty. Set it to a CUDA {}+ toolkit directory",
                format_cuda_version(MINIMUM_CUDA_VERSION)
            )
            .into());
        }

        return Ok(PathBuf::from(value));
    }

    Err(format!(
        "the CUDA toolkit cannot be located: {} is not set. Set one of them to the toolkit \
         root -- the directory holding both lib/libnvrtc.so and include/ -- as in \
         CUDA_TOOLKIT_PATH=/usr/local/cuda",
        toolkit_variable_names()
    )
    .into())
}

fn toolkit_variable_names() -> String {
    TOOLKIT_ENVIRONMENT_VARIABLES.join(" or ")
}

fn require_supported_cuda_version(include_path: &Path) -> Result<u32, MacroError> {
    let cuda_version = cuda_version(include_path)?;
    if cuda_version >= MINIMUM_CUDA_VERSION {
        return Ok(cuda_version);
    }

    Err(format!(
        "{} reports CUDA {}, but compiling with NVRTC needs at least CUDA {}. Point {} at a \
         newer toolkit",
        include_path.join("cuda.h").display(),
        format_cuda_version(cuda_version),
        format_cuda_version(MINIMUM_CUDA_VERSION),
        toolkit_variable_names()
    )
    .into())
}

fn cuda_version(include_path: &Path) -> Result<u32, MacroError> {
    let header = include_path.join("cuda.h");
    let source = std::fs::read_to_string(&header).map_err(|error| {
        format!(
            "{} could not be read ({error}), so the CUDA version is unknown. {} has to name a \
             toolkit root whose include/ holds the CUDA headers descend.cuh includes",
            header.display(),
            toolkit_variable_names()
        )
    })?;

    source
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
                _ => None,
            }
        })
        .ok_or_else(|| {
            format!(
                "{} has no `#define CUDA_VERSION`, so the CUDA version is unknown",
                header.display()
            )
            .into()
        })
}

fn format_cuda_version(version: u32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

fn library_names(cuda_version: u32) -> Vec<String> {
    vec![
        "libnvrtc.so".to_string(),
        format!("libnvrtc.so.{}", cuda_version / 1000),
    ]
}

fn find_library(root: &Path, library_names: &[String]) -> Result<PathBuf, MacroError> {
    let mut tried = Vec::new();
    for directory in LIBRARY_DIRECTORIES {
        for name in library_names {
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

fn resolve_options(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
    include_path: &Path,
) -> Result<Vec<String>, MacroError> {
    let mut options = mandatory_flags(lookup)?;

    options.push("-default-device".to_string());

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

    Ok(options)
}

fn mandatory_flags(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<Vec<String>, MacroError> {
    let Some(value) = lookup(FLAGS_ENVIRONMENT_VARIABLE) else {
        return Err(flags_error("is not set"));
    };

    let options = value
        .to_string_lossy()
        .split_whitespace()
        .map(|option| option.to_string())
        .collect::<Vec<_>>();
    if options.is_empty() {
        return Err(flags_error("is set but empty"));
    }

    Ok(options)
}

fn flags_error(state: &str) -> MacroError {
    format!(
        "{FLAGS_ENVIRONMENT_VARIABLE} {state}, so NVRTC would be given no compilation flags. \
         Set it to every flag NVRTC needs for this toolkit and GPU, including the \
         architecture, as in {FLAGS_ENVIRONMENT_VARIABLE}=-arch=compute_86"
    )
    .into()
}

struct Nvrtc {
    create_program: CreateProgramFn,
    compile_program: CompileProgramFn,
    get_ptx_size: GetPtxSizeFn,
    get_ptx: GetPtxFn,
    get_log_size: GetLogSizeFn,
    get_log: GetLogFn,
    destroy_program: DestroyProgramFn,
    get_error_string: GetErrorStringFn,
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

    fn temporary_directory() -> tempfile::TempDir {
        tempfile::tempdir().expect("a temporary directory must be creatable")
    }

    fn toolkit_reporting(cuda_version: u32) -> tempfile::TempDir {
        let root = temporary_directory();
        let include_directory = root.path().join("include");
        std::fs::create_dir_all(&include_directory).expect("the directory must be creatable");
        std::fs::write(
            include_directory.join("cuda.h"),
            format!("#define CUDA_VERSION {cuda_version}\n"),
        )
        .expect("the header must be writable");

        root
    }

    #[test]
    fn the_library_is_found_in_either_library_directory() {
        for directory in LIBRARY_DIRECTORIES {
            let root = temporary_directory();
            let library_directory = root.path().join(directory);
            std::fs::create_dir_all(&library_directory).expect("the directory must be creatable");
            let library = library_directory.join("libnvrtc.so");
            std::fs::write(&library, []).expect("the file must be writable");

            assert_eq!(
                find_library(root.path(), &library_names(13020))
                    .expect("the library must be found"),
                library,
                "{directory}/ was not searched"
            );
        }
    }

    #[test]
    fn a_versioned_library_is_found_when_there_is_no_unversioned_one() {
        let root = temporary_directory();
        let library_directory = root.path().join("lib");
        std::fs::create_dir_all(&library_directory).expect("the directory must be creatable");
        let library = library_directory.join("libnvrtc.so.13");
        std::fs::write(&library, []).expect("the file must be writable");

        assert_eq!(
            find_library(root.path(), &library_names(13020)).expect("the library must be found"),
            library
        );
    }

    #[test]
    fn a_directory_shaped_like_the_library_is_not_one() {
        let root = temporary_directory();
        std::fs::create_dir_all(root.path().join("lib").join("libnvrtc.so"))
            .expect("the directory must be creatable");

        assert!(find_library(root.path(), &library_names(13020)).is_err());
    }

    #[test]
    fn a_toolkit_without_a_library_names_every_path_it_tried() {
        let root = temporary_directory();
        let error = find_library(root.path(), &library_names(13020)).unwrap_err();

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
    fn the_toolkit_root_is_the_one_the_environment_names() {
        let root = toolkit_root(environment(&[("CUDA_TOOLKIT_PATH", "/usr/local/cuda")]))
            .expect("a toolkit must be found");

        assert_eq!(root, PathBuf::from("/usr/local/cuda"));
    }

    #[test]
    fn cuda_home_names_the_toolkit_when_the_toolkit_path_is_absent() {
        let root = toolkit_root(environment(&[("CUDA_HOME", "/opt/cuda")]))
            .expect("a toolkit must be found");

        assert_eq!(root, PathBuf::from("/opt/cuda"));
    }

    #[test]
    fn the_toolkit_path_wins_over_cuda_home() {
        let root = toolkit_root(environment(&[
            ("CUDA_TOOLKIT_PATH", "/usr/local/cuda"),
            ("CUDA_HOME", "/opt/cuda"),
        ]))
        .expect("a toolkit must be found");

        assert_eq!(root, PathBuf::from("/usr/local/cuda"));
    }

    #[test]
    fn an_empty_value_is_an_error_naming_that_variable() {
        let error = toolkit_root(environment(&[("CUDA_HOME", "")])).unwrap_err();

        assert!(error.message.contains("CUDA_HOME"), "{}", error.message);
    }

    #[test]
    fn a_missing_toolkit_names_both_variables_to_set() {
        let error = toolkit_root(environment(&[])).unwrap_err();

        for name in TOOLKIT_ENVIRONMENT_VARIABLES {
            assert!(error.message.contains(name), "{}", error.message);
        }
    }

    #[test]
    fn the_cuda_version_is_read_from_the_toolkit_header() {
        let root = toolkit_reporting(13020);

        assert_eq!(
            require_supported_cuda_version(&root.path().join("include"))
                .expect("CUDA 13.2 is supported"),
            13020
        );
    }

    #[test]
    fn a_toolkit_below_the_floor_is_refused_with_both_versions_named() {
        let root = toolkit_reporting(12080);

        let error = require_supported_cuda_version(&root.path().join("include")).unwrap_err();

        assert!(error.message.contains("12.8"), "{}", error.message);
        assert!(error.message.contains("13.0"), "{}", error.message);
    }

    #[test]
    fn a_toolkit_without_a_header_is_refused() {
        let root = temporary_directory();

        assert!(require_supported_cuda_version(&root.path().join("include")).is_err());
    }

    #[test]
    fn a_header_without_a_version_define_is_refused() {
        let root = temporary_directory();
        let include_directory = root.path().join("include");
        std::fs::create_dir_all(&include_directory).expect("the directory must be creatable");
        std::fs::write(include_directory.join("cuda.h"), "#define CUDA_API\n")
            .expect("the header must be writable");

        let error = require_supported_cuda_version(&include_directory).unwrap_err();

        assert!(error.message.contains("CUDA_VERSION"), "{}", error.message);
    }

    #[test]
    fn the_library_names_follow_the_toolkit_major_version() {
        assert_eq!(
            library_names(13020),
            ["libnvrtc.so".to_string(), "libnvrtc.so.13".to_string()]
        );
    }

    #[test]
    fn the_readers_flags_come_first_and_the_macro_adds_its_own_after() {
        let root = temporary_directory();
        let include_path = root.path().join("include");

        let options = resolve_options(
            environment(&[(FLAGS_ENVIRONMENT_VARIABLE, "-arch=compute_86 --ftz=true")]),
            &include_path,
        )
        .expect("the flags are set");

        assert_eq!(
            options,
            [
                "-arch=compute_86".to_string(),
                "--ftz=true".to_string(),
                "-default-device".to_string(),
                format!("-I{}", include_path.display()),
            ]
        );
    }

    #[test]
    fn missing_flags_are_an_error_naming_the_variable_and_an_architecture() {
        let error =
            resolve_options(environment(&[]), Path::new("/usr/local/cuda/include")).unwrap_err();

        assert!(
            error.message.contains(FLAGS_ENVIRONMENT_VARIABLE),
            "{}",
            error.message
        );
        assert!(
            error.message.contains("-arch=compute_86"),
            "{}",
            error.message
        );
    }

    #[test]
    fn flags_that_carry_no_option_are_an_error() {
        for flags in ["", "   "] {
            let error = resolve_options(
                environment(&[(FLAGS_ENVIRONMENT_VARIABLE, flags)]),
                Path::new("/usr/local/cuda/include"),
            )
            .unwrap_err();

            assert!(
                error.message.contains("empty"),
                "{flags:?}: {}",
                error.message
            );
        }
    }
}
