//! Where compiled PTX is kept between builds, and what identifies an entry.

use std::io::Write;

/// Everything that determines the PTX one compilation produces.
///
/// The header is a field of its own because it is no longer part of the device source:
/// NVRTC receives it in memory, so a changed `descend.cuh` would otherwise leave every
/// cached kernel looking current.
pub struct CacheKey<'key> {
    pub device_code: &'key str,
    pub header: &'key str,
    pub options: &'key [String],
    pub compiler_version: &'key str,
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

fn cache_path(key: &CacheKey) -> Option<std::path::PathBuf> {
    if std::env::var_os("DESCEND_NO_CACHE").is_some() {
        return None;
    }

    Some(verified_cache_directory()?.join(format!("{:016x}.ptx", cache_key(key))))
}

/// FNV-1a over the device source, the header, the compiler options, and the compiler
/// version.
///
/// `DefaultHasher` would be one line, but it is documented as unstable across Rust
/// releases, so a toolchain bump silently invalidates every cached kernel and the first
/// build afterwards pays for the compilation all over again. A hash spelled out here is
/// stable by construction, which is the only property the value needs: it names a file,
/// and is not a security boundary -- [`verified_cache_directory`] is what makes the file
/// trustworthy.
fn cache_key(key: &CacheKey) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn mix(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(PRIME);
        }
    }

    let mut hash = OFFSET_BASIS;
    mix(&mut hash, key.device_code.as_bytes());
    mix(&mut hash, &[0]);
    mix(&mut hash, key.header.as_bytes());
    for option in key.options {
        // A separator, so `["-a", "-b"]` and `["-a-b"]` are different keys.
        mix(&mut hash, &[0]);
        mix(&mut hash, option.as_bytes());
    }
    mix(&mut hash, &[0]);
    mix(&mut hash, key.compiler_version.as_bytes());
    hash
}

pub fn cached_ptx(key: &CacheKey) -> Option<String> {
    std::fs::read_to_string(cache_path(key)?).ok()
}

pub fn store_ptx(key: &CacheKey, ptx: &str) {
    if let Some(path) = cache_path(key) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key<'key>(
        device_code: &'key str,
        header: &'key str,
        options: &'key [String],
        compiler_version: &'key str,
    ) -> CacheKey<'key> {
        CacheKey {
            device_code,
            header,
            options,
            compiler_version,
        }
    }

    fn options(list: &[&str]) -> Vec<String> {
        list.iter().map(|option| (*option).to_string()).collect()
    }

    /// NVRTC receives the header in memory, so it is no longer part of the device source.
    /// A changed `descend.cuh` has to move the key on its own, or every cached kernel
    /// keeps looking current after the header it was compiled against is gone.
    #[test]
    fn a_changed_header_changes_the_key() {
        assert_ne!(
            cache_key(&key("code", "header as it was", &[], "13.2")),
            cache_key(&key("code", "header as it is now", &[], "13.2"))
        );
    }

    #[test]
    fn a_changed_compiler_version_changes_the_key() {
        assert_ne!(
            cache_key(&key("code", "header", &[], "13.2")),
            cache_key(&key("code", "header", &[], "13.1"))
        );
    }

    #[test]
    fn the_same_compilation_hashes_the_same() {
        let one = cache_key(&key(
            "code",
            "header",
            &options(&["-arch=compute_86"]),
            "13.2",
        ));
        let two = cache_key(&key(
            "code",
            "header",
            &options(&["-arch=compute_86"]),
            "13.2",
        ));
        assert_eq!(one, two);
    }

    #[test]
    fn option_boundaries_survive_into_the_key() {
        assert_ne!(
            cache_key(&key("code", "header", &options(&["-a", "-b"]), "13.2")),
            cache_key(&key("code", "header", &options(&["-a-b"]), "13.2"))
        );
    }
}
