use crate::nat::Dimensions;
use crate::shared_memory::SharedMemoryBytes;

/// What a launch needs, decided before anything is compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: Dimensions,
    pub block: Dimensions,
    pub shared_memory_bytes: SharedMemoryBytes,
}

/// The compiled kernel: the module to load and the symbol to find in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ptx {
    pub text: String,
    pub entry: String,
}
