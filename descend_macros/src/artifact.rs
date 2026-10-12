use crate::nat::Dimensions;
use crate::shared_memory::SharedMemoryBytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: Dimensions,
    pub block: Dimensions,
    pub shared_memory_bytes: SharedMemoryBytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ptx {
    pub text: String,
    pub entry: String,
}
