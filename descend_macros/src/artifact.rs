use crate::nat::Dimensions;

pub struct LaunchConfig {
    pub grid: Dimensions,
    pub block: Dimensions,
    pub shared_memory_bytes: u64,
}

pub struct Ptx {
    pub text: String,
    pub entry: String,
}
