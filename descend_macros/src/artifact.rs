pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u64,
}

pub struct Ptx {
    pub ptx: String,
    pub entry: String,
}
