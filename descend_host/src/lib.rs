mod backend;
pub mod context;
pub mod error;
pub mod gpu_buffer;
pub mod kernel_launch;
pub mod stream;
pub mod traits;

pub use context::Context;
pub use gpu_buffer::GpuBuffer;
pub use kernel_launch::{Kernel, KernelLaunchConfig, KernelLauncher};
pub use stream::Stream;
pub use traits::DescendType;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiple_streams() {
        let ctx = Context::new(0).unwrap();
        let s1 = ctx.new_stream().unwrap();
        let s2 = ctx.new_stream().unwrap();
        let h1 = backend::raw_stream_handle(&s1.inner);
        let h2 = backend::raw_stream_handle(&s2.inner);
        assert_ne!(h1, 0, "s1 should not be stream 0");
        assert_ne!(h2, 0, "s2 should not be stream 0");
        assert_ne!(h1, h2, "Streams should be different");
    }
}
