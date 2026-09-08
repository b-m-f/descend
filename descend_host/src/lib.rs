pub use cuda_core;

pub mod context;
pub mod gpu_buffer;
pub mod kernel_launch;
pub mod stream;

pub use context::Context;
pub use gpu_buffer::GpuBuffer;
pub use kernel_launch::{KernelLaunchConfig, KernelLauncher, Kernel};
pub use stream::Stream;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiple_streams() {
        let ctx = Context::new(0);
        let s1 = ctx.new_stream();
        let s2 = ctx.new_stream();
        assert_ne!(
            s1.inner.cu_stream() as usize,
            0,
            "s1 should not be stream 0"
        );
        assert_ne!(
            s2.inner.cu_stream() as usize,
            0,
            "s2 should not be stream 0"
        );
        assert_ne!(
            s1.inner.cu_stream(),
            s2.inner.cu_stream(),
            "Streams should be different"
        );
    }
}
