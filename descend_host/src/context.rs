use crate::backend::CudaContext;
use crate::stream::Stream;
use std::sync::Arc;

use crate::error::HostError;

#[derive(Clone)]
pub struct Context {
    pub(crate) inner: Arc<CudaContext>,
}

impl Context {
    pub fn new(device_id: u32) -> Result<Self, HostError> {
        let inner = CudaContext::new(device_id as usize)
            .map_err(|e| HostError::driver(e, "context creation"))?;
        Ok(Self { inner })
    }

    pub fn new_stream(&self) -> Result<Stream, HostError> {
        let inner = self
            .inner
            .new_stream()
            .map_err(|e| HostError::driver(e, "stream creation"))?;
        Ok(Stream {
            inner,
            context: self.clone(),
        })
    }

    pub fn check_deferred_errors(&self) -> Result<(), HostError> {
        self.inner
            .check_err()
            .map_err(|e| HostError::driver(e, "error from context"))
    }

    pub fn synchronize(&self) -> Result<(), HostError> {
        self.inner
            .synchronize()
            .map_err(|e| HostError::driver(e, "context synchronize"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_buffer::GpuBuffer;
    use crate::kernel_launch::KernelLauncher;

    const NOOP_PTX: &str =
        ".version 7.0\n.target sm_52\n.address_size 64\n\n.visible .entry noop()\n{\n\tret;\n}\n";

    #[test]
    fn a_healthy_context_reports_no_deferred_errors() {
        let ctx = Context::new(0).unwrap();
        ctx.check_deferred_errors().unwrap();
    }

    #[test]
    fn normal_resource_drops_record_no_deferred_errors() {
        let ctx = Context::new(0).unwrap();

        {
            let stream = ctx.new_stream().unwrap();
            let launcher = KernelLauncher::new(&stream, NOOP_PTX).unwrap();
            let buffer = GpuBuffer::from_host(&[7i32; 64], &stream).unwrap();
            assert_eq!(*buffer.to_host(&stream).unwrap(), [7i32; 64]);
            drop(launcher);
        }

        ctx.synchronize().unwrap();
        ctx.check_deferred_errors()
            .expect("dropping a stream, module and buffer must not record an error");
    }

    #[test]
    fn an_idle_context_synchronizes() {
        let ctx = Context::new(0).unwrap();
        ctx.synchronize().unwrap();
    }

    #[test]
    fn synchronize_is_repeatable_after_a_transfer() {
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let buffer = GpuBuffer::from_host(&[1u32; 32], &stream).unwrap();
        ctx.synchronize().unwrap();
        stream.synchronize().unwrap();
        ctx.synchronize().unwrap();

        assert_eq!(*buffer.to_host(&stream).unwrap(), [1u32; 32]);
    }
}
