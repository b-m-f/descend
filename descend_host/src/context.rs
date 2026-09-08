use crate::stream::Stream;
use cuda_core::simt::CudaContext;
use std::sync::Arc;

#[derive(Clone)]
pub struct Context {
    pub(crate) inner: Arc<CudaContext>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Context {
    pub fn new(device_id: u32) -> Self {
        Self {
            inner: CudaContext::new(device_id as usize).expect("Failed to create CUDA context"),
        }
    }

    pub fn new_stream(&self) -> Stream {
        Stream {
            inner: self.inner.new_stream().expect("Failed to create stream"),
            context: self.clone(),
        }
    }
}
