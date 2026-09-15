use crate::stream::Stream;
use cuda_core::simt::CudaContext;
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
        let inner = self.inner.new_stream()
            .map_err(|e| HostError::driver(e, "stream creation"))?;
        Ok(Stream {
            inner,
            context: self.clone(),
        })
    }
}
