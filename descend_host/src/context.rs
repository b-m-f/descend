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
            .map_err(|e| HostError::driver(e, "resource release"))
    }

    pub fn synchronize(&self) -> Result<(), HostError> {
        self.inner
            .synchronize()
            .map_err(|e| HostError::driver(e, "context synchronize"))
    }
}
