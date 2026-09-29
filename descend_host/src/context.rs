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
        let inner = crate::backend::new_context(device_id as usize)
            .map_err(|e| HostError::driver(e, "context creation"))?;
        Ok(Self { inner })
    }

    pub fn new_stream(&self) -> Result<Stream, HostError> {
        let inner = crate::backend::new_stream(&self.inner)
            .map_err(|e| HostError::driver(e, "stream creation"))?;
        Ok(Stream {
            inner,
            context: self.clone(),
        })
    }

    /// Reports and clears an error recorded while releasing a device resource.
    ///
    /// Destructors cannot return errors, so the backend records them in the
    /// context. Transfers, kernel launches and synchronization ignore recorded
    /// errors, so they never turn one into an unrelated returned failure. Unless
    /// taken here, a recorded error is reported by the next backend-managed
    /// operation on this context, such as stream creation or module loading.
    pub fn check_deferred_errors(&self) -> Result<(), HostError> {
        crate::backend::take_recorded_error(&self.inner)
            .map_err(|e| HostError::driver(e, "resource release"))
    }
}
