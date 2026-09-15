use crate::backend::CudaStream;
use crate::context::Context;
use std::sync::Arc;

pub struct Stream {
    pub(crate) inner: Arc<CudaStream>,
    pub(crate) context: Context,
}

use crate::error::HostError;

impl Stream {
    pub fn synchronize(&self) -> Result<(), HostError> {
        crate::backend::bind_context_to_thread(&self.context.inner)
            .map_err(|e| HostError::driver(e, "bind context for stream synchronize"))?;
        crate::backend::synchronize_stream(&self.inner)
            .map_err(|e| HostError::driver(e, "stream synchronize"))
    }
}
