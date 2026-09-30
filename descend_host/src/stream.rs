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
        self.inner
            .synchronize()
            .map_err(|e| HostError::driver(e, "stream synchronize"))
    }
}
