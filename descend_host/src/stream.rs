use crate::context::Context;
use cuda_core::simt::CudaStream;
use std::sync::Arc;

pub struct Stream {
    pub(crate) inner: Arc<CudaStream>,
    pub(crate) context: Context,
}

impl Stream {
    pub fn synchronize(&self) {
        self.inner
            .synchronize()
            .expect("Failed to synchronize stream");
    }
}
