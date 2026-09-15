use std::fmt;

#[derive(Debug)]
pub enum HostError {
    Driver(Box<dyn std::error::Error + Send + Sync>, &'static str),
    NulError(std::ffi::NulError, &'static str),
    InvalidSize(&'static str),
    InvalidAlignment(&'static str),
    ContextMismatch(&'static str),
    ConversionError(&'static str),
}

impl HostError {
    pub(crate) fn driver<E: Into<Box<dyn std::error::Error + Send + Sync>>>(e: E, op: &'static str) -> Self {
        HostError::Driver(e.into(), op)
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::Driver(e, op) => write!(f, "Driver error during {}: {:?}", op, e),
            HostError::NulError(e, op) => write!(f, "NUL byte in string during {}: {}", op, e),
            HostError::InvalidSize(msg) => write!(f, "Invalid size: {}", msg),
            HostError::InvalidAlignment(msg) => write!(f, "Invalid alignment: {}", msg),
            HostError::ContextMismatch(msg) => write!(f, "Context mismatch: {}", msg),
            HostError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
        }
    }
}

impl std::error::Error for HostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HostError::Driver(e, _) => Some(e.as_ref()),
            HostError::NulError(e, _) => Some(e),
            _ => None,
        }
    }
}
