use std::fmt;

#[derive(Debug)]
pub enum HostError {
    Driver(Box<dyn std::error::Error + Send + Sync>, &'static str),
    NulError(std::ffi::NulError, &'static str),
    InvalidSize(&'static str),
    InvalidAlignment(&'static str),
    ContextMismatch(&'static str),
}

impl HostError {
    pub(crate) fn driver<E: Into<Box<dyn std::error::Error + Send + Sync>>>(
        e: E,
        op: &'static str,
    ) -> Self {
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

#[cfg(test)]
mod tests {
    use super::HostError;
    use std::error::Error;

    #[derive(Debug)]
    struct Sentinel;

    impl std::fmt::Display for Sentinel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("sentinel")
        }
    }

    impl Error for Sentinel {}

    #[test]
    fn driver_error_preserves_operation_and_source() {
        let error = HostError::driver(Sentinel, "test operation");
        assert!(error.to_string().contains("test operation"));
        assert_eq!(error.source().unwrap().to_string(), "sentinel");
    }

    #[test]
    fn nul_error_preserves_source() {
        let nul = std::ffi::CString::new(b"bad\0value".to_vec()).unwrap_err();
        let error = HostError::NulError(nul, "test string");
        assert!(error.source().is_some());
    }
}
