use std::fmt;

#[derive(Debug)]
pub enum HostError {
    Driver(Box<dyn std::error::Error + Send + Sync>, &'static str),
    NulError(std::ffi::NulError, &'static str),
    InvalidSize(&'static str),
    ContextMismatch(&'static str),
    LaunchAndSynchronize {
        launch: Box<HostError>,
        synchronize: Box<HostError>,
    },
}

impl HostError {
    pub(crate) fn driver<E: Into<Box<dyn std::error::Error + Send + Sync>>>(
        e: E,
        op: &'static str,
    ) -> Self {
        HostError::Driver(e.into(), op)
    }

    pub fn combine(launch: HostError, synchronize: HostError) -> Self {
        HostError::LaunchAndSynchronize {
            launch: Box::new(launch),
            synchronize: Box::new(synchronize),
        }
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::Driver(e, op) => write!(f, "Driver error during {}: {:?}", op, e),
            HostError::NulError(e, op) => write!(f, "NUL byte in string during {}: {}", op, e),
            HostError::InvalidSize(msg) => write!(f, "Invalid size: {}", msg),
            HostError::ContextMismatch(msg) => write!(f, "Context mismatch: {}", msg),
            HostError::LaunchAndSynchronize {
                launch,
                synchronize,
            } => {
                let reached_the_driver =
                    matches!(launch.as_ref(), HostError::Driver(_, "kernel launch"));
                let submission = if reached_the_driver {
                    "The launch reached the driver, which rejected it"
                } else {
                    "The launch was refused before the kernel reached the device"
                };
                write!(
                    f,
                    "The kernel launch failed ({launch}), and the synchronize that ran after it \
                     failed as well ({synchronize}). {submission}; the synchronize failure \
                     describes the whole stream, so it may report work other than this launch."
                )
            }
        }
    }
}

impl std::error::Error for HostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HostError::Driver(e, _) => Some(e.as_ref()),
            HostError::NulError(e, _) => Some(e),
            HostError::LaunchAndSynchronize { launch, .. } => Some(launch.as_ref()),
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

    fn interior_nul() -> std::ffi::NulError {
        std::ffi::CString::new(b"bad\0value".to_vec()).unwrap_err()
    }

    fn synchronize_failure() -> HostError {
        HostError::driver(Sentinel, "stream synchronize")
    }

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

    #[test]
    fn combine_keeps_both_halves_of_every_reachable_launch_failure() {
        let launch_failures = [
            (HostError::ContextMismatch("test mismatch"), "test mismatch"),
            (
                HostError::NulError(interior_nul(), "kernel function name"),
                "kernel function name",
            ),
            (
                HostError::driver(Sentinel, "load function"),
                "load function",
            ),
            (
                HostError::driver(Sentinel, "kernel launch"),
                "kernel launch",
            ),
        ];
        for (launch, expected_from_the_launch_half) in launch_failures {
            let combined = HostError::combine(launch, synchronize_failure());
            let rendered = combined.to_string();
            assert!(
                matches!(combined, HostError::LaunchAndSynchronize { .. }),
                "expected a combined error, got {rendered}"
            );
            assert!(
                rendered.contains(expected_from_the_launch_half),
                "the launch half is missing from {rendered}"
            );
            assert!(
                rendered.contains("stream synchronize"),
                "the synchronize half is missing from {rendered}"
            );
        }
    }

    #[test]
    fn combine_says_whether_the_launch_reached_the_driver() {
        let refused_before_the_device = HostError::combine(
            HostError::driver(Sentinel, "load function"),
            synchronize_failure(),
        );
        let rejected_by_the_driver = HostError::combine(
            HostError::driver(Sentinel, "kernel launch"),
            synchronize_failure(),
        );

        assert!(
            refused_before_the_device
                .to_string()
                .contains("refused before the kernel reached the device"),
            "{refused_before_the_device}"
        );
        assert!(
            rejected_by_the_driver
                .to_string()
                .contains("reached the driver, which rejected it"),
            "{rejected_by_the_driver}"
        );
    }

    #[test]
    fn a_combined_error_sources_the_launch_half() {
        let combined = HostError::combine(
            HostError::driver(Sentinel, "kernel launch"),
            HostError::NulError(interior_nul(), "stream synchronize"),
        );
        let rendered_source = combined.source().unwrap().to_string();
        assert!(
            rendered_source.contains("kernel launch"),
            "source must be the launch half, got {rendered_source}"
        );
        assert!(
            !rendered_source.contains("NUL byte"),
            "source must not be the synchronize half, got {rendered_source}"
        );
    }
}
