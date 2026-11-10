use crate::limits::MAX_SHARED_MEMORY_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMemoryBytes(u32);

impl SharedMemoryBytes {
    pub fn new(requested: u64) -> Option<SharedMemoryBytes> {
        SharedMemoryBytes::with_ceiling(requested, MAX_SHARED_MEMORY_BYTES)
    }

    pub fn with_ceiling(requested: u64, ceiling: u32) -> Option<SharedMemoryBytes> {
        let bytes = u32::try_from(requested).ok()?;
        (bytes <= ceiling).then_some(SharedMemoryBytes(bytes))
    }

    pub fn value(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_ceiling_is_the_portable_limit() {
        let at_the_limit = u64::from(MAX_SHARED_MEMORY_BYTES);
        assert_eq!(
            SharedMemoryBytes::new(at_the_limit),
            Some(SharedMemoryBytes(MAX_SHARED_MEMORY_BYTES))
        );
        assert_eq!(SharedMemoryBytes::new(at_the_limit + 1), None);
    }

    #[test]
    fn a_count_the_driver_cannot_take_is_refused_not_truncated() {
        assert_eq!(SharedMemoryBytes::new(u64::from(u32::MAX) + 1), None);
        assert_eq!(SharedMemoryBytes::new(u64::MAX), None);
    }

    #[test]
    fn an_explicit_ceiling_is_honoured() {
        let opted_in: u32 = 100 * 1024;
        assert_eq!(
            SharedMemoryBytes::with_ceiling(u64::from(opted_in), opted_in)
                .map(SharedMemoryBytes::value),
            Some(opted_in)
        );
        assert_eq!(
            SharedMemoryBytes::with_ceiling(u64::from(opted_in) + 1, opted_in),
            None
        );
    }

    #[test]
    fn a_kernel_with_no_shared_parameters_counts_zero() {
        assert_eq!(
            SharedMemoryBytes::new(0).map(SharedMemoryBytes::value),
            Some(0)
        );
    }
}
