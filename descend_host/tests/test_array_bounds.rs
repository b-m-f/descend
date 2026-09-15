use descend_host::{GpuBuffer, traits::DescendType};
use descend_host::error::HostError;

#[derive(Clone, Copy)]
#[repr(C)]
struct OverflowType(i32, i32, i32, i32); // 16 bytes

unsafe impl DescendType for OverflowType {}

#[test]
fn test_multiplication_overflow() {
    // 16 * (usize::MAX / 8) will overflow usize
    const HUGE_N: usize = usize::MAX / 8;
    
    let res = GpuBuffer::<OverflowType, HUGE_N>::validate_buffer_params();
    match res {
        Err(HostError::InvalidSize(msg)) => {
            assert!(msg.contains("overflow"), "Expected overflow error, got: {}", msg);
        }
        _ => panic!("Expected multiplication overflow error"),
    }
}

#[test]
fn test_wrong_conversion_length() {
    // Pass a vector of length 3 to validate_readback where N=4 is expected
    let bad_vec = vec![OverflowType(1, 1, 1, 1); 3];
    
    let res = GpuBuffer::<OverflowType, 4>::validate_readback(bad_vec);
    match res {
        Err(HostError::ConversionError(msg)) => {
            assert!(msg.contains("Size mismatch"), "Expected size mismatch error, got: {}", msg);
        }
        _ => panic!("Expected size mismatch error"),
    }
}
