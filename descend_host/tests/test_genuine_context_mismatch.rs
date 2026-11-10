use descend_host::error::HostError;
use descend_host::{Context, GpuBuffer};

#[test]
#[ignore = "requires a second CUDA device"]
fn test_genuine_mismatch() {
    let ctx0 = Context::new(0).unwrap();

    let ctx1 = match Context::new(1) {
        Ok(c) => c,
        Err(HostError::Driver(source, _)) => {
            let error_str = source.to_string();
            if error_str.contains("invalid device ordinal") {
                println!("Skipping genuine context mismatch test: device 1 not available.");
                return;
            } else {
                panic!(
                    "Failed to create device 1 context for unexpected reason: {}",
                    error_str
                );
            }
        }
        Err(e) => panic!("Unexpected error type: {:?}", e),
    };

    let stream0 = ctx0.new_stream().unwrap();
    let stream1 = ctx1.new_stream().unwrap();

    let arr = [1i32; 10];
    let buf = GpuBuffer::from_host(&arr, &stream0).unwrap();

    let res = buf.to_host(&stream1);
    match res {
        Err(HostError::ContextMismatch(_)) => {}
        Err(e) => panic!("Expected ContextMismatch, got: {:?}", e),
        Ok(_) => panic!("Expected mismatch error, but it succeeded!"),
    }
}
