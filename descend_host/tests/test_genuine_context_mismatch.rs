use descend_host::{Context, GpuBuffer};

// TODO: update readme that not all machines can run all tests
#[test]
fn test_genuine_mismatch() {
    // Attempt to create context on device 0 and device 1.
    // If device 1 fails (e.g. only 1 GPU in system), we successfully exit.
    let ctx0 = Context::new(0).unwrap();

    let ctx1 = match Context::new(1) {
        Ok(c) => c,
        Err(_) => {
            println!("Skipping genuine context mismatch test: only 1 GPU available.");
            return;
        }
    };

    let stream0 = ctx0.new_stream().unwrap();
    let stream1 = ctx1.new_stream().unwrap();

    let arr = [1i32; 10];
    let buf = GpuBuffer::from_host(&arr, &stream0).unwrap();

    // They are on different devices, so to_host should definitively fail!
    let res = buf.to_host(&stream1);
    assert!(
        res.is_err(),
        "Expected mismatch error when mixing device 0 buffer with device 1 stream"
    );
}
