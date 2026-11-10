use descend_host::{Context, GpuBuffer};

#[test]
fn test_cross_thread_binding() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let arr = [42i32; 10];
    let buf = GpuBuffer::from_host(&arr, &stream).unwrap();

    let handle = std::thread::spawn(move || {
        let result = buf.to_host(&stream).unwrap();
        assert_eq!(result[0], 42);
        assert_eq!(result[9], 42);
    });

    handle.join().unwrap();
}
