use descend_host::{Context, GpuBuffer};
use descend_host::traits::DescendType;

// We strictly simulate an application that knows NOTHING about `cuda-core`.
// We define a custom data element that represents a valid Descend ABI type.
#[derive(Copy, Clone)]
#[repr(C)]
struct CustomVector {
    x: f32,
    y: f32,
    z: f32,
}

// We implement DescendType without requiring any backend supertraits!
unsafe impl DescendType for CustomVector {}

#[test]
fn test_consumer_compiles_and_runs() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let arr = [CustomVector { x: 1.0, y: 2.0, z: 3.0 }; 5];

    // GpuBuffer successfully encapsulates everything. 
    // cuda-core is completely hidden from this application code.
    let buf = GpuBuffer::from_host(&arr, &stream).unwrap();
    
    let readback = buf.to_host(&stream).unwrap();
    
    assert_eq!(readback[0].y, 2.0);
    assert_eq!(readback[4].z, 3.0);
}
