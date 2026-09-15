use descend_host::GpuBuffer;
use descend_host::traits::DescendType;
use descend_host::stream::Stream;

// We strictly simulate an application that knows NOTHING about `cuda-core`.
// We define a custom data element that represents a valid Descend ABI type.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CustomVector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// The corresponding Descend struct declaration to prove isomorphism:
// struct CustomVector { x: float, y: float, z: float }

// We implement DescendType without requiring any backend supertraits!
unsafe impl DescendType for CustomVector {}

// A generic helper to prove `GpuBuffer` bounds don't leak backend traits.
// This function compiles strictly against `DescendType`.
pub fn process_buffer<T: DescendType, const N: usize>(buf: &mut GpuBuffer<T, N>, stream: &Stream) {
    // We don't execute anything; this is a compile fixture.
    let _ = buf.context();
}

#[test]
fn test_consumer_compiles_without_backend_traits() {
    // This is purely a compile-time check.
    // If it compiles, we know we successfully severed the backend dependency.
    let _dummy = CustomVector { x: 0.0, y: 0.0, z: 0.0 };
}
