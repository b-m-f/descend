use descend_host::stream::Stream;
use descend_host::traits::DescendType;
use descend_host::GpuBuffer;

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
pub fn process_buffer<T: DescendType, const N: usize>(buf: &mut GpuBuffer<T, N>, _stream: &Stream) {
    // We don't execute anything; this is a compile fixture.
    let _ = buf.context();
}

fn in_place_readback_contract<'a>(
    host: &'a mut [i32; 4],
    stream: &Stream,
) -> Result<&'a mut [i32; 4], descend_host::error::HostError> {
    let buffer = GpuBuffer::from_host_mut(host, stream)?;
    buffer.read_back(stream)
}

fn borrowed_kernel_arguments_contract(
    stream: &Stream,
    launcher: &descend_host::KernelLauncher,
) -> Result<Box<[i32; 4]>, descend_host::error::HostError> {
    let input = GpuBuffer::from_host(&[1i32; 4], stream)?;
    let mut output = GpuBuffer::from_host(&[0i32; 4], stream)?;
    let config = descend_host::KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut kernel = descend_host::Kernel::new("kernel", config);
    kernel.push_arg_mut(&mut output)?;
    kernel.push_arg(&input)?;
    kernel.push_arg(&input)?;
    // SAFETY: never called; this function only checks that the borrows end at launch.
    unsafe { launcher.launch(kernel)? };
    output.to_host(stream)
}

#[test]
fn test_consumer_compiles_without_backend_traits() {
    let _ = borrowed_kernel_arguments_contract;
    // This is purely a compile-time check.
    // If it compiles, we know we successfully severed the backend dependency.
    let _dummy = CustomVector {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let _ = in_place_readback_contract;
}
