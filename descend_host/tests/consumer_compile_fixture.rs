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
pub fn process_buffer<T: DescendType>(buf: &mut GpuBuffer<T>, _stream: &Stream) {
    // We don't execute anything; this is a compile fixture.
    let _ = buf.context();
}

unsafe fn borrowed_kernel_arguments_contract(
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
    kernel.push_arg_uniq(&mut output)?;
    kernel.push_arg_shrd(&input)?;
    kernel.push_arg_shrd(&input)?;
    // SAFETY: the caller supplies a launcher whose `kernel` entry has this
    // signature. The buffers and launcher remain alive until the same-stream
    // readback below establishes completion.
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
}
