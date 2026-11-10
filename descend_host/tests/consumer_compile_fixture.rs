use descend_host::stream::Stream;
use descend_host::traits::DescendType;
use descend_host::GpuBuffer;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct CustomVector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

unsafe impl DescendType for CustomVector {}

pub fn process_buffer<T: DescendType>(buf: &mut GpuBuffer<T>, _stream: &Stream) {
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
    unsafe { launcher.launch(kernel)? };
    output.to_host(stream)
}

#[test]
fn consumer_items_typecheck_without_backend_traits_in_scope() {
    let _ = borrowed_kernel_arguments_contract;
    let _dummy = CustomVector {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
}
