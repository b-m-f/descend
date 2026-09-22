//! By-value kernel arguments on the device.
//!
//! The PTX below is hand-written because the crate embeds no compiled kernel:
//! each entry copies its by-value parameter into the buffer parameter, so a
//! readback shows what the device received. `sm_52` is forward-compatible with
//! the hardware these tests are recorded on.

use descend_host::{Context, GpuBuffer, Kernel, KernelLaunchConfig, KernelLauncher};

const PTX: &str = "
.version 6.4
.target sm_52
.address_size 64

.visible .entry buffer_then_val(
    .param .u64 buffer_then_val_param_0,
    .param .u32 buffer_then_val_param_1
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [buffer_then_val_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u32 %r1, [buffer_then_val_param_1];
    st.global.u32 [%rd2], %r1;
    ret;
}

.visible .entry val_then_buffer(
    .param .f32 val_then_buffer_param_0,
    .param .u64 val_then_buffer_param_1
)
{
    .reg .f32 %f<2>;
    .reg .b64 %rd<3>;

    ld.param.f32 %f1, [val_then_buffer_param_0];
    ld.param.u64 %rd1, [val_then_buffer_param_1];
    cvta.to.global.u64 %rd2, %rd1;
    st.global.f32 [%rd2], %f1;
    ret;
}

.visible .entry array_by_value(
    .param .u64 array_by_value_param_0,
    .param .align 4 .b8 array_by_value_param_1[12]
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [array_by_value_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u32 %r1, [array_by_value_param_1];
    ld.param.u32 %r2, [array_by_value_param_1+4];
    ld.param.u32 %r3, [array_by_value_param_1+8];
    st.global.u32 [%rd2], %r1;
    st.global.u32 [%rd2+4], %r2;
    st.global.u32 [%rd2+8], %r3;
    ret;
}
";

fn single_thread() -> KernelLaunchConfig {
    KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[test]
fn value_after_buffer_reaches_the_device() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let launcher = KernelLauncher::new(&stream, PTX).unwrap();

    let mut out = GpuBuffer::from_host(&[0u32; 1], &stream).unwrap();
    let value = 0xDEAD_BEEFu32;

    let mut kernel = Kernel::new("buffer_then_val", single_thread());
    kernel.push_arg_uniq(&mut out).unwrap();
    kernel.push_arg_val(&value).unwrap();
    // SAFETY: the entry above takes exactly this buffer and this by-value
    // parameter, in this order, writes only the buffer, and one thread is
    // enough for its single store.
    unsafe { launcher.launch(kernel) }.unwrap();

    assert_eq!(*out.to_host(&stream).unwrap(), [value]);
}

#[test]
fn value_before_buffer_keeps_positional_order() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let launcher = KernelLauncher::new(&stream, PTX).unwrap();

    let mut out = GpuBuffer::from_host(&[0.0f32; 1], &stream).unwrap();
    let value = 1.5f32;

    let mut kernel = Kernel::new("val_then_buffer", single_thread());
    kernel.push_arg_val(&value).unwrap();
    kernel.push_arg_uniq(&mut out).unwrap();
    // SAFETY: as above, with the by-value parameter first — the order the
    // pushes use.
    unsafe { launcher.launch(kernel) }.unwrap();

    assert_eq!(*out.to_host(&stream).unwrap(), [value]);
}

#[test]
fn array_is_marshaled_by_value() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let launcher = KernelLauncher::new(&stream, PTX).unwrap();

    let mut out = GpuBuffer::from_host(&[0i32; 3], &stream).unwrap();
    let values = [11i32, 22, 33];

    let mut kernel = Kernel::new("array_by_value", single_thread());
    kernel.push_arg_uniq(&mut out).unwrap();
    kernel.push_arg_val(&values).unwrap();
    // SAFETY: the entry takes a 12-byte by-value parameter with 4-byte
    // alignment, which is the layout of `[i32; 3]`, and writes only the buffer.
    unsafe { launcher.launch(kernel) }.unwrap();

    assert_eq!(*out.to_host(&stream).unwrap(), values);
}
