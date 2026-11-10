//! Kernel parameters passed by value.
//!
//! Descend has always allowed a kernel to take a parameter by value
//! (`alpha: i32` lowers to `const descend::i32 alpha`), but the macro used to
//! treat every parameter as a buffer. Such a parameter is now an ordinary Rust
//! function argument, pushed with `Kernel::push_arg_val`.

use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_add_scalar<r: prv>(
        a_array: &r uniq gpu.global [i32; 256],
        alpha: i32
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + alpha
            }
        }
    }
}

#[test]
fn test_value_argument_reaches_the_kernel() {
    let gpu_add_scalar = descend_fn!(gpu_add_scalar);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut a_gpu = GpuBuffer::from_host(&[20i32; 256], &stream).unwrap();

    gpu_add_scalar(&stream, &mut a_gpu, 22).unwrap();

    assert_eq!(*a_gpu.to_host(&stream).unwrap(), [42i32; 256]);
}

// A value between two buffers: the pushes must keep the kernel's own order.
descend! {
    fn gpu_scale_between<r1: prv, r2: prv>(
        a_array: &r1 uniq gpu.global [i32; 256],
        factor: i32,
        b_array: &r2 shrd gpu.global [i32; 256]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref * factor + (*b_array).to_view.grp::<256>[[block]][[thread]]
            }
        }
    }
}

#[test]
fn test_value_argument_between_buffers_keeps_its_position() {
    let gpu_scale_between = descend_fn!(gpu_scale_between);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut a_gpu = GpuBuffer::from_host(&[5i32; 256], &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&[2i32; 256], &stream).unwrap();

    gpu_scale_between(&stream, &mut a_gpu, 8, &b_gpu).unwrap();

    assert_eq!(*a_gpu.to_host(&stream).unwrap(), [42i32; 256]);
}
