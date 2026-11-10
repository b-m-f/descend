//! A generic kernel instantiated at the point of use.
//!
//! `descend!` cannot compile `gpu_scale_generic` where it is defined: `N` is baked
//! into the generated CUDA and only becomes a literal where someone instantiates it.
//! `descend_fn!` is an expression that evaluates to the instantiated function, so the
//! caller names it with an ordinary `let` — there is no alias to invent, and a second
//! set of sizes is simply a second binding.
//!
//! A procedural macro cannot tell item position from expression position, so the
//! instantiation is always local. Reaching one from another module means wrapping it
//! in an ordinary function, as `kernels::scale_1k` does below.

use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

mod kernels {
    use descend_macros::{descend, descend_fn};

    descend! {
        fn gpu_scale_generic<N: nat, r: prv>(
            a_array: &r uniq gpu.global [i32; N],
            b_array: &r shrd gpu.global [i32; N]
        ) -[grid: gpu.grid<X<4>, X<256>>]-> () {
            sched block in grid {
                sched thread in block {
                    let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                    *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
                }
            }
        }
    }

    /// Exporting an instantiation takes a wrapper, because `descend_fn!` yields a
    /// value rather than an item. The PTX is still compiled once, at expansion time,
    /// and baked into this function.
    pub fn scale_1k(
        stream: &::descend_host::Stream,
        a_array: &mut ::descend_host::GpuBuffer<[i32; 1024]>,
        b_array: &::descend_host::GpuBuffer<[i32; 1024]>,
    ) -> ::core::result::Result<(), ::descend_host::error::HostError> {
        let scale = descend_fn!(gpu_scale_generic, <N: 1024>);
        scale(stream, a_array, b_array)
    }
}

// Used here through a path, from a module that does not contain the `descend!` block.
use kernels::scale_1k;

#[test]
fn test_instantiated_kernel_is_callable_across_modules() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut a_gpu = GpuBuffer::from_host(&[20i32; 1024], &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&[22i32; 1024], &stream).unwrap();

    scale_1k(&stream, &mut a_gpu, &b_gpu).unwrap();

    assert_eq!(*a_gpu.to_host(&stream).unwrap(), [42i32; 1024]);
}

// Two instantiations of one kernel are two functions, each carrying its own PTX
// for its own size.
descend! {
    fn gpu_double<N: nat, r: prv>(
        a_array: &r uniq gpu.global [i32; N],
        b_array: &r shrd gpu.global [i32; N]
    ) -[grid: gpu.grid<X<N/256>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
            }
        }
    }
}

#[test]
fn test_two_instantiations_of_one_kernel_are_two_functions() {
    // Two bindings of the same kernel, in one scope. Each instantiation lives in its
    // own block, so both keep the kernel's name internally without colliding.
    let double_512 = descend_fn!(gpu_double, <N: 512>);
    let double_1k = descend_fn!(gpu_double, <N: 1024>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut small = GpuBuffer::from_host(&[1i32; 512], &stream).unwrap();
    let small_addend = GpuBuffer::from_host(&[2i32; 512], &stream).unwrap();
    let mut large = GpuBuffer::from_host(&Box::new([1i32; 1024]), &stream).unwrap();
    let large_addend = GpuBuffer::from_host(&Box::new([2i32; 1024]), &stream).unwrap();

    double_512(&stream, &mut small, &small_addend).unwrap();
    double_1k(&stream, &mut large, &large_addend).unwrap();

    // Each instantiation fixed its own buffer size in its signature, and with it the
    // grid the kernel declares as `X<N/256>`: 2 blocks for 512 elements, 4 for 1024.
    // A wrong grid would leave part of the buffer untouched, so these results are
    // what makes the specialised launch configuration observable.
    assert_eq!(*small.to_host(&stream).unwrap(), [3i32; 512]);
    assert_eq!(*large.to_host(&stream).unwrap(), [3i32; 1024]);
}

#[test]
fn test_the_binding_names_the_instantiation() {
    // Nothing ties the binding to the kernel's name; this one reads as its size.
    let double_256 = descend_fn!(gpu_double, <N: 256>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut values = GpuBuffer::from_host(&[1i32; 256], &stream).unwrap();
    let addend = GpuBuffer::from_host(&[2i32; 256], &stream).unwrap();

    double_256(&stream, &mut values, &addend).unwrap();

    assert_eq!(*values.to_host(&stream).unwrap(), [3i32; 256]);
}
