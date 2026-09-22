//! A generic kernel instantiated as a named function.
//!
//! `descend!` cannot compile `gpu_vec_add` where it is defined: `N` is baked
//! into the generated CUDA and only becomes a literal where someone names it.
//! `descend_fn!` names it once, and the kernel becomes an ordinary function item
//! — which, unlike the macro form, can be exported and called from another
//! module.

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

    descend_fn!(gpu_scale_generic as scale_1k, <N: 1024>);
}

// The instantiation is an item: it is used here through a path, from a module
// that does not contain the `descend!` block.
use kernels::{scale_1k, SCALE_1K_CONFIG};

#[test]
fn test_instantiated_kernel_is_callable_across_modules() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut a_gpu = GpuBuffer::from_host(&[20i32; 1024], &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&[22i32; 1024], &stream).unwrap();

    scale_1k(&stream, &mut a_gpu, &b_gpu).unwrap();

    assert_eq!(*a_gpu.to_host(&stream).unwrap(), [42i32; 1024]);
    assert_eq!(SCALE_1K_CONFIG.grid_dim, (4, 1, 1));
    assert_eq!(SCALE_1K_CONFIG.block_dim, (256, 1, 1));
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

descend_fn!(gpu_double as double_512, <N: 512>);
descend_fn!(gpu_double as double_1k, <N: 1024>);
// A third instantiation without an alias, which takes the kernel's own name.
descend_fn!(gpu_double, <N: 256>);

#[test]
fn test_two_instantiations_of_one_kernel_are_two_functions() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut small = GpuBuffer::from_host(&[1i32; 512], &stream).unwrap();
    let small_addend = GpuBuffer::from_host(&[2i32; 512], &stream).unwrap();
    let mut large = GpuBuffer::from_host(&Box::new([1i32; 1024]), &stream).unwrap();
    let large_addend = GpuBuffer::from_host(&Box::new([2i32; 1024]), &stream).unwrap();

    double_512(&stream, &mut small, &small_addend).unwrap();
    double_1k(&stream, &mut large, &large_addend).unwrap();

    assert_eq!(*small.to_host(&stream).unwrap(), [3i32; 512]);
    assert_eq!(*large.to_host(&stream).unwrap(), [3i32; 1024]);
    // Each instantiation fixed its own buffer size in its signature, and with it
    // the grid the kernel declares in terms of that size.
    assert_eq!(DOUBLE_512_CONFIG.grid_dim, (2, 1, 1));
    assert_eq!(DOUBLE_1K_CONFIG.grid_dim, (4, 1, 1));
}

#[test]
fn test_instantiation_without_an_alias_takes_the_kernel_name() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut values = GpuBuffer::from_host(&[1i32; 256], &stream).unwrap();
    let addend = GpuBuffer::from_host(&[2i32; 256], &stream).unwrap();

    gpu_double(&stream, &mut values, &addend).unwrap();

    assert_eq!(*values.to_host(&stream).unwrap(), [3i32; 256]);
    assert_eq!(GPU_DOUBLE_CONFIG.grid_dim, (1, 1, 1));
}
