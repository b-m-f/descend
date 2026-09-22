//! The macro-generated launch needs no `unsafe` at the call site.
//!
//! `KernelLauncher::launch` is unsafe because a caller pushing arguments by hand
//! cannot be held to the kernel's signature. The generated function is not such
//! a caller: its PTX, entry name, argument types, argument order and launch
//! configuration all come from one Descend definition, so it discharges those
//! obligations itself and hides the `unsafe` block in its own body.
//!
//! `deny(unsafe_code)` here is what pins that: the test fails to compile if a
//! future change moves an `unsafe` block back out to the call site.
#![deny(unsafe_code)]

use descend_host::{Context, GpuBuffer};
use descend_macros::descend;

descend! {
    fn gpu_scale<r: prv>(
        a_array: &r uniq gpu.global [i32; 256],
        b_array: &r shrd gpu.global [i32; 256]
    ) -[grid: gpu.grid<X<1>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
            }
        }
    }
}

#[test]
fn generated_function_launches_without_unsafe() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut a_gpu = GpuBuffer::from_host(&[20i32; 256], &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&[10i32; 256], &stream).unwrap();

    gpu_scale(&stream, &mut a_gpu, &b_gpu).unwrap();

    assert_eq!(*a_gpu.to_host(&stream).unwrap(), [30i32; 256]);
}

#[test]
fn generated_config_exposes_the_derived_dimensions() {
    // The configuration the macro derived from the kernel's own declaration,
    // `-[grid: gpu.grid<X<1>, X<256>>]->`, and from its lack of shared-memory
    // parameters.
    assert_eq!(GPU_SCALE_CONFIG.grid_dim, (1, 1, 1));
    assert_eq!(GPU_SCALE_CONFIG.block_dim, (256, 1, 1));
    assert_eq!(GPU_SCALE_CONFIG.shared_mem_bytes, 0);
}
