use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

descend! {
    fn gpu_add_generic<N: nat, r: prv>(
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

    fn gpu_scale_generic<M: nat, r: prv>(
        a_array: &r uniq gpu.global [i32; M],
        factor: i32
    ) -[grid: gpu.grid<X<M/256>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref * factor
            }
        }
    }
}

#[test]
fn test_both_kernels_of_one_block_instantiate_separately() {
    let gpu_add = descend_fn!(gpu_add_generic, <N: 512>);
    let gpu_scale = descend_fn!(gpu_scale_generic, <M: 512>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut values = GpuBuffer::from_host(&Box::new([1i32; 512]), &stream).unwrap();
    let addend = GpuBuffer::from_host(&Box::new([2i32; 512]), &stream).unwrap();

    gpu_add(&stream, &mut values, &addend).unwrap();
    assert_eq!(*values.to_host(&stream).unwrap(), [3i32; 512]);

    gpu_scale(&stream, &mut values, 10).unwrap();
    assert_eq!(*values.to_host(&stream).unwrap(), [30i32; 512]);
}
