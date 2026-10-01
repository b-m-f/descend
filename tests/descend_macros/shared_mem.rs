use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

// Two kernels that exchange data through `gpu.shared` parameters. Until the
// launch configuration carried a real byte count, the macro rejected these
// kernels outright, so this file is the evidence that they can be launched at
// all.
//
// Note what these tests do *not* establish: the hardware does not bounds-check
// shared memory, so a launch that requests too few bytes still reads and writes
// within the block's shared window and can produce correct results anyway. The
// byte count itself is pinned at compile time — the macro cross-checks the
// count the Descend compiler emitted against the one the parameter types give,
// and refuses to generate a launch when they disagree.
descend! {
    fn gpu_reduce<gs: nat, bs: nat, r1: prv, r2: prv, r3: prv>(
        vec: &r1 shrd gpu.global [i32; gs*bs],
        out_vec: &r2 uniq gpu.global [i32; gs],
        [grid.forall(X)] tmp: &r3 uniq gpu.shared [i32; bs]
    ) -[grid: gpu.grid<X<gs>, X<bs>>]-> () {
        let block_group = &shrd (*vec).to_view.grp::<bs>;
        let view_out = &uniq (*out_vec).to_view.grp::<1>;
        sched block in grid {
            let ib = &shrd (*block_group)[[block]];
            let out_elem = &uniq (*view_out)[[block]];
            let tmp_view = &uniq (*tmp).to_view;
            sched thread in block {
                let inp = &shrd (*ib)[[thread]];
                let tmp_in = &uniq (*tmp_view)[[thread]];
                *tmp_in = *inp
            };

            for k in halved_range(bs/2) {
                sync(block);
                let split_tmp_view = &uniq (*tmp_view)[0..2*k];
                let active_half0 = &uniq (*split_tmp_view)[0..k];
                let active_half1 = &uniq (*split_tmp_view)[k..2*k];
                indep(X) k block {
                    active =>
                        sched thread in active {
                            let a = &uniq (*active_half0)[[thread]];
                            let b = &shrd (*active_half1)[[thread]];
                            *a = *a + *b
                        },
                    inactive => { () }
                }
            };

            sync(block);
            indep(X) 1 block {
                single_thread =>
                    sched thread in single_thread {
                        let out = &uniq (*out_elem)[[thread]];
                        let tmp_val = &shrd (*tmp_view)[0..1][[thread]];
                        *out = *tmp_val
                    },
                rest => { () }
            }
        }
    }
}

// Two shared arrays of different element sizes, which the code generator packs
// into one dynamic buffer: 64 * 8 bytes for the `f64` tile plus 64 * 4 for the
// `i32` tile. Reaching the GPU at all means the two byte counts agreed.
descend! {
    fn gpu_two_tiles<bs: nat, r1: prv, r2: prv, r3: prv, r4: prv>(
        values: &r1 uniq gpu.global [i32; bs],
        scales: &r2 uniq gpu.global [f64; bs],
        [grid.forall(X)] tmp_i: &r3 uniq gpu.shared [i32; bs],
        [grid.forall(X)] tmp_f: &r4 uniq gpu.shared [f64; bs]
    ) -[grid: gpu.grid<X<1>, X<bs>>]-> () {
        let value_group = &uniq (*values).to_view.grp::<bs>;
        let scale_group = &uniq (*scales).to_view.grp::<bs>;
        sched block in grid {
            let ib = &uniq (*value_group)[[block]];
            let sb = &uniq (*scale_group)[[block]];
            let tmp_i_view = &uniq (*tmp_i).to_view;
            let tmp_f_view = &uniq (*tmp_f).to_view;
            sched thread in block {
                let v = &uniq (*ib)[[thread]];
                let s = &uniq (*sb)[[thread]];
                let ti = &uniq (*tmp_i_view)[[thread]];
                let tf = &uniq (*tmp_f_view)[[thread]];
                *ti = *v + 1;
                *tf = *s + 1.0;
                *v = *ti;
                *s = *tf
            }
        }
    }
}

const GRID: usize = 16;
const BLOCK: usize = 64;

#[test]
fn test_block_reduction_through_shared_memory() {
    let reduce_16x64 = descend_fn!(gpu_reduce, <gs: 16, bs: 64>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let mut input = [0i32; GRID * BLOCK];
    for (i, value) in input.iter_mut().enumerate() {
        *value = (i % 7) as i32;
    }
    let output = [0i32; GRID];

    let input_gpu = GpuBuffer::from_host(&Box::new(input), &stream).unwrap();
    let mut output_gpu = GpuBuffer::from_host(&Box::new(output), &stream).unwrap();

    reduce_16x64(&stream, &input_gpu, &mut output_gpu).unwrap();

    let mut expected = [0i32; GRID];
    for (block, sum) in expected.iter_mut().enumerate() {
        *sum = input[block * BLOCK..(block + 1) * BLOCK].iter().sum();
    }
    assert_eq!(*output_gpu.to_host(&stream).unwrap(), expected);
}

#[test]
fn test_two_shared_tiles_of_different_element_sizes() {
    let two_tiles_64 = descend_fn!(gpu_two_tiles, <bs: 64>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let values = [7i32; BLOCK];
    let scales = [0.5f64; BLOCK];

    let mut values_gpu = GpuBuffer::from_host(&values, &stream).unwrap();
    let mut scales_gpu = GpuBuffer::from_host(&scales, &stream).unwrap();

    two_tiles_64(&stream, &mut values_gpu, &mut scales_gpu).unwrap();

    assert_eq!(*values_gpu.to_host(&stream).unwrap(), [8i32; BLOCK]);
    assert_eq!(*scales_gpu.to_host(&stream).unwrap(), [1.5f64; BLOCK]);
}

// A kernel whose sizes are fixed where it is defined becomes a function item, and
// with it a constant holding the launch configuration the macro derived. That makes
// the byte count itself assertable -- which an instantiation through `descend_fn!`
// cannot be, because its constant lives inside the expression's block.
descend! {
    fn gpu_shared_copy<r1: prv, r2: prv>(
        vec: &r1 uniq gpu.global [i32; 64],
        [grid.forall(X)] tmp: &r2 uniq gpu.shared [i32; 64]
    ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
        let view = &uniq (*vec).to_view.grp::<64>;
        sched block in grid {
            let ib = &uniq (*view)[[block]];
            let tmp_view = &uniq (*tmp).to_view;
            sched thread in block {
                let v = &uniq (*ib)[[thread]];
                let t = &uniq (*tmp_view)[[thread]];
                *t = *v + 1;
                *v = *t
            }
        }
    }
}

#[test]
fn test_shared_bytes_reach_the_launch_configuration() {
    // 64 elements of `i32`, and nothing else in shared memory.
    assert_eq!(GPU_SHARED_COPY_CONFIG.shared_mem_bytes, 256);
    assert_eq!(GPU_SHARED_COPY_CONFIG.block_dim, (64, 1, 1));

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut vec_gpu = GpuBuffer::from_host(&[41i32; 64], &stream).unwrap();

    gpu_shared_copy(&stream, &mut vec_gpu).unwrap();

    assert_eq!(*vec_gpu.to_host(&stream).unwrap(), [42i32; 64]);
}
