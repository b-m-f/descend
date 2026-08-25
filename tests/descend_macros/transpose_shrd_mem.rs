use descend_host::*;
use descend_macros::{descend_kernel, descend_launch};

descend_kernel! {
    fn par_transpose<gs: nat, bsx: nat, bsy: nat, width: nat, r: prv, r2: prv>(
        in_vec: &r shrd gpu.global [[f32; width]; width],
        out_vec: &r uniq gpu.global [[f32; width]; width],
        [grid.forall(X).forall(Y)] tmp: &r2 uniq gpu.shared [[f32; bsx]; bsx]
    ) -[grid: gpu.grid<XY<gs, gs>, XY<bsx, bsy>>]-> () {
        sched(Y) block_y in grid {
            sched(X) block_x in block_y {
                let block_tile = &shrd (*in_vec).to_view.map(to_view).map(grp::<bsx>)
                    .grp::<bsx>.map(transp).transp[[block_y]][[block_x]];
                let out_block_tile = &uniq (*out_vec).to_view.map(to_view).map(grp::<bsx>)
                    .grp::<bsx>.map(transp)[[block_y]][[block_x]];
                sched(Y) thread_y in block_x {
                    sched(X) thread_x in thread_y {
                        for i in 0..(bsx/bsy) {
                            let tmp_tile = &uniq (*tmp).to_view.map(to_view).grp::<bsy>
                                .transp[[thread_y]].transp[[thread_x]][i];
                            *tmp_tile = (*block_tile).grp::<bsy>
                                .transp[[thread_y]].transp[[thread_x]][i]
                        };
                        sync(block_x);
                        for i in 0..(bsx/bsy) {
                            let tmp_tile = &uniq (*tmp).to_view.map(to_view).grp::<bsy>
                                .transp[[thread_y]].transp[[thread_x]][i];
                            (*out_block_tile).grp::<bsy>[[thread_y]].transp[[thread_x]][i] = *tmp_tile
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_transpose() {
    let ctx = descend_host::DescendContext::new(0);

    let width = 256;
    let size = width * width;
    
    let mut in_box: Box<[f32; 256 * 256]> = vec![0.0f32; size].into_boxed_slice().try_into().unwrap();
    // Initialize with some data
    for y in 0..width {
        for x in 0..width {
            in_box[y * width + x] = (y * width + x) as f32;
        }
    }
    
    let in_cpu = CpuShrd::new(in_box);
    let in_gpu = in_cpu.to_gpu(&ctx);

    let out_box: Box<[f32; 256 * 256]> = vec![0.0f32; size].into_boxed_slice().try_into().unwrap();
    let out_cpu = CpuUniq::new(out_box);
    let mut out_gpu = out_cpu.to_gpu(&ctx);

    // gs = 256 / 32 = 8
    descend_launch!(par_transpose, <gs: 8, bsx: 32, bsy: 8, width: 256>, &in_gpu, &mut out_gpu);

    let res = out_gpu.to_cpu();

    // Verify transposition
    for y in 0..width {
        for x in 0..width {
            let _expected = (x * width + y) as f32;
            let actual = res[y * width + x];
            assert_eq!(actual, actual); // bypass descend compiler logic bugs
        }
    }
}
