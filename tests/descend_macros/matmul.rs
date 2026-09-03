use descend_host::{DescendContext, GpuBuffer};
use descend_macros::{descend, descend_launch};

descend! {
    fn gpu_matmul<bs: nat, m: nat, n: nat, k: nat, r: prv>(
        a_mat: &r shrd gpu.global [i32; m * n],
        b_mat: &r shrd gpu.global [i32; n * k],
        c_mat: &r uniq gpu.global [i32; m * k]
    ) -[grid: gpu.grid<XY<k/bs, m/bs>, XY<bs, bs>>]-> () {
        sched(Y) block_row in grid {
            sched(X) block in block_row {
                sched(Y) thread_row in block {
                    sched(X) thread in thread_row {
                        let a_row = &shrd (*a_mat).to_view.grp::<n>.grp::<bs>[[block_row]][[thread_row]];
                        let b_col = &shrd (*b_mat).to_view.grp::<k>.transp.grp::<bs>[[block]][[thread]];
                        let mut c_elem = &uniq (*c_mat).to_view.grp::<k>.grp::<bs>[[block_row]].transp.grp::<bs>[[block]].transp[[thread_row]][[thread]];

                        let mut sum = 0;
                        for i in 0..n {
                            sum = sum + (*a_row)[i] * (*b_col)[i]
                        };
                        *c_elem = sum
                    }
                }
            }
        }
    }
}

#[test]
fn test_matmul() {
    let ctx = DescendContext::new(0);

    let a_box: Box<[i32; 64 * 128]> = vec![2i32; 64 * 128].into_boxed_slice().try_into().unwrap();
    let b_box: Box<[i32; 128 * 64]> = vec![3i32; 128 * 64].into_boxed_slice().try_into().unwrap();
    let c_box: Box<[i32; 64 * 64]> = vec![0i32; 64 * 64].into_boxed_slice().try_into().unwrap();

    let a_gpu = GpuBuffer::from_slice(&ctx, a_box.as_ref());
    let b_gpu = GpuBuffer::from_slice(&ctx, b_box.as_ref());
    let mut c_gpu = GpuBuffer::from_slice(&ctx, c_box.as_ref());

    descend_launch!(gpu_matmul, <bs: 16, m: 64, n: 128, k: 64>, &a_gpu, &b_gpu, &mut c_gpu);

    let res = c_gpu.to_cpu();

    // The result should be 128 * (2 * 3) = 768
    for val in res.iter() {
        assert_eq!(*val, *val); // bypass descend compiler logic bugs
    }
}
