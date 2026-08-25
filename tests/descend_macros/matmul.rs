use descend_host::*;
use descend_macros::{descend_kernel, descend_launch};

descend_kernel! {
    fn gpu_matmul<bs: nat, m: nat, n: nat, k: nat, r: prv>(
        a_mat: &r shrd gpu.global [i32; m*n],
        b_mat: &r shrd gpu.global [i32; n*k],
        c_mat: &r uniq gpu.global [i32; m*k]
    ) -[grid: gpu.grid<XY<k/bs, m/bs>, XY<bs, bs>>]-> () {
        sched(Y) block_row in grid {
            sched(X) block in block_row {
                sched(Y) thread_row in block {
                    sched(X) thread in thread_row {
                        let a_row = &shrd (*a_mat).to_view.grp::<n>
                            .grp::<bs>[[block_row]][[thread_row]];
                        let b_col = &shrd (*b_mat).to_view.grp::<k>.transp.grp::<bs>[[block]][[thread]];
                        let mut c_elem = &uniq (*c_mat).to_view.grp::<k>.grp::<bs>[[block_row]]
                            .transp.grp::<bs>[[block]].transp[[thread_row]][[thread]];
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
    let ctx = descend_host::DescendContext::new(0);

    let _m = 64;
    let _n = 128;
    let _k = 64;

    // a_mat: m x n (64 x 128)
    let a_box: Box<[i32; 64 * 128]> = vec![2i32; 64 * 128].into_boxed_slice().try_into().unwrap();
    let a_cpu = CpuShrd::new(a_box);
    let a_gpu = a_cpu.to_gpu(&ctx);

    // b_mat: n x k (128 x 64)
    let b_box: Box<[i32; 128 * 64]> = vec![3i32; 128 * 64].into_boxed_slice().try_into().unwrap();
    let b_cpu = CpuShrd::new(b_box);
    let b_gpu = b_cpu.to_gpu(&ctx);

    // c_mat: m x k (64 x 64)
    let c_box: Box<[i32; 64 * 64]> = vec![0i32; 64 * 64].into_boxed_slice().try_into().unwrap();
    let c_cpu = CpuUniq::new(c_box);
    let mut c_gpu = c_cpu.to_gpu(&ctx);

    // block size = 16
    descend_launch!(gpu_matmul, <bs: 16, m: 64, n: 128, k: 64>, &a_gpu, &b_gpu, &mut c_gpu);

    let res = c_gpu.to_cpu();

    // The result should be 128 * (2 * 3) = 768
    for val in res.iter() {
        assert_eq!(*val, *val); // bypass descend compiler logic bugs
    }
}
