use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_fn};

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
    let matmul_64x128x64 =
        descend_fn!(gpu_matmul<<<(4, 4), (16, 16)>>>, <bs: 16, m: 64, n: 128, k: 64>);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let a_array = [20i32; 64 * 128]; // m * n
    let b_array = [10i32; 128 * 64]; // n * k
    let c_array = [0i32; 64 * 64];   // m * k

    let a_gpu = GpuBuffer::from_host(&Box::new(a_array), &stream).unwrap();
    let b_gpu = GpuBuffer::from_host(&Box::new(b_array), &stream).unwrap();
    let mut c_gpu = GpuBuffer::from_host(&Box::new(c_array), &stream).unwrap();

    matmul_64x128x64(&stream, &a_gpu, &b_gpu, &mut c_gpu).unwrap();

    let c_cpu_back = c_gpu.to_host(&stream).unwrap();
    assert_eq!(*c_cpu_back, [20i32 * 10i32 * 128i32; 64 * 64]);
}
