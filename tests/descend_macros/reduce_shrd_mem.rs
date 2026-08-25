use descend_host::*;
use descend_macros::{descend_kernel, descend_launch};

descend_kernel! {
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

#[test]
fn test_reduce_shrd_mem() {
    let ctx = descend_host::DescendContext::new(0);

    let a_box: Box<[i32; 128 * 512]> = vec![1i32; 128 * 512].into_boxed_slice().try_into().unwrap();
    let a_cpu = CpuShrd::new(a_box);
    let a_gpu = a_cpu.to_gpu(&ctx);

    let out_box: Box<[i32; 128]> = vec![0i32; 128].into_boxed_slice().try_into().unwrap();
    let out_cpu = CpuUniq::new(out_box);
    let mut out_gpu = out_cpu.to_gpu(&ctx);

    descend_launch!(gpu_reduce, <gs: 128, bs: 512>, &a_gpu, &mut out_gpu);

    let res = out_gpu.to_cpu();

    for val in res.iter() {
        assert_eq!(*val, 512);
    }
}
