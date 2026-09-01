use descend_host::*;
use descend_macros::{descend_kernel, descend_launch};

descend_kernel! {
    fn gpu_par_scan<n: nat, gs: nat, bs: nat, r1: prv, r2: prv, r3: prv, r4: prv>(
        input: &r1 shrd gpu.global [i32; n],
        output: &r2 uniq gpu.global [i32; n],
        block_sums: &r3 uniq gpu.global [i32; gs],
        [grid.forall(X)] tmp: &r4 uniq gpu.shared [i32; 1024]
    ) -[grid: gpu.grid<X<gs>, X<bs>>]-> () {
        let block_group = &shrd (*input).to_view.grp::<(1024)>;
        let out_grp = &uniq (*output).to_view.grp::<(1024)>;
        let block_sums_grp = &uniq (*block_sums).to_view.grp::<1>;

        sched block in grid {
            let ib = &shrd (*block_group)[[block]];
            let block_out = &uniq (*out_grp)[[block]];
            let bsum = &uniq (*block_sums_grp)[[block]];
            {
                let split_tmp_view = &uniq (*tmp).to_view[0..1024];
                let tmp_half0 = &uniq (*split_tmp_view)[0..bs];
                let tmp_half1 = &uniq (*split_tmp_view)[bs..1024];
                let input_half0 = &shrd (*ib)[0..bs];
                let input_half1 = &shrd (*ib)[bs..1024];

                sched thread in block {
                    (*tmp_half0)[[thread]] = (*input_half0)[[thread]];
                    (*tmp_half1)[[thread]] = (*input_half1)[[thread]]
                }
            };

            for dd in halved_range(bs) {
                sync(block);
                let tmp_up_view = &uniq (*tmp).to_view.grp::<((1024)/dd)>;
                indep(X) dd block {
                    active =>
                        sched thread in active {
                            let arr = &uniq (*tmp_up_view)[[thread]];
                            (*arr)[(1024)/dd-1] = (*arr)[(1024)/dd-1] + (*arr)[bs/dd-1]
                        },
                    inactive => { () }
                }
            };

            sync(block);
            {
                let tmp_last = &uniq (*tmp).to_view[(1024)-1..1024];
                indep(X) 1 block {
                    active =>
                        sched thread in active {
                            (*bsum)[[thread]] = (*tmp_last)[[thread]];
                            (*tmp_last)[[thread]] = 0
                        },
                    inactive => { () }
                }
            };

           for dd in doubled_range(bs) {
                sync(block);
                let tmp_down_view = &uniq (*tmp).to_view.grp::<(1024)/dd>;

                indep(X) dd block {
                    active =>
                        sched thread in active {
                            let arr = &uniq (*tmp_down_view)[[thread]];
                            let t = (*arr)[bs/dd-1];
                            (*arr)[bs/dd-1] = (*arr)[1024/dd-1];
                            (*arr)[(1024)/dd-1] = (*arr)[1024/dd-1] + t
                        },
                    inactive => { () }
                }
            };

            sync(block);
           {
                let split_tmp_view = &uniq (*tmp).to_view[0..1024];
                let tmp_half0 = &uniq (*split_tmp_view)[0..bs];
                let tmp_half1 = &uniq (*split_tmp_view)[bs..1024];
                let output_half0 = &uniq (*block_out)[0..bs];
                let output_half1 = &uniq (*block_out)[bs..1024];
                sched thread in block {
                    (*output_half0)[[thread]] = (*tmp_half0)[[thread]];
                    (*output_half1)[[thread]] = (*tmp_half1)[[thread]]
                }
            }
        }
    }
}

descend_kernel! {
    fn gpu_add<n: nat, gs: nat, bs: nat, r1: prv, r2: prv>(
        input: &r1 uniq gpu.global [i32; n],
        block_sums: &r2 shrd gpu.global [i32; gs]
    ) -[grid: gpu.grid<X<gs>, X<bs>>]-> () {
        sched block in grid {
            let value_grp = &shrd (*block_sums).to_view.grp::<1>[[block]];
            let chunk_grp = &uniq (*input).to_view.grp::<2*bs>[[block]];
            sched thread in block {
                let chunk = &uniq (*chunk_grp).grp::<2>[[thread]];
                let value = (*value_grp)[0];
                (*chunk)[0] = (*chunk)[0] + value;
                (*chunk)[1] = (*chunk)[1] + value
            }
        }
    }
}

// Simple CPU sequential scan
fn cpu_scan_inplace(data: &mut [i32]) {
    let mut accum = 0;
    for d in data.iter_mut() {
        let next = *d + accum;
        *d = accum;
        accum = next;
    }
}

#[test]
fn test_scan() {
    let ctx = descend_host::DescendContext::new(0);

    let n = 2048;
    let _bs = 512;
    // block dim is bs, but each block processes 1024 elements, so grid size = n / (1024) = 2
    let _gs = 2;

    let mut input_data = vec![0i32; n];
    for i in 0..n {
        input_data[i] = 1;
    }

    let a_box: Box<[i32; 2048]> = input_data.into_boxed_slice().try_into().unwrap();
    let a_cpu = CpuShrd::new(a_box);
    let a_gpu = a_cpu.to_gpu(&ctx);

    let out_box: Box<[i32; 2048]> = vec![0i32; 2048].into_boxed_slice().try_into().unwrap();
    let out_cpu = CpuUniq::new(out_box);
    let mut out_gpu = out_cpu.to_gpu(&ctx);

    let block_sums_box: Box<[i32; 2]> = vec![0i32; 2].into_boxed_slice().try_into().unwrap();
    let block_sums_cpu = CpuUniq::new(block_sums_box);
    let mut block_sums_gpu = block_sums_cpu.to_gpu(&ctx);

    // 1. Run the parallel scan kernel
    descend_launch!(gpu_par_scan, <n: 2048, gs: 2, bs: 512>, &a_gpu, &mut out_gpu, &mut block_sums_gpu);

    // 2. Synchronize and copy block sums back to CPU
    let mut cpu_sums = block_sums_gpu.to_cpu();

    // 3. Scan the block sums on the CPU
    cpu_scan_inplace(&mut *cpu_sums);

    // 4. Send updated sums back to GPU (now as read-only)
    // Wait, GpuUniq::to_cpu consumes the GPU buffer.
    // Let's create a new GpuShrd buffer from the modified CPU sums!
    let block_sums_shrd = CpuShrd::new(cpu_sums.0);
    let block_sums_shrd_gpu = block_sums_shrd.to_gpu(&ctx);

    // 5. Run the add kernel to add the sums to the scanned blocks
    descend_launch!(gpu_add, <n: 2048, gs: 2, bs: 512>, &mut out_gpu, &block_sums_shrd_gpu);

    // 6. Get the final output
    let res = out_gpu.to_cpu();

    // Verify exclusive scan
    let mut expected = 0;
    for i in 0..n {
        if res[i] != expected {
            println!(
                "Mismatch at index {}: expected {}, got {}",
                i, expected, res[i]
            );
            assert_eq!(res[i], expected);
        }
        expected += 1;
    }
}
