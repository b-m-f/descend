use descend_host::{Context, GpuBuffer};
use descend_macros::{descend, descend_launch};

descend! {
    // Tests: Mul (*), Add (+), Div (/), Sub (-), Mod (%)
    fn gpu_vec_add_arithmetic<N: nat, M: nat, O: nat, r: prv>(
        a_array: &r uniq gpu.global [i32; (N * M) + O],
        b_array: &r shrd gpu.global [i32; (N / M) - (O % 3)]
    ) -[grid: gpu.grid<X<64>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                // simple computation to verify parsing logic
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
            }
        }
    }
}

#[test]
fn test_vec_add_arithmetic() {
    let ctx = Context::new(0);
    let stream = ctx.new_stream();

    // Provide values that make (N * M) + O == 16384 and (N / M) - (O % 3) == 16384
    // E.g. N = 16384, M = 1, O = 0 -> (16384 * 1) + 0 = 16384, (16384 / 1) - 0 = 16384
    let a_array = [20i32; 16384];
    let b_array = [10i32; 16384];

    let mut a_gpu = GpuBuffer::to_gpu(&Box::new(a_array), &stream);
    let b_gpu = GpuBuffer::to_gpu(&Box::new(b_array), &stream);

    // Testing parsing of multiple generics and their multiplication inside the Descend AST!
    descend_launch!(gpu_vec_add_arithmetic, &stream, <N: 16384, M: 1, O: 0>, &mut a_gpu, &b_gpu);

    let a_cpu_back = a_gpu.to_cpu(&stream);
    assert_eq!(*a_cpu_back, [30i32; 16384]);
}
