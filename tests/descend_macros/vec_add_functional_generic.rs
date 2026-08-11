use descend_macros::{descend_kernel, descend_launch};

#[test]
fn test_vec_add_generic() {
    // Host-side arrays defined in Rust code before macro invocation.
    let mut a_array = [20i32; 16384];
    let b_array = [10i32; 16384];

    let s = descend_kernel! {
        fn gpu_vec_add<N: nat, r: prv>(
            a_array: &r uniq gpu.global [i32; N],
            b_array: &r shrd gpu.global [i32; N]
        ) -[grid: gpu.grid<X<64>, X<256>>]-> () {
            sched block in grid {
                sched thread in block {
                    let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                    *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
                }
            }
        }
    };

    println!("{}", s);

    // Launch the kernel directly using the descend_launch! macro
    descend_launch!(gpu_vec_add<N: 16384>, &mut a_array, &b_array);
    assert_eq!(a_array, [30i32; 16384])
}
