use descend_macros::descend;

fn main() {
    // let a_h = [];
    // let a = GPU::init(type,size)

    let s = descend! {
        fn gpu_vec_add<r: prv>(
            a_array: &r uniq gpu.global [i32; 16384],
            b_array: &r shrd gpu.global [i32; 16384]
        ) -[grid: gpu.grid<X<64>, X<256>>]-> () {
            sched block in grid {
                sched thread in block {
                    let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                    *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
                }
            }
        }
    };
    // kernel_call(gpu_vec_add, a_array: , b_array: ,
    println!("{}", s);
}
