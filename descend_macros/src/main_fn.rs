use crate::kernel::Kernel;

pub fn synthesize_main_fn(kernel: &Kernel) -> String {
    let mut main_params = Vec::new();
    let mut main_vars = Vec::new();
    let mut main_args = Vec::new();
    let mut shared_mem_configs = Vec::new();

    for (i, param) in kernel.params.iter().enumerate() {
        let mem_qual = if param.is_uniq { "uniq" } else { "shrd" };

        if param.is_shared_mem {
            shared_mem_configs.push(param.host.descend());
            continue;
        }

        if param.is_by_value {
            main_params.push(format!("arg{}: {}", i, param.host.descend()));
            main_args.push(format!("arg{}", i));
            continue;
        }

        // The host holds the buffer's contents in CPU memory; `mem_qual` is the same
        // ownership the kernel's reference declares.
        main_params.push(format!(
            "arg{}: &'a {} cpu.mem {}",
            i,
            mem_qual,
            param.host.descend()
        ));

        main_vars.push(format!(
            "let mut arg{}_gpu = gpu_alloc_copy(&uniq gpu, &shrd *arg{});",
            i, i
        ));

        main_args.push(format!("&{} arg{}_gpu", mem_qual, i));
    }

    let mut grid_dim_str = kernel.grid_dim_str.clone();
    if !shared_mem_configs.is_empty() {
        grid_dim_str.push_str("; ");
        grid_dim_str.push_str(&shared_mem_configs.join(", "));
    }

    format!(
        "
fn main({}) -[t: cpu.thread]-> () <'a> {{
    let mut gpu = gpu_device(0);
    {}
    {}::<<<{}>>>({});
    ()
}}
",
        main_params.join(", "),
        main_vars.join("\n    "),
        kernel.function_name,
        grid_dim_str,
        main_args.join(", ")
    )
}
