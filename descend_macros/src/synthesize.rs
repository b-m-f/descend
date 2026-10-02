use crate::kernel::Kernel;
use crate::spell::format_dim;

pub fn synthesize_main_fn(kernel: &Kernel) -> Result<String, String> {
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

    let mut launch_dims = format!(
        "{}, {}",
        format_dim(&kernel.dims.grid)?,
        format_dim(&kernel.dims.block)?
    );
    if !shared_mem_configs.is_empty() {
        launch_dims.push_str("; ");
        launch_dims.push_str(&shared_mem_configs.join(", "));
    }

    Ok(format!(
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
        launch_dims,
        main_args.join(", ")
    ))
}
