use crate::kernel::Kernel;
use crate::spell::format_dimension;

pub fn synthesize_main_function(kernel: &Kernel) -> Result<String, String> {
    let mut main_parameters = Vec::new();
    let mut main_bindings = Vec::new();
    let mut main_arguments = Vec::new();
    let mut shared_memory_configurations = Vec::new();

    for (index, parameter) in kernel.parameters.iter().enumerate() {
        let ownership = if parameter.is_uniq { "uniq" } else { "shrd" };

        if parameter.is_shared_memory {
            shared_memory_configurations.push(parameter.host.descend());
            continue;
        }

        if parameter.is_by_value {
            main_parameters.push(format!("arg{}: {}", index, parameter.host.descend()));
            main_arguments.push(format!("arg{}", index));
            continue;
        }

        // The host holds the buffer's contents in CPU memory; `ownership` is the same
        // ownership the kernel's reference declares.
        main_parameters.push(format!(
            "arg{}: &'a {} cpu.mem {}",
            index,
            ownership,
            parameter.host.descend()
        ));

        main_bindings.push(format!(
            "let mut arg{}_gpu = gpu_alloc_copy(&uniq gpu, &shrd *arg{});",
            index, index
        ));

        main_arguments.push(format!("&{} arg{}_gpu", ownership, index));
    }

    let mut launch_dimensions = format!(
        "{}, {}",
        format_dimension(&kernel.dimensions.grid)?,
        format_dimension(&kernel.dimensions.block)?
    );
    if !shared_memory_configurations.is_empty() {
        launch_dimensions.push_str("; ");
        launch_dimensions.push_str(&shared_memory_configurations.join(", "));
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
        main_parameters.join(", "),
        main_bindings.join("\n    "),
        kernel.function_name,
        launch_dimensions,
        main_arguments.join(", ")
    ))
}
