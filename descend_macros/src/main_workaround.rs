use crate::kernel_info::Kernel;

/// Synthesizes a valid full Descend program by appending a dummy `main` function wrapper.
/// This satisfies `descend::ty_check`, which currently hardcodes a requirement for a `main` entry point.
pub fn synthesize_main_program(source_str: &str, kernel: &Kernel) -> String {
    let mut main_params = Vec::new();
    let mut main_vars = Vec::new();
    let mut main_args = Vec::new();

    for param in &kernel.params {
        let rgn = if param.is_uniq { "uniq" } else { "shrd" };
        main_params.push(format!(
            "h_{}: &'a {} cpu.mem [i32; {}]",
            param.name, rgn, param.size_str
        ));
        main_vars.push(format!(
            "let mut {} = gpu_alloc_copy(&uniq gpu, &shrd *h_{});",
            param.name, param.name
        ));
        main_args.push(format!("&{} {}", rgn, param.name));
    }

    let mut full_descend = source_str.to_string();
    full_descend.push_str(&format!(
        "
fn main({}) -[t: cpu.thread]-> () <'a> {{
    let mut gpu = gpu_device(0);
    {}
    {}{}::<<<{}>>>({});
    ()
}}
",
        main_params.join(", "),
        main_vars.join("\n    "),
        kernel.function_name,
        kernel.generic_params_str,
        kernel.grid_dim_str,
        main_args.join(", ")
    ));
    full_descend
}
