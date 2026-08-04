use crate::kernel_info::KernelInfo;

/// Synthesizes a valid full Descend program by appending a dummy `main` function wrapper.
/// This satisfies `descend::ty_check`, which currently hardcodes a requirement for a `main` entry point.
pub fn synthesize_main_program(source_str: &str, info: &KernelInfo) -> String {
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
        info.main_params.join(", "),
        info.main_vars.join("\n    "),
        info.fn_name,
        info.gen_params_str,
        info.grid_dim_str,
        info.main_args.join(", ")
    ));
    full_descend
}
