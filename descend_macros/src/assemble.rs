use crate::artifact::Ptx;
use crate::nvcc::{cached_ptx, resolve_nvcc_options, run_nvcc, store_ptx};

pub fn assemble(cpp: &str, kernel_name: &str) -> Result<Ptx, String> {
    let nvcc_arguments = resolve_nvcc_options()?;
    let ptx_text = match cached_ptx(cpp, nvcc_arguments) {
        Some(cached) => {
            debug!("PTX cache hit for {}", kernel_name);
            cached
        }
        None => {
            let ptx_text = run_nvcc(cpp, nvcc_arguments)?;
            store_ptx(cpp, nvcc_arguments, &ptx_text);
            ptx_text
        }
    };

    let entry = extract_mangled_name(&ptx_text, kernel_name);

    Ok(Ptx {
        text: ptx_text,
        entry,
    })
}

fn extract_mangled_name(ptx_text: &str, kernel_name: &str) -> String {
    let mut mangled_name = kernel_name.to_string();
    for line in ptx_text.lines() {
        if line.starts_with(".visible .entry ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let name = parts[2].trim_end_matches('(');
                if name.contains(kernel_name) {
                    mangled_name = name.to_string();
                    break;
                }
            }
        }
    }
    mangled_name
}
