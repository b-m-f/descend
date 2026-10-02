use crate::artifact::Ptx;
use crate::nvcc::{cached_ptx, resolve_nvcc_options, run_nvcc, store_ptx};

pub fn assemble(cpp: &str, fn_name: &str) -> Result<Ptx, String> {
    let nvcc_args = resolve_nvcc_options()?;
    let ptx = match cached_ptx(cpp, nvcc_args) {
        Some(cached) => {
            debug!("PTX cache hit for {}", fn_name);
            cached
        }
        None => {
            let ptx = run_nvcc(cpp, nvcc_args)?;
            store_ptx(cpp, nvcc_args, &ptx);
            ptx
        }
    };

    let entry = extract_mangled_name(&ptx, fn_name);

    Ok(Ptx { ptx, entry })
}

fn extract_mangled_name(ptx_string: &str, original_fn_name: &str) -> String {
    let mut mangled_name = original_fn_name.to_string();
    for line in ptx_string.lines() {
        if line.starts_with(".visible .entry ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let name = parts[2].trim_end_matches('(');
                if name.contains(original_fn_name) {
                    mangled_name = name.to_string();
                    break;
                }
            }
        }
    }
    mangled_name
}
