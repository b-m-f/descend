use crate::kernel_ir::Kernel;
use std::collections::HashMap;
use proc_macro::{TokenStream, TokenTree, Group, Delimiter};
use std::str::FromStr;

/// Orchestrates the embedding of compiled payload to standard Rust APIs.
/// 
/// What it does: Re-generates `cudarc` `LaunchConfig` arithmetic dynamically, constructs strongly typed mutable/immutable CUDA references, and returns a callable anonymous Rust module wrapped securely.
/// Why it is needed: Forms the actual host-side execution binary allowing `descend_launch!` to trigger GPU resources via `cudarc` at runtime seamlessly.
pub fn generate_host_wrapper(
    kernel: &Kernel,
    ptx_content: &str,
    args: Vec<TokenStream>,
    mangled_name: &str,
    generic_to_size: &HashMap<String, String>,
) -> TokenStream {
    if args.is_empty() {
        return TokenStream::from_str("compile_error!(\"Descend kernels must take at least one array argument to operate upon.\");").unwrap();
    }

    let mut out = TokenStream::new();
    let mut block = TokenStream::new();

    let mut param_pointers = Vec::new();

    for (i, param) in kernel.params.iter().enumerate() {
        if i >= args.len() {
            break;
        }
        let arg_expr = &args[i];

        let mut expected_size_str = param.size_str.clone();
        for (gen, size) in generic_to_size {
            expected_size_str = substitute_word(&expected_size_str, gen, size);
        }

        if param.is_uniq {
            let decl = format!("let mut arg_{}: &mut ::descend_host::GpuUniq<_, {{ {} }}> =", i, expected_size_str);
            block.extend(TokenStream::from_str(&decl).unwrap());
            block.extend(arg_expr.clone());
            block.extend(TokenStream::from_str(";").unwrap());
            param_pointers.push(format!("&mut arg_{}.dev_buf", i));
        } else {
            let decl = format!("let arg_{}: &::descend_host::GpuShrd<_, {{ {} }}> =", i, expected_size_str);
            block.extend(TokenStream::from_str(&decl).unwrap());
            block.extend(arg_expr.clone());
            block.extend(TokenStream::from_str(";").unwrap());
            param_pointers.push(format!("&arg_{}.dev_buf", i));
        }
    }

    let ptx_byte_literal = format!("b\"{}\"", escape_bytes(ptx_content.as_bytes()));
    
    let mut grid_dim = (1, 1, 1);
    let mut block_dim = (1, 1, 1);
    

    let mut actual_grid_dim_str = kernel.grid_dim_str.clone();
    for (gen, size) in generic_to_size {
        actual_grid_dim_str = substitute_word(&actual_grid_dim_str, gen, size);
    }
    let grid_block_parts: Vec<&str> = actual_grid_dim_str.split(">,").collect();
    let mut gb_parsed = Vec::new();
    for part in grid_block_parts {
        let s = part.replace("XYZ<", "")
                    .replace("XY<", "")
                    .replace("XZ<", "")
                    .replace("YZ<", "")
                    .replace("X<", "")
                    .replace("Y<", "")
                    .replace("Z<", "")
                    .replace(">", "")
                    .replace(" ", "");
        let mut dims = (1, 1, 1);
        let vals: Vec<&str> = s.split(',').collect();
        if vals.len() > 0 { dims.0 = eval_math(vals[0]).unwrap_or(1); }
        if vals.len() > 1 { dims.1 = eval_math(vals[1]).unwrap_or(1); }
        if vals.len() > 2 { dims.2 = eval_math(vals[2]).unwrap_or(1); }
        gb_parsed.push(dims);
    }
    
    if gb_parsed.len() > 0 {
        grid_dim = gb_parsed[0];
    }
    if gb_parsed.len() > 1 {
        block_dim = gb_parsed[1];
    }

    let mut shared_mem_exprs = vec![];
    for param in &kernel.params {
        if param.is_shared_mem {
            let mut sz = param.size_str.clone();
            for (gen, size) in generic_to_size {
                sz = substitute_word(&sz, gen, size);
            }
            if let Some(num_elements) = eval_math(&sz) {
                shared_mem_exprs.push(format!("({} * std::mem::size_of::<{}>())", num_elements, param.elem_base_ty));
            }
        }
    }
    
    let shared_mem_bytes = if shared_mem_exprs.is_empty() {
        "0".to_string()
    } else {
        shared_mem_exprs.join(" + ")
    };

    let builder_args = param_pointers
        .into_iter()
        .map(|p| format!("builder.arg({});", p))
        .collect::<Vec<_>>()
        .join("\n        ");

    let core_str = format!("
        use ::descend_host::cudarc::driver::PushKernelArg;
        let stream = arg_0.dev_buf.stream().clone();
        let ctx = stream.context();
        let fn_name_str = unsafe {{
            std::ffi::CStr::from_bytes_with_nul_unchecked(b\"{}\\0\").to_str().unwrap()
        }};
        let ptx_str = std::str::from_utf8({}).unwrap();
        let module = ctx.load_module(::descend_host::cudarc::nvrtc::Ptx::from_src(ptx_str)).expect(\"Failed to load PTX\");
        let mut f = module.load_function(fn_name_str).expect(\"Failed to get function\");
        let cfg = ::descend_host::cudarc::driver::LaunchConfig {{
            grid_dim: ({}, {}, {}),
            block_dim: ({}, {}, {}),
            shared_mem_bytes: {},
        }};
        let mut builder = stream.launch_builder(&mut f);
        {}
        unsafe {{ builder.launch(cfg) }}.expect(\"Kernel launch failed\");
    ", mangled_name, ptx_byte_literal, grid_dim.0, grid_dim.1, grid_dim.2, block_dim.0, block_dim.1, block_dim.2, shared_mem_bytes, builder_args);

    block.extend(TokenStream::from_str(&core_str).unwrap());
    
    out.extend(std::iter::once(TokenTree::Group(Group::new(Delimiter::Brace, block))));
    out
}

fn escape_bytes(bytes: &[u8]) -> String {
    let mut s = String::new();
    for &b in bytes {
        if b == b'\"' {
            s.push_str("\\\"");
        } else if b == b'\\' {
            s.push_str("\\\\");
        } else if b >= 32 && b <= 126 {
            s.push(b as char);
        } else {
            s.push_str(&format!("\\x{:02x}", b));
        }
    }
    s
}

/// Statically evaluates elementary math operations embedded within dimension parameters.
/// 
/// What it does: Splits common syntax bounds (like `/` or `*`) to perform lightweight local reductions during macro execution.
/// Why it is needed: Provides fallback dimensionality estimations directly to the macro builder independent of external evaluations.
fn eval_math(expr: &str) -> Option<u32> {
    if expr.contains('*') {
        let parts: Vec<&str> = expr.split('*').collect();
        if parts.len() == 2 {
            let left = parts[0].trim().parse::<u32>().unwrap_or(1);
            let right = parts[1].trim().parse::<u32>().unwrap_or(1);
            return Some(left * right);
        }
    }
    if expr.contains('/') {
        let parts: Vec<&str> = expr.split('/').collect();
        if parts.len() == 2 {
            let left = parts[0].trim().parse::<u32>().unwrap_or(1);
            let right = parts[1].trim().parse::<u32>().unwrap_or(1);
            if right != 0 {
                return Some(left / right);
            }
        }
    }
    expr.trim().parse::<u32>().ok()
}

pub fn substitute_word(s: &str, word: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut last_end = 0;

    let bytes = s.as_bytes();
    let word_bytes = word.as_bytes();
    let mut i = 0;
    while i + word.len() <= bytes.len() {
        if &bytes[i..i + word.len()] == word_bytes {
            let prev_ok = if i == 0 {
                true
            } else {
                !bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_'
            };
            let next_idx = i + word.len();
            let next_ok = if next_idx == bytes.len() {
                true
            } else {
                !bytes[next_idx].is_ascii_alphanumeric() && bytes[next_idx] != b'_'
            };

            if prev_ok && next_ok {
                result.push_str(&s[last_end..i]);
                result.push_str(replacement);
                i += word.len();
                last_end = i;
                continue;
            }
        }
        i += 1;
    }
    result.push_str(&s[last_end..]);
    result
}
