use std::collections::HashMap;
use crate::kernel_ir::Kernel;
use proc_macro::{Delimiter, Group, TokenStream, TokenTree};
use std::str::FromStr;

pub fn generate_host_wrapper(
    kernel: &Kernel,
    ptx_content: &str,
    stream_expr: TokenStream,
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
            let decl = format!(
                "let mut arg_{}: &mut ::descend_host::GpuBuffer<_, {{ {} }}> =",
                i, expected_size_str
            );
            block.extend(TokenStream::from_str(&decl).unwrap());
            block.extend(arg_expr.clone());
            block.extend(TokenStream::from_str(";").unwrap());
        } else {
            let decl = format!(
                "let arg_{}: &::descend_host::GpuBuffer<_, {{ {} }}> =",
                i, expected_size_str
            );
            block.extend(TokenStream::from_str(&decl).unwrap());
            block.extend(arg_expr.clone());
            block.extend(TokenStream::from_str(";").unwrap());
        }

        param_pointers.push(format!("kernel.push_arg(arg_{})?;", i));
    }

    let ptx_byte_literal = format!("b\"{}\"", escape_bytes(ptx_content.as_bytes()));

    let grid_dim = if let Some(g) = &kernel.grid_dim {
        eval_dim(g, generic_to_size)
    } else {
        panic!("Kernel must specify explicit grid dimensions in AST")
    };

    let block_dim = if let Some(b) = &kernel.block_dim {
        eval_dim(b, generic_to_size)
    } else {
        panic!("Kernel must specify explicit block dimensions in AST")
    };

    let mut shared_mem_exprs = vec![];
    for param in &kernel.params {
        if param.is_shared_mem {
            let mut sz = param.size_str.clone();
            for (gen, size) in generic_to_size {
                sz = substitute_word(&sz, gen, size);
            }
            shared_mem_exprs.push(format!(
                "({} * std::mem::size_of::<{}>())",
                sz, param.elem_base_ty
            ));
        }
    }

    let shared_mem_bytes = if shared_mem_exprs.is_empty() {
        "0".to_string()
    } else {
        shared_mem_exprs.join(" + ")
    };

    let ptrs = param_pointers.join("\n        ");

    let core_str = format!(
        "
        (|| -> Result<(), ::descend_host::error::HostError> {{
            let fn_name_str = unsafe {{
                std::ffi::CStr::from_bytes_with_nul_unchecked(b\"{}\\0\").to_str().unwrap()
            }};
        let ptx_str = std::str::from_utf8({}).unwrap();
        
        let config = ::descend_host::KernelLaunchConfig {{
            grid_dim: ({}, {}, {}),
            block_dim: ({}, {}, {}),
            shared_mem_bytes: {},
        }};
        
        let mut kernel = ::descend_host::Kernel::new(fn_name_str, config);
        {}
        
        let launcher = ::descend_host::KernelLauncher::new(__STREAM_EXPR__, ptx_str)?;
        launcher.launch(kernel)
    }})()
    ",
        mangled_name,
        ptx_byte_literal,
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        shared_mem_bytes,
        ptrs
    );

    let parsed = TokenStream::from_str(&core_str).unwrap();
    let replaced = replace_placeholder(parsed, "__STREAM_EXPR__", &stream_expr);
    block.extend(replaced);

    out.extend(std::iter::once(TokenTree::Group(Group::new(
        Delimiter::Brace,
        block,
    ))));
    out
}

fn replace_placeholder(
    ts: TokenStream,
    placeholder: &str,
    replacement: &TokenStream,
) -> TokenStream {
    let mut out = TokenStream::new();
    for tt in ts {
        match tt {
            TokenTree::Group(g) => {
                let mut new_g = Group::new(
                    g.delimiter(),
                    replace_placeholder(g.stream(), placeholder, replacement),
                );
                new_g.set_span(g.span());
                out.extend(std::iter::once(TokenTree::Group(new_g)));
            }
            TokenTree::Ident(ref id) if id.to_string() == placeholder => {
                out.extend(replacement.clone());
            }
            _ => out.extend(std::iter::once(tt)),
        }
    }
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
use descend::ast::Nat;

pub fn eval_nat(nat: &Nat, bindings: &HashMap<String, String>) -> u32 {
    match nat {
        Nat::Lit(val) => *val as u32,
        Nat::Ident(ident) => {
            let name = ident.name.to_string();
            if let Some(val_str) = bindings.get(&name) {
                val_str.parse::<u32>().unwrap_or_else(|_| panic!("Failed to parse bound generic {} as u32", name))
            } else {
                panic!("Unknown identifier in Nat expression: {}", name)
            }
        },
        Nat::BinOp(op, lhs, rhs) => {
            let l = eval_nat(lhs, bindings);
            let r = eval_nat(rhs, bindings);
            match op {
                descend::ast::BinOpNat::Add => l + r,
                descend::ast::BinOpNat::Sub => l - r,
                descend::ast::BinOpNat::Mul => l * r,
                descend::ast::BinOpNat::Div => l / r,
                descend::ast::BinOpNat::Mod => l % r,
            }
        },
        _ => panic!("Unsupported Nat expression type for dimension evaluation: {:?}", nat),
    }
}

pub fn eval_dim(dim: &descend::ast::Dim, bindings: &HashMap<String, String>) -> (u32, u32, u32) {
    match dim {
        descend::ast::Dim::X(n) => (eval_nat(&n.0, bindings), 1, 1),
        descend::ast::Dim::Y(n) => (1, eval_nat(&n.0, bindings), 1),
        descend::ast::Dim::Z(n) => (1, 1, eval_nat(&n.0, bindings)),
        descend::ast::Dim::XY(d) => (eval_nat(&d.0, bindings), eval_nat(&d.1, bindings), 1),
        descend::ast::Dim::XZ(d) => (eval_nat(&d.0, bindings), 1, eval_nat(&d.1, bindings)),
        descend::ast::Dim::YZ(d) => (1, eval_nat(&d.0, bindings), eval_nat(&d.1, bindings)),
        descend::ast::Dim::XYZ(d) => (eval_nat(&d.0, bindings), eval_nat(&d.1, bindings), eval_nat(&d.2, bindings)),
    }
}
