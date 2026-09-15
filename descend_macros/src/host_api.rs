use crate::kernel_ir::Kernel;
use proc_macro::{Delimiter, Group, TokenStream, TokenTree};
use std::collections::HashMap;
use std::str::FromStr;

pub fn generate_host_wrapper(
    kernel: &Kernel,
    ptx_content: &str,
    stream_expr: TokenStream,
    args: Vec<TokenStream>,
    mangled_name: &str,
    generic_to_size: &HashMap<String, String>,
) -> TokenStream {
    let function = &kernel.function_name;

    if let Some(param) = kernel.params.iter().find(|p| p.is_shared_mem) {
        return compile_error(&format!(
            "kernel `{function}`: parameter `{}` uses gpu.shared memory, which is not supported \
             until a shared-memory layout contract exists",
            param.name
        ));
    }

    if args.len() != kernel.params.len() {
        return compile_error(&format!(
            "kernel `{function}` expects {} buffer argument(s), found {}",
            kernel.params.len(),
            args.len()
        ));
    }

    let mut out = TokenStream::new();
    let mut block = TokenStream::new();

    let mut param_pointers = Vec::new();

    for (i, (param, arg_expr)) in kernel.params.iter().zip(&args).enumerate() {
        let Some(elem_ty) = &param.elem_base_ty else {
            return compile_error(&format!(
                "kernel `{function}`: parameter `{}` has an element type without a supported \
                 host representation",
                param.name
            ));
        };

        let mut expected_size_str = param.size_str.clone();
        for (gen, size) in generic_to_size {
            expected_size_str = substitute_word(&expected_size_str, gen, size);
        }

        let (reference, push) = if param.is_uniq {
            ("&mut", "push_arg_mut")
        } else {
            ("&", "push_arg")
        };
        let decl = format!(
            "let arg_{i}: {reference} ::descend_host::GpuBuffer<{elem_ty}, {{ {expected_size_str} }}> ="
        );
        block.extend(TokenStream::from_str(&decl).unwrap());
        block.extend(arg_expr.clone());
        block.extend(TokenStream::from_str(";").unwrap());

        param_pointers.push(format!("kernel.{push}(arg_{i})?;"));
    }

    let ptx_byte_literal = format!("b\"{}\"", escape_bytes(ptx_content.as_bytes()));

    let (Some(grid), Some(block_ast)) = (&kernel.grid_dim, &kernel.block_dim) else {
        return compile_error(&format!(
            "kernel `{function}` must declare explicit grid and block dimensions"
        ));
    };
    let grid_dim = match eval_dim(grid, generic_to_size) {
        Ok(dim) => dim,
        Err(error) => return compile_error(&error),
    };
    let block_dim = match eval_dim(block_ast, generic_to_size) {
        Ok(dim) => dim,
        Err(error) => return compile_error(&error),
    };

    let shared_mem_bytes = "0";

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

fn compile_error(message: &str) -> TokenStream {
    let escaped = format!("compile_error!({message:?})");
    TokenStream::from_str(&escaped).expect("compile_error token generation must succeed")
}

pub fn eval_nat(nat: &Nat, bindings: &HashMap<String, String>) -> Result<u32, String> {
    match nat {
        Nat::Lit(val) => {
            u32::try_from(*val).map_err(|_| format!("dimension literal {val} exceeds u32::MAX"))
        }
        Nat::Ident(ident) => {
            let name = ident.name.to_string();
            if let Some(val_str) = bindings.get(&name) {
                val_str
                    .parse::<u32>()
                    .map_err(|_| format!("generic dimension {name}={val_str:?} is not a valid u32"))
            } else {
                Err(format!(
                    "unknown identifier in dimension expression: {name}"
                ))
            }
        }
        Nat::BinOp(op, lhs, rhs) => {
            let l = eval_nat(lhs, bindings)?;
            let r = eval_nat(rhs, bindings)?;
            match op {
                descend::ast::BinOpNat::Add => l.checked_add(r),
                descend::ast::BinOpNat::Sub => l.checked_sub(r),
                descend::ast::BinOpNat::Mul => l.checked_mul(r),
                descend::ast::BinOpNat::Div => l.checked_div(r),
                descend::ast::BinOpNat::Mod => l.checked_rem(r),
            }
            .ok_or_else(|| format!("invalid or overflowing dimension expression: {l} {op:?} {r}"))
        }
        _ => Err(format!("unsupported dimension expression: {nat:?}")),
    }
}

pub fn eval_dim(
    dim: &descend::ast::Dim,
    bindings: &HashMap<String, String>,
) -> Result<(u32, u32, u32), String> {
    Ok(match dim {
        descend::ast::Dim::X(n) => (eval_nat(&n.0, bindings)?, 1, 1),
        descend::ast::Dim::Y(n) => (1, eval_nat(&n.0, bindings)?, 1),
        descend::ast::Dim::Z(n) => (1, 1, eval_nat(&n.0, bindings)?),
        descend::ast::Dim::XY(d) => (eval_nat(&d.0, bindings)?, eval_nat(&d.1, bindings)?, 1),
        descend::ast::Dim::XZ(d) => (eval_nat(&d.0, bindings)?, 1, eval_nat(&d.1, bindings)?),
        descend::ast::Dim::YZ(d) => (1, eval_nat(&d.0, bindings)?, eval_nat(&d.1, bindings)?),
        descend::ast::Dim::XYZ(d) => (
            eval_nat(&d.0, bindings)?,
            eval_nat(&d.1, bindings)?,
            eval_nat(&d.2, bindings)?,
        ),
    })
}
