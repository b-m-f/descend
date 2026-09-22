use crate::compiler::CompiledKernel;
use crate::kernel_ir::Kernel;
use proc_macro::TokenStream;
use std::collections::HashMap;
use std::str::FromStr;

/// Generates a callable Rust function for a kernel whose sizes are all known
/// where it is defined, plus a constant holding the launch configuration the
/// kernel's own declaration determines.
///
/// The function is safe. `KernelLauncher::launch` is unsafe because a caller
/// pushing arguments by hand cannot be held to the kernel's signature; here the
/// PTX, the entry symbol, the parameter types, their order and the launch
/// configuration are all generated from one Descend definition, so the
/// obligations that method documents are discharged by construction. What
/// remains delegated is a user's own `unsafe impl DescendType` for a struct
/// element type.
pub fn generate_launch_fn(
    kernel: &Kernel,
    compiled: &CompiledKernel,
    bindings: &HashMap<String, String>,
    item_name: &str,
) -> Result<TokenStream, String> {
    let function = &kernel.function_name;

    check_shared_mem(kernel, compiled.shared_mem_bytes, bindings)?;

    let (Some(grid), Some(block)) = (&kernel.grid_dim, &kernel.block_dim) else {
        return Err(format!(
            "kernel `{function}` must declare explicit grid and block dimensions"
        ));
    };
    let grid_dim = eval_dim(grid, bindings)?;
    let block_dim = eval_dim(block, bindings)?;

    let mut parameters = vec!["stream: &::descend_host::Stream".to_string()];
    let mut pushes = Vec::new();
    for param in kernel.params.iter().filter(|p| !p.is_shared_mem) {
        let binding = rust_parameter_name(&param.name, function)?;

        if param.is_by_value {
            let mut value_ty = param.value_ty_str.clone();
            for (generic, value) in bindings {
                value_ty = substitute_word(&value_ty, generic, value);
            }
            parameters.push(format!("{binding}: {value_ty}"));
            pushes.push(format!("kernel.push_arg_val(&{binding})?;"));
            continue;
        }

        let Some(elem_ty) = &param.elem_base_ty else {
            return Err(format!(
                "kernel `{function}`: parameter `{}` has an element type without a supported \
                 host representation",
                param.name
            ));
        };
        if param.cpu_ty_str == "UNKNOWN_TY" {
            return Err(format!(
                "kernel `{function}`: parameter `{}` is taken by value, which the macro cannot \
                 pass yet; only reference parameters in gpu.global or gpu.shared memory are \
                 supported",
                param.name
            ));
        }

        let mut size_str = param.size_str.clone();
        for (generic, value) in bindings {
            size_str = substitute_word(&size_str, generic, value);
        }

        let binding = rust_parameter_name(&param.name, function)?;
        let (reference, push) = if param.is_uniq {
            ("&mut", "push_arg_uniq")
        } else {
            ("&", "push_arg_shrd")
        };
        parameters.push(format!(
            "{binding}: {reference} ::descend_host::GpuBuffer<[{elem_ty}; {{ {size_str} }}]>"
        ));
        pushes.push(format!("kernel.{push}({binding})?;"));
    }

    let config_name = format!("{}_CONFIG", item_name.to_uppercase());
    let ptx_byte_literal = format!("b\"{}\"", escape_bytes(compiled.ptx.as_bytes()));

    let code = format!(
        "
        /// The launch configuration derived from this kernel's declaration.
        pub const {config_name}: ::descend_host::KernelLaunchConfig =
            ::descend_host::KernelLaunchConfig {{
                grid_dim: ({}u32, {}u32, {}u32),
                block_dim: ({}u32, {}u32, {}u32),
                shared_mem_bytes: {}u32,
            }};

        #[allow(dead_code)]
        pub fn {item_name}({}) -> ::core::result::Result<(), ::descend_host::error::HostError> {{
            const PTX_BYTES: &[u8] = {};
            let mut kernel = ::descend_host::Kernel::new({:?}, {config_name});
            {}
            let launcher = ::descend_host::KernelLauncher::new(
                stream,
                ::core::str::from_utf8(PTX_BYTES).unwrap(),
            )?;
            // SAFETY: the PTX is this kernel's own compilation, the entry name was
            // read back out of it, the arguments are pushed in the kernel's
            // parameter order with the ownership its Descend types declare, and the
            // launch configuration comes from the kernel's own grid declaration and
            // shared-memory parameters.
            unsafe {{ launcher.launch(kernel) }}
        }}
        ",
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        compiled.shared_mem_bytes,
        parameters.join(",\n            "),
        ptx_byte_literal,
        compiled.mangled_name,
        pushes.join("\n            "),
    );

    TokenStream::from_str(&code)
        .map_err(|error| format!("kernel `{function}`: generated code is malformed: {error}"))
}

/// The Rust binding name for a Descend parameter.
///
/// Descend and Rust do not reserve the same words, so a parameter may need the
/// raw form. The few identifiers Rust cannot make raw are reported rather than
/// silently renamed, because the generated function's parameter names are part
/// of what a caller reads.
fn rust_parameter_name(name: &str, function: &str) -> Result<String, String> {
    const UNRAWABLE: [&str; 5] = ["self", "Self", "crate", "super", "_"];
    const KEYWORDS: [&str; 49] = [
        "as", "async", "await", "break", "const", "continue", "dyn", "else", "enum", "extern",
        "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
        "pub", "ref", "return", "static", "struct", "trait", "true", "type", "union", "unsafe",
        "use", "where", "while", "abstract", "become", "box", "do", "final", "macro", "override",
        "priv", "try", "typeof", "unsized", "virtual", "yield", "gen",
    ];

    if UNRAWABLE.contains(&name) {
        return Err(format!(
            "kernel `{function}`: parameter `{name}` cannot name a Rust function parameter"
        ));
    }
    if KEYWORDS.contains(&name) {
        return Ok(format!("r#{name}"));
    }
    Ok(name.to_string())
}

/// Checks what can be known about a kernel's shared-memory parameters from its
/// signature alone.
///
/// This runs before the kernel is compiled, because both failures are otherwise
/// reported by the Descend compiler in a form the user cannot act on: a
/// mis-ordered shared parameter as a type error in synthesized source they never
/// wrote, and an element type without a defined byte size as a panic inside the
/// code generator.
pub fn check_shared_params(kernel: &Kernel) -> Result<(), String> {
    let function = &kernel.function_name;
    let mut seen_shared = false;

    for param in &kernel.params {
        if !param.is_shared_mem {
            if seen_shared {
                return Err(format!(
                    "kernel `{function}`: parameter `{}` follows a gpu.shared parameter. \
                     Shared-memory parameters must come last, because the compiler removes \
                     them from the end of the kernel signature",
                    param.name
                ));
            }
            continue;
        }
        seen_shared = true;

        if param.elem_bytes.is_none() {
            return Err(format!(
                "kernel `{function}`: shared parameter `{}` has an element type whose byte \
                 size the Descend compiler does not define, so its shared-memory allocation \
                 cannot be sized",
                param.name
            ));
        }
    }

    Ok(())
}

/// Checks the dynamic shared-memory request the generated launch will make.
///
/// The number itself comes from the Descend code generator, which also emitted
/// the device code that slices the buffer. Recomputing it from the parameter
/// types is a second opinion on the same contract: element size times element
/// count, summed over the shared parameters, with no padding term because the
/// generator orders the arrays by descending element size precisely so that none
/// is needed. A disagreement would mean the launch allocates less than the
/// kernel reads, which no runtime test can be relied on to expose.
fn check_shared_mem(
    kernel: &Kernel,
    emitted: u64,
    bindings: &HashMap<String, String>,
) -> Result<(), String> {
    let function = &kernel.function_name;
    let mut expected: u64 = 0;

    for param in &kernel.params {
        if !param.is_shared_mem {
            continue;
        }

        let elem_bytes = param.elem_bytes.ok_or_else(|| {
            format!(
                "kernel `{function}`: shared parameter `{}` has an element type whose byte \
                 size the Descend compiler does not define, so its shared-memory allocation \
                 cannot be sized",
                param.name
            )
        })?;

        let mut elements: u64 = 1;
        for nat in &param.dim_nats {
            let dim = eval_nat(nat, bindings)?;
            elements = elements.checked_mul(u64::from(dim)).ok_or_else(|| {
                format!(
                    "kernel `{function}`: shared parameter `{}` has more elements than can be \
                     counted",
                    param.name
                )
            })?;
        }
        expected += u64::from(elem_bytes) * elements;
    }

    if expected != emitted {
        return Err(format!(
            "internal error: shared-memory byte count disagreement for kernel `{function}`: \
             the Descend compiler's generated launch requests {emitted} bytes, the parameter \
             types give {expected}. Refusing to launch"
        ));
    }

    // Above 48 KiB a kernel must opt in through cuFuncSetAttribute, which the
    // host API does not do; the launch would fail in the driver instead.
    if emitted > 49152 {
        return Err(format!(
            "kernel `{function}` requests {emitted} bytes of dynamic shared memory, above the \
             48 KiB a launch may request without opting in through cuFuncSetAttribute"
        ));
    }

    Ok(())
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
