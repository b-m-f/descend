//! The Rust items a kernel becomes: a launch function and the constant holding
//! the configuration derived from the kernel's own declaration.

use crate::compile::CompiledKernel;
use crate::ir::Kernel;
use crate::nat::{eval_dim, eval_nat};
use crate::shared_mem::check_shared_mem;
use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
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

    let mut parameters: Vec<TokenStream> = vec![quote! { stream: &::descend_host::Stream }];
    let mut pushes: Vec<TokenStream> = Vec::new();
    for param in kernel.params.iter().filter(|p| !p.is_shared_mem) {
        let binding = rust_parameter_name(&param.name, function)?;

        if param.is_by_value {
            let value_ty = match &param.value_dty {
                Some(dty) => crate::ast::format_dty_eval(dty, bindings)?,
                None => param.value_ty_str.clone(),
            };
            let name = format_ident!("{}", binding);
            let value_ty = TokenStream::from_str(&value_ty).map_err(|error| {
                format!(
                    "kernel `{function}`: parameter `{}` has a type the generator cannot \
                         spell in Rust: {error}",
                    param.name
                )
            })?;
            parameters.push(quote! { #name: #value_ty });
            pushes.push(quote! { kernel.push_arg_val(&#name)?; });
            continue;
        }

        let Some(elem_ty) = &param.elem_base_ty else {
            return Err(format!(
                "kernel `{function}`: parameter `{}` has an element type without a supported \
                 host representation",
                param.name
            ));
        };
        // The buffer's element count is evaluated from the parameter's own
        // `Nat`s, so the generated type carries a literal rather than an
        // expression patched together from text.
        let mut elements: u64 = 1;
        for nat in &param.dim_nats {
            let dim = eval_nat(nat, bindings)?;
            elements = elements.checked_mul(u64::from(dim)).ok_or_else(|| {
                format!(
                    "kernel `{function}`: parameter `{}` has more elements than can be counted",
                    param.name
                )
            })?;
        }

        let name = format_ident!("{}", binding);
        let elem = format_ident!("{}", elem_ty);
        let elements = Literal::usize_unsuffixed(elements as usize);
        let contents = quote! { ::descend_host::GpuBuffer<[#elem; #elements]> };

        if param.is_uniq {
            parameters.push(quote! { #name: &mut #contents });
            pushes.push(quote! { kernel.push_arg_uniq(#name)?; });
        } else {
            parameters.push(quote! { #name: &#contents });
            pushes.push(quote! { kernel.push_arg_shrd(#name)?; });
        }
    }

    let config_name = format_ident!("{}_CONFIG", item_name.to_uppercase());
    let function_name = format_ident!("{}", item_name);
    let entry = &compiled.mangled_name;
    let ptx = Literal::byte_string(compiled.ptx.as_bytes());

    let (grid_x, grid_y, grid_z) = grid_dim;
    let (block_x, block_y, block_z) = block_dim;
    let shared_mem_bytes =
        Literal::u32_unsuffixed(u32::try_from(compiled.shared_mem_bytes).map_err(|_| {
            format!(
                "kernel `{function}` requests {} bytes of shared memory, more than a launch can \
                 express",
                compiled.shared_mem_bytes
            )
        })?);

    Ok(quote! {
        /// The launch configuration derived from this kernel's declaration.
        pub const #config_name: ::descend_host::KernelLaunchConfig =
            ::descend_host::KernelLaunchConfig {
                grid_dim: (#grid_x, #grid_y, #grid_z),
                block_dim: (#block_x, #block_y, #block_z),
                shared_mem_bytes: #shared_mem_bytes,
            };

        #[allow(dead_code)]
        pub fn #function_name(
            #(#parameters),*
        ) -> ::core::result::Result<(), ::descend_host::error::HostError> {
            const PTX_BYTES: &[u8] = #ptx;
            let mut kernel = ::descend_host::Kernel::new(#entry, #config_name);
            #(#pushes)*
            let launcher = ::descend_host::KernelLauncher::new(
                stream,
                ::core::str::from_utf8(PTX_BYTES).unwrap(),
            )?;
            // SAFETY: the PTX is this kernel's own compilation, the entry name was
            // read back out of it, the arguments are pushed in the kernel's
            // parameter order with the ownership its Descend types declare, and the
            // launch configuration comes from the kernel's own grid declaration and
            // shared-memory parameters.
            unsafe { launcher.launch(kernel) }
        }
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ordinary_parameter_name_is_kept() {
        assert_eq!(rust_parameter_name("a_array", "k").unwrap(), "a_array");
    }

    #[test]
    fn a_rust_keyword_becomes_a_raw_identifier() {
        // Descend and Rust do not reserve the same words.
        assert_eq!(rust_parameter_name("type", "k").unwrap(), "r#type");
        assert_eq!(rust_parameter_name("move", "k").unwrap(), "r#move");
        assert_eq!(rust_parameter_name("ref", "k").unwrap(), "r#ref");
    }

    #[test]
    fn an_identifier_rust_cannot_make_raw_is_reported() {
        for name in ["self", "Self", "crate", "super", "_"] {
            let error = rust_parameter_name(name, "gpu_k").unwrap_err();
            assert!(
                error.contains(name) && error.contains("gpu_k"),
                "unhelpful error for `{name}`: {error}"
            );
        }
    }
}
