use crate::kernel_ir::Kernel;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

/// Generates the mock host launcher wrapper that embeds the compiled PTX
/// and dispatches the GPU kernel using the CUDA driver API.
pub fn generate_host_wrapper(
    kernel: &Kernel,
    ptx_content: &str,
    args: Vec<syn::Expr>,
    mangled_name: &str,
    generic_to_size: &std::collections::HashMap<String, String>,
) -> TokenStream2 {
    if args.is_empty() {
        return quote! {
            compile_error!("Descend kernels must take at least one array argument to operate upon.");
        };
    }

    let mut arg_conversions = Vec::new();
    let mut param_pointers = Vec::new();

    for (i, param) in kernel.params.iter().enumerate() {
        if i >= args.len() {
            break;
        }
        let arg_expr = &args[i];

        let expected_size_str = generic_to_size
            .get(&param.size_str)
            .unwrap_or(&param.size_str);
        let expected_size: syn::Expr =
            syn::parse_str(expected_size_str).unwrap_or_else(|_| syn::parse_str("0").unwrap());

        let arg_ident = syn::Ident::new(&format!("arg_{}", i), proc_macro2::Span::call_site());

        if param.is_uniq {
            arg_conversions.push(quote! {
                let mut #arg_ident: &mut ::descend_host::GpuUniq<_, #expected_size> = #arg_expr;
            });
            param_pointers.push(quote! { &mut #arg_ident.dev_buf });
        } else {
            arg_conversions.push(quote! {
                let #arg_ident: &::descend_host::GpuShrd<_, #expected_size> = #arg_expr;
            });
            param_pointers.push(quote! { &#arg_ident.dev_buf });
        }
    }

    let ptx_byte_literal =
        syn::LitByteStr::new(ptx_content.as_bytes(), proc_macro2::Span::call_site());
    let mut grid_dim = (1, 1, 1);
    let mut block_dim = (1, 1, 1);

    // Parse grid and block dims from grid_dim_str which is like "X<64>, X<256>"
    let dims_str = kernel
        .grid_dim_str
        .replace("X<", "")
        .replace(">", "")
        .replace(" ", "");
    let mut parts = dims_str.split(',');

    if let Some(x) = parts.next() {
        grid_dim.0 = x.parse().unwrap_or(1);
    }
    if let Some(x) = parts.next() {
        block_dim.0 = x.parse().unwrap_or(1);
    }

    let grid_x = grid_dim.0 as u32;
    let grid_y = grid_dim.1 as u32;
    let grid_z = grid_dim.2 as u32;

    let block_x = block_dim.0 as u32;
    let block_y = block_dim.1 as u32;
    let block_z = block_dim.2 as u32;

    let fn_name_c_str = format!("{}\0", mangled_name);
    let fn_name_lit =
        syn::LitByteStr::new(fn_name_c_str.as_bytes(), proc_macro2::Span::call_site());

    let ptx_ident = syn::Ident::new(
        &format!("{}_PTX", kernel.function_name.to_uppercase()),
        proc_macro2::Span::call_site(),
    );

    quote! {
        {
            use ::descend_host::cudarc::driver::PushKernelArg;

            const #ptx_ident: &[u8] = #ptx_byte_literal;

            #( #arg_conversions )*

            let stream = arg_0.dev_buf.stream().clone();
            // the host API has this baked into the GPU array types
            let ctx = stream.context();

            let fn_name_str = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(#fn_name_lit).to_str().unwrap() };
            let ptx_str = std::str::from_utf8(#ptx_ident).unwrap();

            // Load the module temporarily for this launch (matches original cuModuleLoadData behavior)
            let module = ctx.load_module(::descend_host::cudarc::nvrtc::Ptx::from_src(ptx_str)).expect("Failed to load PTX");
            let mut f = module.load_function(fn_name_str).expect("Failed to get function");

            let cfg = ::descend_host::cudarc::driver::LaunchConfig {
                grid_dim: (#grid_x, #grid_y, #grid_z),
                block_dim: (#block_x, #block_y, #block_z),
                shared_mem_bytes: 0,
            };

            let mut builder = stream.launch_builder(&mut f);
            #( builder.arg(#param_pointers); )*

            unsafe { builder.launch(cfg) }.expect("Kernel launch failed");
        }
    }
}
