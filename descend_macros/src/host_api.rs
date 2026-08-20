use crate::kernel_ir::Kernel;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

/// Generates the mock host launcher wrapper that embeds the compiled PTX
/// and dispatches the GPU kernel using the CUDA driver API.
pub fn generate_host_wrapper(
    kernel: &Kernel,
    ptx_lib_path_str: &str,
    args: Vec<syn::Expr>,
    mangled_name: &str,
    generic_to_size: &std::collections::HashMap<String, String>,
) -> TokenStream2 {
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
                let _type_check: &mut ::descend_host::GpuUniq<_, #expected_size> = #arg_expr;
                let mut #arg_ident = _type_check.dev_buf.cu_deviceptr();
            });
        } else {
            arg_conversions.push(quote! {
                let _type_check: &::descend_host::GpuShrd<_, #expected_size> = #arg_expr;
                let mut #arg_ident = _type_check.dev_buf.cu_deviceptr();
            });
        }

        param_pointers.push(quote! { &mut #arg_ident as *mut _ as *mut std::ffi::c_void });
    }

    let ptx_path_lit = syn::LitStr::new(ptx_lib_path_str, proc_macro2::Span::call_site());
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
            const #ptx_ident: &[u8] = include_bytes!(#ptx_path_lit);

            unsafe {
                let mut module: ::cuda_bindings::CUmodule = std::ptr::null_mut();
                assert_eq!(::cuda_bindings::cuModuleLoadData(&mut module, #ptx_ident.as_ptr() as *const std::ffi::c_void), 0, "cuModuleLoadData failed");

                let mut function: ::cuda_bindings::CUfunction = std::ptr::null_mut();
                assert_eq!(::cuda_bindings::cuModuleGetFunction(&mut function, module, #fn_name_lit.as_ptr() as *const i8), 0, "cuModuleGetFunction failed");

                #( #arg_conversions )*

                let mut args_ptrs: Vec<*mut std::ffi::c_void> = vec![
                    #( #param_pointers ),*
                ];

                assert_eq!(::cuda_bindings::cuLaunchKernel(
                    function,
                    #grid_x, #grid_y, #grid_z, // grid
                    #block_x, #block_y, #block_z, // block
                    0, std::ptr::null_mut(), // shared mem, stream
                    args_ptrs.as_mut_ptr(), std::ptr::null_mut()
                ), 0, "cuLaunchKernel failed");

                assert_eq!(::cuda_bindings::cuCtxSynchronize(), 0, "cuCtxSynchronize failed");

                // Unload module to prevent memory leak on repeated calls
                ::cuda_bindings::cuModuleUnload(module);
            }
        }
    }
}
