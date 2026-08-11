use crate::kernel_ir::Kernel;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

/// Generates the mock host launcher wrapper that embeds the compiled PTX
/// and dispatches the GPU kernel using the CUDA driver API.
pub fn generate_host_wrapper(
    kernel: &Kernel,
    ptx_lib_path_str: &str,
    args: Vec<String>,
    mangled_name: &str,
    generic_to_size: &std::collections::HashMap<String, String>,
) -> TokenStream2 {
    let mut arg_conversions = Vec::new();
    let mut param_pointers = Vec::new();
    let mut host_to_device = Vec::new();
    let mut device_to_host = Vec::new();
    let mut frees = Vec::new();

    for (i, param) in kernel.params.iter().enumerate() {
        if i >= args.len() {
            break;
        }
        let arg_expr_str = &args[i];
        let arg_expr: syn::Expr = syn::parse_str(arg_expr_str).unwrap_or_else(|_| {
            syn::parse_str("()").unwrap()
        });
        
        let arg_ident = syn::Ident::new(&format!("arg_{}", i), proc_macro2::Span::call_site());
        let dev_ptr_ident = syn::Ident::new(&format!("dev_ptr_{}", i), proc_macro2::Span::call_site());
        
        // Compute byte size
        let mut actual_size_str = param.size_str.clone();
        for (gen, size) in generic_to_size {
            actual_size_str = crate::main_workaround::replace_word(&actual_size_str, gen, size);
        }
        let size_expr: syn::Expr = syn::parse_str(&actual_size_str).unwrap_or_else(|_| {
            syn::parse_str("0").unwrap()
        });
        
        // We assume i32 for now, as the user only uses i32 arrays.
        let bytes_size = quote! { ((#size_expr) as usize) * std::mem::size_of::<i32>() };

        arg_conversions.push(quote! { 
            let mut #arg_ident = (#arg_expr).as_mut_ptr() as *mut i32; 
            let mut #dev_ptr_ident: *mut std::ffi::c_void = std::ptr::null_mut();
        });
        
        host_to_device.push(quote! {
            assert_eq!(cu_mem_alloc(&mut #dev_ptr_ident, #bytes_size), 0, "cuMemAlloc failed");
            assert_eq!(cu_memcpy_htod(#dev_ptr_ident, #arg_ident as *const std::ffi::c_void, #bytes_size), 0, "cuMemcpyHtoD failed");
        });
        
        if param.is_uniq {
            device_to_host.push(quote! {
                assert_eq!(cu_memcpy_dtoh(#arg_ident as *mut std::ffi::c_void, #dev_ptr_ident, #bytes_size), 0, "cuMemcpyDtoH failed");
            });
        }
        
        frees.push(quote! {
            cu_mem_free(#dev_ptr_ident);
        });
        
        param_pointers.push(quote! { &mut #dev_ptr_ident as *mut _ as *mut std::ffi::c_void });
    }

    let ptx_path_lit = syn::LitStr::new(ptx_lib_path_str, proc_macro2::Span::call_site());
    let mut grid_dim = (1, 1, 1);
    let mut block_dim = (1, 1, 1);

    // Parse grid and block dims from grid_dim_str which is like "X<64>, X<256>"
    let dims_str = kernel.grid_dim_str.replace("X<", "").replace(">", "").replace(" ", "");
    let mut parts = dims_str.split(',');
    
    // First element is usually grid x
    if let Some(x) = parts.next() { grid_dim.0 = x.parse().unwrap_or(1); }
    // Second element is usually block x
    if let Some(x) = parts.next() { block_dim.0 = x.parse().unwrap_or(1); }

    let grid_x = grid_dim.0 as u32;
    let grid_y = grid_dim.1 as u32;
    let grid_z = grid_dim.2 as u32;

    let block_x = block_dim.0 as u32;
    let block_y = block_dim.1 as u32;
    let block_z = block_dim.2 as u32;

    let fn_name_c_str = format!("{}\0", mangled_name);
    let fn_name_lit = syn::LitByteStr::new(fn_name_c_str.as_bytes(), proc_macro2::Span::call_site());

    let ptx_ident = syn::Ident::new(
        &format!("{}_PTX", kernel.function_name.to_uppercase()),
        proc_macro2::Span::call_site(),
    );

    quote! {
        {
            const #ptx_ident: &[u8] = include_bytes!(#ptx_path_lit);
            unsafe {
                let _cuda_lib = ::libloading::os::unix::Library::open(
                    Some("/run/opengl-driver/lib/libcuda.so"),
                    ::libloading::os::unix::RTLD_GLOBAL | ::libloading::os::unix::RTLD_LAZY
                ).or_else(|_| {
                    ::libloading::os::unix::Library::open(
                        Some("libcuda.so"),
                        ::libloading::os::unix::RTLD_GLOBAL | ::libloading::os::unix::RTLD_LAZY
                    )
                }).expect("Failed to load libcuda.so");

                let cu_init = _cuda_lib.get::<unsafe extern "C" fn(u32) -> i32>(b"cuInit").unwrap();
                let cu_device_get = _cuda_lib.get::<unsafe extern "C" fn(*mut i32, i32) -> i32>(b"cuDeviceGet").unwrap();
                let cu_ctx_create = _cuda_lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> i32>(b"cuCtxCreate_v2").unwrap();

                let cu_module_load_data = _cuda_lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, *const std::ffi::c_void) -> i32>(b"cuModuleLoadData").unwrap();
                let cu_module_get_function = _cuda_lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, *mut std::ffi::c_void, *const i8) -> i32>(b"cuModuleGetFunction").unwrap();
                let cu_launch_kernel = _cuda_lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, u32, u32, u32, u32, u32, u32, u32, *mut std::ffi::c_void, *mut *mut std::ffi::c_void, *mut *mut std::ffi::c_void) -> i32>(b"cuLaunchKernel").unwrap();
                let cu_ctx_synchronize = _cuda_lib.get::<unsafe extern "C" fn() -> i32>(b"cuCtxSynchronize").unwrap();
                
                let cu_mem_alloc = _cuda_lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, usize) -> i32>(b"cuMemAlloc_v2").unwrap();
                let cu_memcpy_htod = _cuda_lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, usize) -> i32>(b"cuMemcpyHtoD_v2").unwrap();
                let cu_memcpy_dtoh = _cuda_lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, usize) -> i32>(b"cuMemcpyDtoH_v2").unwrap();
                let cu_mem_free = _cuda_lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>(b"cuMemFree_v2").unwrap();

                assert_eq!(cu_init(0), 0, "cuInit failed");
                let mut device: i32 = 0;
                assert_eq!(cu_device_get(&mut device, 0), 0, "cuDeviceGet failed");
                let mut context: *mut std::ffi::c_void = std::ptr::null_mut();
                assert_eq!(cu_ctx_create(&mut context, 0, device), 0, "cuCtxCreate failed");

                let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
                assert_eq!(cu_module_load_data(&mut module, #ptx_ident.as_ptr() as *const std::ffi::c_void), 0, "cuModuleLoadData failed");

                let mut function: *mut std::ffi::c_void = std::ptr::null_mut();
                assert_eq!(cu_module_get_function(&mut function, module, #fn_name_lit.as_ptr() as *const i8), 0, "cuModuleGetFunction failed");

                #( #arg_conversions )*
                #( #host_to_device )*

                let mut args_ptrs: Vec<*mut std::ffi::c_void> = vec![
                    #( #param_pointers ),*
                ];

                assert_eq!(cu_launch_kernel(
                    function,
                    #grid_x, #grid_y, #grid_z, // grid
                    #block_x, #block_y, #block_z, // block
                    0, std::ptr::null_mut(), // shared mem, stream
                    args_ptrs.as_mut_ptr(), std::ptr::null_mut()
                ), 0, "cuLaunchKernel failed");
                
                assert_eq!(cu_ctx_synchronize(), 0, "cuCtxSynchronize failed");
                
                #( #device_to_host )*
                #( #frees )*
            }
        }
    }
}
