use crate::kernel_ir::Kernel;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

/// Generates the mock host launcher wrapper that embeds the compiled PTX
/// and dispatches the GPU kernel using the CUDA driver API.
pub fn generate_host_wrapper(kernel: &Kernel, ptx_lib_path_str: &str) -> TokenStream2 {
    let fn_name = &kernel.function_name;

    let mut rust_params = Vec::new();
    let mut rust_args = Vec::new();
    let mut arg_idents = Vec::new();

    for param in &kernel.params {
        let arg_ident = syn::Ident::new(&param.name, proc_macro2::Span::call_site());
        let (qual, c_qual) = if param.is_uniq {
            ("&mut", "*mut")
        } else {
            ("&", "*const")
        };

        let rust_type: syn::Type =
            syn::parse_str(&format!("{} gpu::GpuBuffer<i32>", qual)).unwrap();
        let c_type: syn::Type = syn::parse_str(&format!("{} i32", c_qual)).unwrap();

        rust_params.push(quote! { #arg_ident: #rust_type });
        rust_args.push(quote! { #arg_ident.device_ptr as #c_type });
        arg_idents.push(arg_ident.clone());
    }

    let fn_ident = syn::Ident::new(fn_name, proc_macro2::Span::call_site());
    let ptx_ident = syn::Ident::new(
        &format!("{}_PTX", fn_name.to_uppercase()),
        proc_macro2::Span::call_site(),
    );
    let kernel_name_str = format!("{}\0", fn_name);

    quote! {
        pub const #ptx_ident: &[u8] = include_bytes!(#ptx_lib_path_str);

        // Strongly typed launcher wrapper using libcuda.so as our mock "cuda-oxide" runtime
        pub fn #fn_ident(#(#rust_params),*) {
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

                // MOCK cuda-oxide logic: raw driver API calls to load PTX and launch kernel
                // In a real system, the user would use a proper crate here, but we embed this
                // raw loading logic to satisfy the example compilation & execution seamlessly.
                let cu_init = _cuda_lib.get::<unsafe extern "C" fn(u32) -> i32>(b"cuInit").unwrap();
                let cu_device_get = _cuda_lib.get::<unsafe extern "C" fn(*mut i32, i32) -> i32>(b"cuDeviceGet").unwrap();
                let cu_ctx_create = _cuda_lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> i32>(b"cuCtxCreate_v2").unwrap();

                let cu_module_load_data = _cuda_lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, *const std::ffi::c_void) -> i32>(b"cuModuleLoadData").unwrap();
                let cu_module_get_function = _cuda_lib.get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, *mut std::ffi::c_void, *const i8) -> i32>(b"cuModuleGetFunction").unwrap();
                let cu_launch_kernel = _cuda_lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, u32, u32, u32, u32, u32, u32, u32, *mut std::ffi::c_void, *mut *mut std::ffi::c_void, *mut *mut std::ffi::c_void) -> i32>(b"cuLaunchKernel").unwrap();

                // Initialize CUDA Driver API and create a mock context for the kernel
                assert_eq!(cu_init(0), 0, "cuInit failed");
                let mut device: i32 = 0;
                assert_eq!(cu_device_get(&mut device, 0), 0, "cuDeviceGet failed");
                let mut context: *mut std::ffi::c_void = std::ptr::null_mut();
                assert_eq!(cu_ctx_create(&mut context, 0, device), 0, "cuCtxCreate failed");

                let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
                assert_eq!(cu_module_load_data(&mut module, #ptx_ident.as_ptr() as *const std::ffi::c_void), 0, "cuModuleLoadData failed");

                let mut function: *mut std::ffi::c_void = std::ptr::null_mut();
                assert_eq!(cu_module_get_function(&mut function, module, #kernel_name_str.as_ptr() as *const i8), 0, "cuModuleGetFunction failed");

                #(
                    let mut #arg_idents = #rust_args;
                )*
                // Execute the kernel (assuming 64 blocks of 256 threads as per concrete example)
                let mut args_ptrs: Vec<*mut std::ffi::c_void> = vec![
                    #( &mut #arg_idents as *mut _ as *mut std::ffi::c_void ),*
                ];

                assert_eq!(cu_launch_kernel(
                    function,
                    64, 1, 1, // grid
                    256, 1, 1, // block
                    0, std::ptr::null_mut(), // shared mem, stream
                    args_ptrs.as_mut_ptr(), std::ptr::null_mut()
                ), 0, "cuLaunchKernel failed");
            }
        }
    }
}
