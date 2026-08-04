use std::ffi::c_void;
use std::ptr;
use libloading::{Library, Symbol};

pub struct Gpu {
    pub ctx: *mut c_void,
}

impl Gpu {
    pub fn init(device_id: i32) -> Self {
        unsafe {
            // Load libcuda.so (checking both absolute NixOS/openGL driver path and generic path)
            let lib = Library::new("/run/opengl-driver/lib/libcuda.so")
                .or_else(|_| Library::new("libcuda.so"))
                .expect("Failed to load libcuda.so");

            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> = lib.get(b"cuInit").unwrap();
            let cu_device_get: Symbol<unsafe extern "C" fn(*mut i32, i32) -> i32> = lib.get(b"cuDeviceGet").unwrap();
            let cu_ctx_create: Symbol<unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32> = lib.get(b"cuCtxCreate_v2").unwrap();

            assert_eq!(cu_init(0), 0, "cuInit failed");
            let mut dev = 0;
            assert_eq!(cu_device_get(&mut dev, device_id), 0, "cuDeviceGet failed");
            let mut ctx = ptr::null_mut();
            assert_eq!(cu_ctx_create(&mut ctx, 0, dev), 0, "cuCtxCreate failed");

            // Keep the loaded library open globally to avoid unloading
            std::mem::forget(lib);

            Gpu { ctx }
        }
    }
}

pub struct GpuBuffer<T> {
    pub device_ptr: u64,
    pub len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let lib = Library::new("/run/opengl-driver/lib/libcuda.so")
                .or_else(|_| Library::new("libcuda.so"))
                .expect("Failed to load libcuda.so");
            let cu_mem_free: Symbol<unsafe extern "C" fn(u64) -> i32> = lib.get(b"cuMemFree_v2").unwrap();
            cu_mem_free(self.device_ptr);
        }
    }
}

pub fn gpu_alloc_copy<T>(_gpu: &Gpu, host_slice: &[T]) -> GpuBuffer<T> {
    unsafe {
        let lib = Library::new("/run/opengl-driver/lib/libcuda.so")
            .or_else(|_| Library::new("libcuda.so"))
            .expect("Failed to load libcuda.so");
        let cu_mem_alloc: Symbol<unsafe extern "C" fn(*mut u64, usize) -> i32> = lib.get(b"cuMemAlloc_v2").unwrap();
        let cu_memcpy_h_to_d: Symbol<unsafe extern "C" fn(u64, *const c_void, usize) -> i32> = lib.get(b"cuMemcpyHtoD_v2").unwrap();

        let bytes = host_slice.len() * std::mem::size_of::<T>();
        let mut device_ptr = 0;
        assert_eq!(cu_mem_alloc(&mut device_ptr, bytes), 0, "cuMemAlloc failed");
        assert_eq!(cu_memcpy_h_to_d(device_ptr, host_slice.as_ptr() as *const c_void, bytes), 0, "cuMemcpyHtoD failed");

        GpuBuffer {
            device_ptr,
            len: host_slice.len(),
            _marker: std::marker::PhantomData,
        }
    }
}

pub fn copy_to_host<T>(device_buffer: &GpuBuffer<T>, host_slice: &mut [T]) {
    unsafe {
        let lib = Library::new("/run/opengl-driver/lib/libcuda.so")
            .or_else(|_| Library::new("libcuda.so"))
            .expect("Failed to load libcuda.so");
        let cu_memcpy_d_to_h: Symbol<unsafe extern "C" fn(*mut c_void, u64, usize) -> i32> = lib.get(b"cuMemcpyDtoH_v2").unwrap();

        let bytes = device_buffer.len * std::mem::size_of::<T>();
        assert_eq!(cu_memcpy_d_to_h(host_slice.as_mut_ptr() as *mut c_void, device_buffer.device_ptr, bytes), 0, "cuMemcpyDtoH failed");
    }
}
