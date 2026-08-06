use descend_macros::descend;

pub mod gpu {
    use libloading::os::unix::Library;
    use std::ffi::c_void;

    pub struct GpuBuffer<T> {
        pub device_ptr: *mut T,
        pub size: usize,
    }

    impl<T: Copy> GpuBuffer<T> {
        pub fn from_slice(slice: &[T]) -> Self {
            unsafe {
                let cuda_lib = Library::open(
                    Some("/run/opengl-driver/lib/libcuda.so"),
                    libloading::os::unix::RTLD_GLOBAL | libloading::os::unix::RTLD_LAZY,
                )
                .or_else(|_| {
                    Library::open(
                        Some("libcuda.so"),
                        libloading::os::unix::RTLD_GLOBAL | libloading::os::unix::RTLD_LAZY,
                    )
                })
                .expect("Failed to load libcuda.so");

                let cu_init = cuda_lib
                    .get::<unsafe extern "C" fn(u32) -> i32>(b"cuInit")
                    .unwrap();
                let cu_device_get = cuda_lib
                    .get::<unsafe extern "C" fn(*mut i32, i32) -> i32>(b"cuDeviceGet")
                    .unwrap();
                let cu_ctx_create = cuda_lib
                    .get::<unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32>(
                        b"cuCtxCreate_v2",
                    )
                    .unwrap();
                let cu_mem_alloc = cuda_lib
                    .get::<unsafe extern "C" fn(*mut *mut c_void, usize) -> i32>(b"cuMemAlloc_v2")
                    .unwrap();
                let cu_memcpy_htod = cuda_lib
                    .get::<unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> i32>(
                        b"cuMemcpyHtoD_v2",
                    )
                    .unwrap();

                assert_eq!(cu_init(0), 0, "cuInit failed");
                let mut device = 0;
                assert_eq!(cu_device_get(&mut device, 0), 0, "cuDeviceGet failed");
                let mut context = std::ptr::null_mut();
                assert_eq!(
                    cu_ctx_create(&mut context, 0, device),
                    0,
                    "cuCtxCreate failed"
                );

                let byte_size = slice.len() * std::mem::size_of::<T>();
                let mut dptr = std::ptr::null_mut();
                assert_eq!(cu_mem_alloc(&mut dptr, byte_size), 0, "cuMemAlloc failed");
                assert_eq!(
                    cu_memcpy_htod(dptr, slice.as_ptr() as *const c_void, byte_size),
                    0,
                    "cuMemcpyHtoD failed"
                );

                GpuBuffer {
                    device_ptr: dptr as *mut T,
                    size: slice.len(),
                }
            }
        }

        pub fn copy_to_vec(&self) -> Vec<T> {
            unsafe {
                let cuda_lib = Library::open(
                    Some("/run/opengl-driver/lib/libcuda.so"),
                    libloading::os::unix::RTLD_GLOBAL | libloading::os::unix::RTLD_LAZY,
                )
                .or_else(|_| {
                    Library::open(
                        Some("libcuda.so"),
                        libloading::os::unix::RTLD_GLOBAL | libloading::os::unix::RTLD_LAZY,
                    )
                })
                .expect("Failed to load libcuda.so");

                let cu_memcpy_dtoh = cuda_lib
                    .get::<unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> i32>(
                        b"cuMemcpyDtoH_v2",
                    )
                    .unwrap();

                let mut host_vec = vec![std::mem::zeroed::<T>(); self.size];
                let byte_size = self.size * std::mem::size_of::<T>();
                assert_eq!(
                    cu_memcpy_dtoh(
                        host_vec.as_mut_ptr() as *mut c_void,
                        self.device_ptr as *mut c_void,
                        byte_size
                    ),
                    0,
                    "cuMemcpyDtoH failed"
                );
                host_vec
            }
        }
    }
}

descend! {
    fn test_gpu_vec_add<r: prv>(
        a_array: &r uniq gpu.global [i32; 16384],
        b_array: &r shrd gpu.global [i32; 16384]
    ) -[grid: gpu.grid<X<64>, X<256>>]-> () {
        sched block in grid {
            sched thread in block {
                let a_ref = &uniq (*a_array).to_view.grp::<256>[[block]][[thread]];
                *a_ref = *a_ref + (*b_array).to_view.grp::<256>[[block]][[thread]]
            }
        }
    }
}

#[test]
fn test_macro_end_to_end_gpu_execution() {
    const SIZE: usize = 16384;
    let host_a = vec![15i32; SIZE];
    let host_b = vec![25i32; SIZE];

    let mut dev_a = gpu::GpuBuffer::from_slice(&host_a);
    let dev_b = gpu::GpuBuffer::from_slice(&host_b);

    test_gpu_vec_add(&mut dev_a, &dev_b);

    let result = dev_a.copy_to_vec();
    assert_eq!(result.len(), SIZE);

    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, 40,
            "Verification failed at index {}: expected 40, got {}",
            i, val
        );
    }
}
