use crate::backend::CudaModule;
use crate::context::Context;
use crate::error::HostError;
use crate::gpu_buffer::GpuBuffer;
use crate::stream::Stream;
use crate::traits::DescendType;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

pub struct KernelLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

enum Param {
    Device(u64),
    Val(NonNull<c_void>),
}

pub struct Kernel<'a> {
    pub fn_name: &'a str,
    pub config: KernelLaunchConfig,
    params: Vec<Param>,
    context: Option<Context>,
}

impl<'a> Kernel<'a> {
    pub fn new(fn_name: &'a str, config: KernelLaunchConfig) -> Self {
        Self {
            fn_name,
            config,
            params: Vec::new(),
            context: None,
        }
    }

    pub fn push_arg_shrd<T: DescendType>(
        &mut self,
        buf: &'a GpuBuffer<T>,
    ) -> Result<(), HostError> {
        self.push_arg_dev(buf)
    }

    pub fn push_arg_uniq<T: DescendType>(
        &mut self,
        buf: &'a mut GpuBuffer<T>,
    ) -> Result<(), HostError> {
        self.push_arg_dev(buf)
    }

    pub fn push_arg_val<T: DescendType>(&mut self, value: &'a T) -> Result<(), HostError> {
        if std::mem::size_of::<T>() == 0 {
            return Err(HostError::InvalidSize("Zero-sized types are not supported"));
        }
        self.params
            .push(Param::Val(NonNull::from(value).cast::<c_void>()));
        Ok(())
    }

    fn push_arg_dev<T: DescendType>(&mut self, buf: &GpuBuffer<T>) -> Result<(), HostError> {
        if let Some(ctx) = &self.context {
            if ctx.inner != buf.context().inner {
                return Err(HostError::ContextMismatch(
                    "Buffer context does not match existing kernel arguments context",
                ));
            }
        } else {
            self.context = Some(buf.context());
        }
        self.params.push(Param::Device(buf.dev_buf.cu_deviceptr()));
        Ok(())
    }

    fn param_ptrs(&mut self) -> Vec<*mut c_void> {
        self.params
            .iter_mut()
            .map(|param| match param {
                Param::Device(address) => std::ptr::from_mut(address).cast::<c_void>(),
                Param::Val(value) => value.as_ptr(),
            })
            .collect()
    }
}

pub struct KernelLauncher<'a> {
    stream: &'a Stream,
    module: Arc<CudaModule>,
}

impl<'a> KernelLauncher<'a> {
    pub fn new(stream: &'a Stream, ptx_str: &str) -> Result<Self, HostError> {
        std::ffi::CString::new(ptx_str).map_err(|e| HostError::NulError(e, "PTX source"))?;
        let module = stream
            .context
            .inner
            .load_module_from_ptx_src(ptx_str)
            .map_err(|e| HostError::driver(e, "load module"))?;
        Ok(Self { stream, module })
    }

    pub unsafe fn launch(&self, mut kernel: Kernel) -> Result<(), HostError> {
        if let Some(ctx) = &kernel.context {
            if ctx.inner != self.stream.context.inner {
                return Err(HostError::ContextMismatch(
                    "Kernel arguments context does not match execution stream context",
                ));
            }
        }
        std::ffi::CString::new(kernel.fn_name)
            .map_err(|e| HostError::NulError(e, "kernel function name"))?;
        let f = self
            .module
            .load_function(kernel.fn_name)
            .map_err(|e| HostError::driver(e, "load function"))?;

        let mut ptrs = kernel.param_ptrs();

        unsafe {
            crate::backend::launch_kernel_on_stream(
                &f,
                kernel.config.grid_dim,
                kernel.config.block_dim,
                kernel.config.shared_mem_bytes,
                &self.stream.inner,
                &mut ptrs,
            )
        }
        .map_err(|e| HostError::driver(e, "kernel launch"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct Record(i32, i32, i32, i32);
    unsafe impl DescendType for Record {}

    #[derive(Clone, Copy)]
    struct Zst;
    unsafe impl DescendType for Zst {}

    fn config() -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn address_of<T>(value: &T) -> *mut c_void {
        std::ptr::from_ref(value).cast::<c_void>().cast_mut()
    }

    #[test]
    fn val_slots_hold_the_callers_addresses() {
        let integer = 7i32;
        let float = 1.5f32;
        let flag = true;
        let array = [1i32, 2, 3];
        let record = Record(1, 2, 3, 4);

        let mut kernel = Kernel::new("kernel", config());
        kernel.push_arg_val(&integer).unwrap();
        kernel.push_arg_val(&float).unwrap();
        kernel.push_arg_val(&flag).unwrap();
        kernel.push_arg_val(&array).unwrap();
        kernel.push_arg_val(&record).unwrap();

        let ptrs = kernel.param_ptrs();
        assert_eq!(
            ptrs,
            vec![
                address_of(&integer),
                address_of(&float),
                address_of(&flag),
                address_of(&array),
                address_of(&record),
            ],
            "each slot must hold the address of the caller's own value, in push order"
        );
    }

    #[test]
    fn val_slots_are_aligned_for_their_types() {
        let integer = 7i32;
        let double = 1.5f64;
        let array = [1i32, 2, 3];
        let record = Record(1, 2, 3, 4);

        let mut kernel = Kernel::new("kernel", config());
        kernel.push_arg_val(&integer).unwrap();
        kernel.push_arg_val(&double).unwrap();
        kernel.push_arg_val(&array).unwrap();
        kernel.push_arg_val(&record).unwrap();

        let ptrs = kernel.param_ptrs();
        let alignments = [
            std::mem::align_of::<i32>(),
            std::mem::align_of::<f64>(),
            std::mem::align_of::<[i32; 3]>(),
            std::mem::align_of::<Record>(),
        ];
        for (ptr, align) in ptrs.iter().zip(alignments) {
            assert_eq!(
                *ptr as usize % align,
                0,
                "a by-value slot has the alignment of the caller's value"
            );
        }
    }

    #[test]
    fn val_rejects_zero_sized_values() {
        let zst = Zst;
        let mut kernel = Kernel::new("kernel", config());
        assert!(matches!(
            kernel.push_arg_val(&zst),
            Err(HostError::InvalidSize("Zero-sized types are not supported"))
        ));
        assert!(kernel.param_ptrs().is_empty());
    }

    #[test]
    fn rejected_push_leaves_earlier_slots_in_order() {
        let first = 1i32;
        let zst = Zst;
        let second = 2i32;

        let mut kernel = Kernel::new("kernel", config());
        kernel.push_arg_val(&first).unwrap();
        assert!(kernel.push_arg_val(&zst).is_err());
        kernel.push_arg_val(&second).unwrap();

        assert_eq!(
            kernel.param_ptrs(),
            vec![address_of(&first), address_of(&second)]
        );
    }
}
