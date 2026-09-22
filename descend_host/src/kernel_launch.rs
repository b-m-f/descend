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

/// One argument slot, in the kernel's signature order.
enum Param {
    /// The slot holds the buffer's device address.
    Device(u64),
    /// The slot is the caller's own value.
    Scalar(NonNull<c_void>),
}

/// Arguments and configuration for one kernel submission.
///
/// Every pushed argument — a buffer or a value passed by value — stays borrowed
/// for `'a`, i.e. until the `Kernel` is consumed by [`KernelLauncher::launch`].
/// Because launch is synchronous, safe code cannot drop a pushed argument during
/// execution, and a buffer pushed with [`Kernel::push_arg_uniq`] cannot be read
/// or pushed again until completion.
///
/// Arguments of both kinds share one positional sequence, because order across
/// argument kinds is the kernel's signature order.
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

    /// Pushes a buffer for a parameter the kernel only reads (`shrd`).
    pub fn push_arg_shrd<T: DescendType>(&mut self, buf: &'a GpuBuffer<T>) -> Result<(), HostError> {
        self.push_arg_dev(buf)
    }

    /// Pushes a buffer for a parameter the kernel may write (`uniq`).
    pub fn push_arg_uniq<T: DescendType>(
        &mut self,
        buf: &'a mut GpuBuffer<T>,
    ) -> Result<(), HostError> {
        self.push_arg_dev(buf)
    }

    /// Pushes a value for a parameter the kernel takes by value.
    ///
    /// The value stays with the caller, so its storage has exactly `T`'s size
    /// and alignment; only its address is recorded. The borrow must outlive this
    /// `Kernel`, so safe code cannot drop or move the value before
    /// [`KernelLauncher::launch`] consumes the kernel — a computed temporary is
    /// therefore rejected and the caller binds a local, as it already does for
    /// buffers. (A borrow of a literal is promoted to `'static` and is
    /// accepted.) The driver takes the number of
    /// bytes to read from the compiled kernel signature, so no size is recorded
    /// here.
    ///
    /// A value has no device allocation and hence no context, so this records
    /// none: a kernel whose arguments are all by value is not context-checked.
    pub fn push_arg_val<T: DescendType>(&mut self, value: &'a T) -> Result<(), HostError> {
        if std::mem::size_of::<T>() == 0 {
            return Err(HostError::InvalidSize("Zero-sized types are not supported"));
        }
        self.params
            .push(Param::Scalar(NonNull::from(value).cast::<c_void>()));
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
        self.params
            .push(Param::Device(crate::backend::buffer_device_ptr(
                &buf.dev_buf,
            )));
        Ok(())
    }

    /// Builds the driver's argument array: one pointer per slot, in push order.
    /// A device slot yields the address of its own address word, a by-value slot
    /// the address of the caller's value.
    fn param_ptrs(&mut self) -> Vec<*mut c_void> {
        self.params
            .iter_mut()
            .map(|param| match param {
                Param::Device(address) => std::ptr::from_mut(address).cast::<c_void>(),
                Param::Scalar(value) => value.as_ptr(),
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
        let module = crate::backend::load_module_from_ptx(&stream.context.inner, ptx_str)
            .map_err(|e| HostError::driver(e, "load module"))?;
        Ok(Self { stream, module })
    }

    /// Submits `kernel` on this launcher's stream and waits for completion.
    ///
    /// # Safety
    ///
    /// The borrow checker already guarantees that every pushed argument outlives
    /// this call and that buffers pushed with `push_arg_uniq` are not accessed
    /// by other safe code until it returns. The caller must additionally
    /// guarantee facts that raw PTX does not expose:
    ///
    /// - **Signature**: the named kernel exists and its argument count, order
    ///   and ABI match the pushed arguments. For a buffer, element size,
    ///   alignment, representation, size and address space must match; for a
    ///   value pushed with `push_arg_val`, the kernel's parameter must itself be
    ///   taken by value and its size, alignment and representation must agree
    ///   with the pushed `T`. The driver reads a by-value argument's bytes
    ///   according to the compiled signature, not according to the host slot, so
    ///   a mismatch misreads host memory instead of failing.
    /// - **Permissions**: every parameter the kernel may write was pushed with
    ///   `push_arg_uniq`, and no kernel running on another stream accesses the
    ///   same buffers in a conflicting way. A by-value parameter cannot be
    ///   written through and so needs no exclusivity, but the pointer handed to
    ///   the driver for it derives from a shared reference: the kernel must only
    ///   read that storage.
    /// - **Execution parameters**: grid, block and shared-memory sizes satisfy
    ///   the kernel's indexing assumptions.
    /// - **Output validity**: every value the kernel writes is a valid value of
    ///   its Rust `DescendType`, including the restricted encodings of `bool`.
    ///
    /// `push_arg_val` records no context, so a kernel whose arguments are all by
    /// value passes the context check below unexamined; its module and stream
    /// context agreement is the caller's obligation.
    ///
    /// Success means all work submitted to the stream, including this kernel,
    /// has completed. If the driver's stream synchronization fails, completion
    /// is unknown and the process terminates before argument, allocation or
    /// module storage is released. Errors recorded by earlier resource releases
    /// do not cause termination; see [`Context::check_deferred_errors`].
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
        let f = crate::backend::load_function(&self.module, kernel.fn_name)
            .map_err(|e| HostError::driver(e, "load function"))?;
        crate::backend::bind_context_to_thread(&self.stream.context.inner)
            .map_err(|e| HostError::driver(e, "bind context for kernel launch"))?;

        let mut ptrs = kernel.param_ptrs();

        let launch_result = unsafe {
            crate::backend::launch(
                &f,
                kernel.config.grid_dim,
                kernel.config.block_dim,
                kernel.config.shared_mem_bytes,
                &self.stream.inner,
                &mut ptrs,
            )
        };

        let completion = crate::backend::synchronize_stream(&self.stream.inner);
        if completion.is_err() {
            crate::backend::fail_stop("kernel completion");
        }

        launch_result.map_err(|e| HostError::driver(e, "kernel launch"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16-byte value: the case a fixed-width argument slot would truncate.
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
