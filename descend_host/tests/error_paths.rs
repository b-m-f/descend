use descend_host::error::HostError;
use descend_host::{Context, GpuBuffer, Kernel, KernelLaunchConfig, KernelLauncher};

const NOOP_PTX: &str =
    ".version 7.0\n.target sm_52\n.address_size 64\n\n.visible .entry noop()\n{\n\tret;\n}\n";

fn config() -> KernelLaunchConfig {
    KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn second_device() -> Option<Context> {
    match Context::new(1) {
        Ok(ctx) => Some(ctx),
        Err(HostError::Driver(source, _))
            if source.to_string().contains("invalid device ordinal") =>
        {
            None
        }
        Err(other) => panic!("unexpected error creating a context on device 1: {other:?}"),
    }
}

#[test]
fn ptx_with_an_interior_nul_is_rejected_before_the_driver() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    match KernelLauncher::new(&stream, "\0.version 7.0\n") {
        Err(HostError::NulError(_, "PTX source")) => {}
        Err(other) => panic!("expected a NUL error for the PTX source, got {other:?}"),
        Ok(_) => panic!("a PTX string with an interior NUL unexpectedly loaded"),
    }
}

#[test]
fn a_kernel_name_with_an_interior_nul_is_rejected() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let launcher = KernelLauncher::new(&stream, NOOP_PTX).unwrap();

    let kernel = Kernel::new("no\0op", config());
    match unsafe { launcher.launch(kernel) } {
        Err(HostError::NulError(_, "kernel function name")) => {}
        Err(other) => panic!("expected a NUL error for the kernel name, got {other:?}"),
        Ok(()) => panic!("a kernel name with an interior NUL unexpectedly launched"),
    }
}

#[test]
fn a_missing_symbol_in_valid_ptx_is_recoverable() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let launcher = KernelLauncher::new(&stream, NOOP_PTX).unwrap();

    let kernel = Kernel::new("not_in_this_module", config());
    match unsafe { launcher.launch(kernel) } {
        Err(HostError::Driver(_, "load function")) => {}
        Err(other) => panic!("expected a function loading error, got {other:?}"),
        Ok(()) => panic!("a symbol absent from the module unexpectedly launched"),
    }

    let kernel = Kernel::new("noop", config());
    unsafe { launcher.launch(kernel) }.unwrap();
    stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires a second CUDA device"]
fn launching_arguments_from_another_context_is_rejected() {
    let Some(other_ctx) = second_device() else {
        return;
    };
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let other_stream = other_ctx.new_stream().unwrap();

    let buffer = GpuBuffer::from_host(&[0i32; 4], &other_stream).unwrap();
    let launcher = KernelLauncher::new(&stream, NOOP_PTX).unwrap();

    let mut kernel = Kernel::new("noop", config());
    kernel.push_arg_shrd(&buffer).unwrap();

    match unsafe { launcher.launch(kernel) } {
        Err(HostError::ContextMismatch(_)) => {}
        Err(other) => panic!("expected a context mismatch, got {other:?}"),
        Ok(()) => panic!("arguments from another context unexpectedly launched"),
    }
}

#[test]
#[ignore = "requires a second CUDA device"]
fn mixing_buffers_from_two_contexts_in_one_kernel_is_rejected() {
    let Some(other_ctx) = second_device() else {
        return;
    };
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let other_stream = other_ctx.new_stream().unwrap();

    let first = GpuBuffer::from_host(&[0i32; 4], &stream).unwrap();
    let second = GpuBuffer::from_host(&[0i32; 4], &other_stream).unwrap();

    let mut kernel = Kernel::new("noop", config());
    kernel.push_arg_shrd(&first).unwrap();

    match kernel.push_arg_shrd(&second) {
        Err(HostError::ContextMismatch(_)) => {}
        Err(other) => panic!("expected a context mismatch, got {other:?}"),
        Ok(()) => panic!("a buffer from another context was unexpectedly accepted"),
    }
}
