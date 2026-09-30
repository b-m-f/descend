use descend_host::error::HostError;
use descend_host::{Context, GpuBuffer, Kernel, KernelLaunchConfig, KernelLauncher};

const FAULTING_PTX: &str = ".version 7.0
.target sm_52
.address_size 64

.visible .entry fault()
{
\t.reg .u64 \t%rd<2>;
\t.reg .u32 \t%r<2>;
\tmov.u64 \t%rd1, 0;
\tmov.u32 \t%r1, 1;
\tst.global.u32 \t[%rd1], %r1;
\tret;
}
";

#[test]
#[ignore = "deliberately faults the device and poisons its context; run explicitly"]
fn a_faulting_kernel_surfaces_through_check_deferred_errors() {
    let ctx = Context::new(0).unwrap();

    let launch_error = {
        let stream = ctx.new_stream().unwrap();
        let launcher = KernelLauncher::new(&stream, FAULTING_PTX).unwrap();
        let _buffer = GpuBuffer::from_host(&[0i32; 16], &stream).unwrap();

        let kernel = Kernel::new(
            "fault",
            KernelLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
        );
        unsafe { launcher.launch(kernel) }.unwrap();

        stream.synchronize().err()
    };

    assert!(
        launch_error.is_some(),
        "synchronizing after an illegal access must report the fault"
    );

    match ctx.check_deferred_errors() {
        Err(HostError::Driver(_, "error from context")) => {}
        Err(other) => panic!("expected a deferred driver error, got {other:?}"),
        Ok(()) => panic!("failing destructors on a poisoned context recorded nothing"),
    }

    ctx.check_deferred_errors()
        .expect("reading the deferred errors must clear them");
}
