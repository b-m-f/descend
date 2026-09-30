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
fn only_failing_destructors_reach_check_deferred_errors() {
    let ctx = Context::new(0).unwrap();

    let stream = ctx.new_stream().unwrap();
    let launcher = KernelLauncher::new(&stream, FAULTING_PTX).unwrap();
    let buffer = GpuBuffer::from_host(&[0i32; 16], &stream).unwrap();

    let kernel = Kernel::new(
        "fault",
        KernelLaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        },
    );
    unsafe { launcher.launch(kernel) }
        .expect("submission itself succeeds; the fault is not visible yet");

    assert!(
        stream.synchronize().is_err(),
        "a kernel fault surfaces as the Err of synchronize, not through the context"
    );

    ctx.check_deferred_errors()
        .expect("a kernel fault on its own records nothing on the context");

    drop(buffer);

    match ctx.check_deferred_errors() {
        Err(HostError::Driver(_, "error from context")) => {}
        Err(other) => panic!("expected a deferred driver error, got {other:?}"),
        Ok(()) => panic!("a destructor failing on a poisoned context recorded nothing"),
    }

    drop(launcher);
    drop(stream);
    ctx.check_deferred_errors()
        .expect_err("each failing destructor records again after a read cleared the state");
    ctx.check_deferred_errors()
        .expect("reading the deferred errors must clear them");
}
