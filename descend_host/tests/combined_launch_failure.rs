use descend_host::error::HostError;
use descend_host::{Context, GpuBuffer, Kernel, KernelLaunchConfig, KernelLauncher};
use descend_macros::{descend, descend_fn};

mod common;

fn launch_configuration() -> KernelLaunchConfig {
    KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

descend! {
    fn gpu_increment<r: prv>(
        values: &r uniq gpu.global [i32; 64]
    ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
        let view = &uniq (*values).to_view.grp::<64>;
        sched block in grid {
            let block_view = &uniq (*view)[[block]];
            sched thread in block {
                let element = &uniq (*block_view)[[thread]];
                *element = *element + 1
            }
        }
    }
}

#[test]
#[ignore = "deliberately faults the device and poisons its context; run explicitly"]
fn a_launch_refused_while_a_fault_is_pending_reports_both_failures() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();

    let launcher = KernelLauncher::new(&stream, common::FAULTING_PTX).unwrap();
    unsafe { launcher.launch(Kernel::new("fault", launch_configuration())) }
        .expect("submitting the faulting kernel succeeds; the fault is not visible yet");

    let launch_error =
        unsafe { launcher.launch(Kernel::new("not_in_this_module", launch_configuration())) }
            .expect_err("the absent symbol must be refused");
    let synchronize_error = stream
        .synchronize()
        .expect_err("the pending device fault must surface at the synchronize");

    let combined = HostError::combine(launch_error, synchronize_error);
    let rendered = combined.to_string();
    match combined {
        HostError::LaunchAndSynchronize {
            launch,
            synchronize,
        } => {
            assert!(
                matches!(*launch, HostError::Driver(_, "load function")),
                "{rendered}"
            );
            assert!(
                matches!(*synchronize, HostError::Driver(_, "stream synchronize")),
                "{rendered}"
            );
        }
        other => panic!("expected both failures to be reported, got {other:?}"),
    }

    assert!(
        rendered.contains("named symbol not found"),
        "the launch half lost its driver error: {rendered}"
    );
    assert!(
        rendered.contains("illegal memory access"),
        "the synchronize half lost its driver error: {rendered}"
    );
    assert!(
        rendered.contains("refused before the kernel reached the device"),
        "{rendered}"
    );
}

#[test]
#[ignore = "deliberately faults the device and poisons its context; run explicitly"]
fn a_generated_call_after_a_device_fault_fails_before_it_can_launch() {
    let gpu_increment = descend_fn!(gpu_increment);

    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut values_gpu = GpuBuffer::from_host(&[0i32; 64], &stream).unwrap();

    let faulting_launcher = KernelLauncher::new(&stream, common::FAULTING_PTX).unwrap();
    unsafe { faulting_launcher.launch(Kernel::new("fault", launch_configuration())) }
        .expect("submitting the faulting kernel succeeds; the fault is not visible yet");
    assert!(
        stream.synchronize().is_err(),
        "the device fault did not surface at the synchronize"
    );

    match gpu_increment(&stream, &mut values_gpu) {
        Err(HostError::Driver(_, "load module")) => {}
        Err(other) => panic!(
            "a generated call reloads its module on every call, so a poisoned context is \
             expected to stop it there rather than let it reach a launch; got {other:?}"
        ),
        Ok(()) => panic!("a generated call on a poisoned context unexpectedly succeeded"),
    }
}
