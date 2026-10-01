use descend_host::{Context, GpuBuffer, Kernel, KernelLaunchConfig, KernelLauncher};
use descend_macros::{descend, descend_fn};

const NOOP_PTX: &str =
    ".version 7.0\n.target sm_52\n.address_size 64\n\n.visible .entry noop()\n{\n\tret;\n}\n";

fn config() -> KernelLaunchConfig {
    KernelLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

descend! {
    fn gpu_shared_bump<r1: prv, r2: prv>(
        vec: &r1 uniq gpu.global [i32; 64],
        [grid.forall(X)] tmp: &r2 uniq gpu.shared [i32; 64]
    ) -[grid: gpu.grid<X<1>, X<64>>]-> () {
        let view = &uniq (*vec).to_view.grp::<64>;
        sched block in grid {
            let ib = &uniq (*view)[[block]];
            let tmp_view = &uniq (*tmp).to_view;
            sched thread in block {
                let v = &uniq (*ib)[[thread]];
                let t = &uniq (*tmp_view)[[thread]];
                *t = *v + 1;
                *v = *t
            }
        }
    }
}

#[test]
fn a_stream_outlives_the_context_handle_it_came_from() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    drop(ctx);

    let buffer = GpuBuffer::from_host(&[5i32; 32], &stream).unwrap();
    assert_eq!(*buffer.to_host(&stream).unwrap(), [5i32; 32]);
    stream.synchronize().unwrap();
}

#[test]
fn a_launcher_keeps_its_module_after_the_ptx_text_is_dropped() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let ptx = String::from(NOOP_PTX);
    let launcher = KernelLauncher::new(&stream, &ptx).unwrap();
    drop(ptx);

    for _ in 0..2 {
        let kernel = Kernel::new("noop", config());
        unsafe { launcher.launch(kernel) }.unwrap();
    }

    stream.synchronize().unwrap();
    ctx.check_deferred_errors().unwrap();
}

#[test]
fn a_launch_with_non_zero_shared_memory_completes() {
    let gpu_shared_bump = descend_fn!(gpu_shared_bump);

    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut vec_gpu = GpuBuffer::from_host(&[41i32; 64], &stream).unwrap();

    gpu_shared_bump(&stream, &mut vec_gpu).unwrap();

    ctx.synchronize().unwrap();
    ctx.check_deferred_errors().unwrap();
    assert_eq!(*vec_gpu.to_host(&stream).unwrap(), [42i32; 64]);
}
