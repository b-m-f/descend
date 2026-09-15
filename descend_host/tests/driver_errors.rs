//! Real driver failures that can be provoked without stressing the device.

use descend_host::error::HostError;
use descend_host::{Context, KernelLauncher};

#[test]
fn invalid_device_ordinal_is_recoverable() {
    match Context::new(u32::MAX) {
        Err(HostError::Driver(_, "context creation")) => {}
        Err(other) => panic!("expected a context creation error, got {other:?}"),
        Ok(_) => panic!("device u32::MAX unexpectedly exists"),
    }
}

#[test]
fn invalid_ptx_is_recoverable() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    match KernelLauncher::new(&stream, "this is not a PTX module") {
        Err(HostError::Driver(_, "load module")) => {}
        Err(other) => panic!("expected a module loading error, got {other:?}"),
        Ok(_) => panic!("invalid PTX unexpectedly loaded"),
    }

    stream.synchronize().unwrap();
}
