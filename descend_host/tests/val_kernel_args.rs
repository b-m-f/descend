//! By-value kernel arguments generated from Descend definitions.

use descend_host::{Context, GpuBuffer};
use descend_macros::descend;

descend! {
    fn gpu_buffer_then_val<r: prv>(
        out: &r uniq gpu.global [u32; 1],
        value: u32
    ) -[grid: gpu.grid<X<1>, X<1>>]-> () {
        sched block in grid {
            sched thread in block {
                let out_ref = &uniq (*out).to_view.grp::<1>[[block]][[thread]];
                *out_ref = value
            }
        }
    }
}

descend! {
    fn gpu_val_then_buffer<r: prv>(
        value: f32,
        out: &r uniq gpu.global [f32; 1]
    ) -[grid: gpu.grid<X<1>, X<1>>]-> () {
        sched block in grid {
            sched thread in block {
                let out_ref = &uniq (*out).to_view.grp::<1>[[block]][[thread]];
                *out_ref = value
            }
        }
    }
}

descend! {
    fn gpu_array_by_val<r: prv>(
        out: &r uniq gpu.global [i32; 3],
        values: [i32; 3]
    ) -[grid: gpu.grid<X<1>, X<3>>]-> () {
        sched block in grid {
            sched thread in block {
                let out_ref = &uniq (*out).to_view.grp::<3>[[block]][[thread]];
                *out_ref = values.to_view[[thread]]
            }
        }
    }
}

#[test]
fn value_after_buffer_reaches_the_device() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut out = GpuBuffer::from_host(&[0u32; 1], &stream).unwrap();
    let value = 0xDEAD_BEEFu32;

    gpu_buffer_then_val(&stream, &mut out, value).unwrap();

    assert_eq!(*out.to_host(&stream).unwrap(), [value]);
}

#[test]
fn value_before_buffer_keeps_positional_order() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut out = GpuBuffer::from_host(&[0.0f32; 1], &stream).unwrap();

    gpu_val_then_buffer(&stream, 1.5, &mut out).unwrap();

    assert_eq!(*out.to_host(&stream).unwrap(), [1.5]);
}

#[test]
fn array_is_marshaled_by_value() {
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut out = GpuBuffer::from_host(&[0i32; 3], &stream).unwrap();
    let values = [11i32, 22, 33];

    gpu_array_by_val(&stream, &mut out, values).unwrap();

    assert_eq!(*out.to_host(&stream).unwrap(), values);
}
