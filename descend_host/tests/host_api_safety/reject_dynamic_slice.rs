use descend_host::{Context, GpuBuffer};

fn main() {
    let ctx = Context::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();

    let arr = [1, 2, 3];
    let slice: &[i32] = &arr;
    // Should fail because &[T] has no length N at compile time
    let _buf = GpuBuffer::from_host(slice, &stream);
}
