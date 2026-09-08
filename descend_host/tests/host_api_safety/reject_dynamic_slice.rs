use descend_host::{Context, GpuBuffer};

fn main() {
    let ctx = Context::new(0);
    let stream = ctx.new_stream();

    let arr = [1, 2, 3];
    let slice: &[i32] = &arr;
    // Should fail because &[T] has no length N at compile time
    let _buf = GpuBuffer::to_gpu(slice, &stream);
}
