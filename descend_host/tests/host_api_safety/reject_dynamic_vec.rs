use descend_host::{Context, GpuBuffer};

fn main() {
    let ctx = Context::new(0);
    let stream = ctx.new_stream();

    let vec: Vec<i32> = vec![1, 2, 3];
    // Should fail because Vec does not guarantee exact length N at compile time
    let _buf = GpuBuffer::from_host(&vec, &stream);
}
