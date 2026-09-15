use descend_host::{GpuBuffer, Stream};

fn host_access_while_borrowed(host: &mut [i32; 4], stream: &Stream) {
    let buffer = GpuBuffer::from_host_mut(host, stream).unwrap();
    host[0] = 7;
    let _still_alive = buffer.gpu_buffer();
}

fn main() {}
