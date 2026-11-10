use descend_host::traits::DescendType;
use descend_host::{GpuBuffer, Stream};

#[derive(Clone, Copy)]
#[repr(C)]
struct OverflowType(i32, i32, i32, i32);
unsafe impl DescendType for OverflowType {}

static CONTENTS: [OverflowType; usize::MAX / 8] = [OverflowType(0, 0, 0, 0); usize::MAX / 8];

fn upload_oversized(stream: &Stream) {
    let _ = GpuBuffer::from_host(&CONTENTS, stream);
}

fn main() {}
