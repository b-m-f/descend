use descend_host::traits::DescendType;

#[derive(Clone, Copy)]
#[repr(C)]
struct OverflowType(i32, i32, i32, i32); // 16 bytes

unsafe impl DescendType for OverflowType {}

fn main() {
    // 16 * (usize::MAX / 8) exceeds the address space. With the buffer size in
    // the contents type, an oversized buffer is an oversized *type*: rustc
    // rejects it when it computes its layout, so the runtime overflow check
    // this replaces can no longer be reached. The `const` forces that layout
    // during checking; in a full build the same error is reported at codegen,
    // when `GpuBuffer::<[OverflowType; usize::MAX / 8]>::validate_buffer_params`
    // is instantiated.
    const CONTENTS_SIZE: usize = std::mem::size_of::<[OverflowType; usize::MAX / 8]>();
    let _ = CONTENTS_SIZE;
}
