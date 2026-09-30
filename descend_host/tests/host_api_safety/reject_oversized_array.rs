use descend_host::traits::DescendType;

#[derive(Clone, Copy)]
#[repr(C)]
struct OverflowType(i32, i32, i32, i32);

unsafe impl DescendType for OverflowType {}

fn main() {
    const CONTENTS_SIZE: usize = std::mem::size_of::<[OverflowType; usize::MAX / 8]>();
    let _ = CONTENTS_SIZE;
}
