/// A host type whose byte representation may be transferred to and from the
/// corresponding Descend device type.
///
/// # Safety
///
/// Implementers must ensure that the host and device types agree in size,
/// alignment, layout, and value encoding. Values must not contain host pointers
/// or references whose validity depends on host ownership, and every value a
/// kernel can write and return to the host must be a valid Rust value of this
/// type.
pub unsafe trait DescendType: Copy {}

unsafe impl DescendType for i8 {}
unsafe impl DescendType for u8 {}
unsafe impl DescendType for i16 {}
unsafe impl DescendType for u16 {}
unsafe impl DescendType for i32 {}
unsafe impl DescendType for u32 {}
unsafe impl DescendType for i64 {}
unsafe impl DescendType for u64 {}
unsafe impl DescendType for f32 {}
unsafe impl DescendType for f64 {}
unsafe impl DescendType for bool {}
unsafe impl<T: DescendType, const N: usize> DescendType for [T; N] {}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct DescendTypeWrapper<T>(pub T);

unsafe impl<T: DescendType> crate::backend::DeviceCopy for DescendTypeWrapper<T> {}
