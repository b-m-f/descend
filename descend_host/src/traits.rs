/// A marker trait indicating that a type is a valid element type within the Descend language,
/// and can be safely transferred bitwise across the host-device boundary.
///
/// # Safety
/// The implementor must ensure that the Rust type has an identical binary representation,
/// size, and alignment to its corresponding type in the Descend language and the generated
/// CUDA C++ ABI. It must not contain host-side pointers, references, or padding bytes
/// that violate Descend's memory model.
pub unsafe trait DescendType: Copy {}

unsafe impl DescendType for i32 {}
unsafe impl DescendType for u32 {}
unsafe impl DescendType for i64 {}
unsafe impl DescendType for u64 {}
unsafe impl DescendType for f32 {}
unsafe impl DescendType for f64 {}
unsafe impl DescendType for u8 {}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct DescendTypeWrapper<T>(pub T);

unsafe impl<T: DescendType> cuda_core::simt::DeviceCopy for DescendTypeWrapper<T> {}
unsafe impl DescendType for bool {}
