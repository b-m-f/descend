/// A value type that may be copied between Rust host memory and a Descend kernel.
///
/// # Safety
/// Implementors must satisfy all of the following:
///
/// - values can be duplicated by copying their bytes and have no ownership or
///   drop obligations;
/// - the all-zero bit pattern is valid;
/// - size, alignment, field layout, padding, and valid value encodings agree
///   with the corresponding Descend type and generated CUDA C++ ABI;
/// - values contain no host pointers or references whose validity depends on
///   host-side ownership; and
/// - every value written by a kernel and subsequently read back is a valid Rust
///   value of this type.
///
/// Implementing this trait establishes a representation contract for a type. It
/// does not prove that a particular kernel parameter has that type. In
/// particular, unsafe launch callers must ensure that kernels writing `bool`
/// produce only Rust's valid `0` and `1` encodings.
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

unsafe impl<T: DescendType> crate::backend::DeviceCopy for DescendTypeWrapper<T> {}
unsafe impl DescendType for bool {}
