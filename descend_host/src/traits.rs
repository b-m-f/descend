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
///
/// The contract is closed under arrays: `[T; N]` is representable whenever `T`
/// is, so an array is both a buffer's contents type and a value a kernel may
/// take by value.
pub unsafe trait DescendType: Copy {}

unsafe impl DescendType for i32 {}
unsafe impl DescendType for u32 {}
unsafe impl DescendType for i64 {}
unsafe impl DescendType for u64 {}
unsafe impl DescendType for f32 {}
unsafe impl DescendType for f64 {}
unsafe impl DescendType for u8 {}

// SAFETY: `[T; N]` inherits every obligation above from `T`. It is `Copy`
// because `T` is, its all-zero bit pattern is the all-zero bit pattern of `N`
// values of `T` and therefore valid, and it holds no host pointers that `T`
// does not hold. Rust lays out `[T; N]` as `N` contiguous values of `T` with no
// padding between elements and with `align_of::<T>()` alignment, which is the
// layout of its CUDA counterpart `descend::array<T, N>`, declared as
// `struct array { T data_[n]; }` (`cuda-examples/descend.cuh:226`). `[T; 0]` is
// zero-sized and remains rejected wherever a value is required.
unsafe impl<T: DescendType, const N: usize> DescendType for [T; N] {}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct DescendTypeWrapper<T>(pub T);

unsafe impl<T: DescendType> crate::backend::DeviceCopy for DescendTypeWrapper<T> {}
unsafe impl DescendType for bool {}
