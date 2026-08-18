//! # Descend Host
//!
//! `descend_host` provides the stable host-side orchestration API for the Descend language.
//! It serves as the primary runtime interface for CPU-bound Rust programs to safely and seamlessly
//! launch, manage, and interact with compiled Descend GPU kernels.
//!
//! This crate completely abstracts away all underlying CUDA driver and runtime specifics.
//! The structures and methods provided here are designed as a stable interface, ensuring that
//! downstream applications do not break if the internal GPU backend or execution mechanisms evolve.
//!
//! ## Core Components
//!
//! - **`DescendContext`**: A stable, abstracted context for GPU execution management.
//! - **CPU Types (`CpuShrd`, `CpuUniq`)**: Safe host-side representations of data.
//! - **GPU Types (`GpuShrd`, `GpuUniq`)**: Abstract representations of device memory, carefully
//!   designed to prevent accidental host accesses.

use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy};
use std::sync::Arc;

/// An abstracted execution context for Descend kernels.
///
/// `DescendContext` hides the underlying hardware initialization and state, providing a stable
/// barrier between user-facing code and lower-level orchestration details.
#[derive(Clone)]
pub struct DescendContext(Arc<CudaContext>);

impl Default for DescendContext {
    fn default() -> Self {
        Self::new(0)
    }
}

impl DescendContext {
    /// Initializes a new Descend execution context for the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - The zero-based index of the GPU device to initialize.
    pub fn new(device: u32) -> Self {
        Self(CudaContext::new(device as usize).expect("Failed to create CUDA context"))
    }
}

/// A CPU-side representation of shared (read-only) data meant for GPU processing.
pub struct CpuShrd<T: Clone + Default + DeviceCopy, const N: usize>(pub Box<[T; N]>);

/// A CPU-side representation of unique (mutable) data meant for GPU processing.
pub struct CpuUniq<T: Clone + Default + DeviceCopy, const N: usize>(pub Box<[T; N]>);

/// A GPU-side representation of shared (read-only) data.
///
/// This stable type hides internal device buffers and prevents direct host access,
/// maintaining the safety guarantees of Descend's type system.
pub struct GpuShrd<T: DeviceCopy, const N: usize> {
    pub dev_buf: DeviceBuffer<T>,
}

/// A GPU-side representation of unique (mutable) data.
///
/// This stable type enforces exclusive access on the device and prevents direct,
/// unsafe host interactions.
pub struct GpuUniq<T: DeviceCopy, const N: usize> {
    pub dev_buf: DeviceBuffer<T>,
}

impl<T: Clone + Default + DeviceCopy, const N: usize> CpuShrd<T, N> {
    /// Wraps an existing boxed array into a new `CpuShrd`.
    pub fn new(arr: Box<[T; N]>) -> Self {
        CpuShrd(arr)
    }

    /// Safely transfers the shared CPU data to the GPU using the provided execution context.
    pub fn to_gpu(&self, context: &DescendContext) -> GpuShrd<T, N> {
        let stream = context.0.new_stream().expect("Failed to create stream");
        let dev_buf = DeviceBuffer::from_host(&stream, &*self.0).expect("Failed to copy HtoD");
        stream.synchronize().expect("Failed to synchronize stream");
        GpuShrd { dev_buf }
    }
}

impl<T: Clone + Default + DeviceCopy, const N: usize> CpuUniq<T, N> {
    /// Wraps an existing boxed array into a new `CpuUniq`.
    pub fn new(arr: Box<[T; N]>) -> Self {
        CpuUniq(arr)
    }

    /// Safely transfers the unique CPU data to the GPU using the provided execution context.
    pub fn to_gpu(&self, context: &DescendContext) -> GpuUniq<T, N> {
        let stream = context.0.new_stream().expect("Failed to create stream");
        let dev_buf = DeviceBuffer::from_host(&stream, &*self.0).expect("Failed to copy HtoD");
        stream.synchronize().expect("Failed to synchronize stream");
        GpuUniq { dev_buf }
    }
}

impl<T: DeviceCopy + Clone + Default, const N: usize> GpuUniq<T, N> {
    /// Safely transfers the unique GPU data back to the CPU after kernel execution.
    pub fn to_cpu(&self, context: &DescendContext) -> CpuUniq<T, N> {
        let stream = context.0.new_stream().expect("Failed to create stream");
        let mut vec = vec![T::default(); N];
        self.dev_buf
            .copy_to_host(&stream, &mut vec)
            .expect("Failed to copy DtoH");
        stream.synchronize().expect("Failed to synchronize stream");

        let boxed_slice = vec.into_boxed_slice();
        let boxed_array: Box<[T; N]> = boxed_slice
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to convert slice to array"));
        CpuUniq(boxed_array)
    }
}

// Deref is intentionally NOT implemented for GpuShrd and GpuUniq to prevent accidental host accesses.

// Implement Deref/DerefMut for CPU arrays for convenience
impl<T: Clone + Default + DeviceCopy, const N: usize> std::ops::Deref for CpuUniq<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T: Clone + Default + DeviceCopy, const N: usize> std::ops::DerefMut for CpuUniq<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<T: Clone + Default + DeviceCopy, const N: usize> std::ops::Deref for CpuShrd<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
