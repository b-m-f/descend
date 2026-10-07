//! The device limits a launch is checked against.
//!
//! Each constant mirrors a `CU_DEVICE_ATTRIBUTE_*` that the driver can report at run
//! time. They are written down here rather than queried, for two reasons:
//!
//! * Opening a CUDA context during macro expansion would make `cargo build` need a GPU,
//!   which rules out CI and cross-compilation. This crate shells out to `nvidia-smi` and
//!   loads NVRTC as a library, but it never talks to the driver.
//! * A queried answer describes the *build* machine. A launch legal there could be
//!   rejected for a different target, and one illegal there could be accepted and then
//!   fail at launch -- the second is the failure this module exists to prevent.
//!
//! So each value is the minimum guaranteed across every architecture Descend supports,
//! which makes the check conservative in the right direction: a launch these constants
//! accept runs on any of them. The trade is that a device with a more generous limit is
//! held to the portable one.
//!
//! Raising a limit is not a matter of editing a number here. The shared-memory ceiling
//! in particular is the *portable* default; `cuFuncSetAttribute` lifts it to 100 KiB and
//! beyond on recent architectures, but only for a kernel that opts in, and the generated
//! launch code does not opt in. Making that available is a deliberate feature, not a
//! larger constant.
//!
//! The same reasoning governs [`crate::host_ty::ScalarKind::shared_memory_bytes`], which
//! mirrors Descend's own `size_of_dty` rather than answering from Rust's `size_of`.

/// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK` (39).
///
/// The product of a block's three dimensions, not any one of them. 512 on Tesla
/// (compute 1.x); 1024 on every architecture from Fermi (compute 2.0, 2010) onward.
pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

/// `CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X` (40). Stable since Fermi.
pub const MAX_BLOCK_DIM_X: u32 = 1024;

/// `CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y` (41). Stable since Fermi.
pub const MAX_BLOCK_DIM_Y: u32 = 1024;

/// `CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z` (42). Stable since Fermi, and the tightest of
/// the three by a wide margin, so a three-dimensional block is easy to get wrong.
pub const MAX_BLOCK_DIM_Z: u32 = 64;

/// `CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X` (43).
///
/// 65535 on compute 1.x and 2.x; raised to 2^31-1 in Kepler (compute 3.0, 2012) and
/// unchanged since.
pub const MAX_GRID_DIM_X: u32 = 2_147_483_647;

/// `CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y` (44). Never raised, unlike the x axis.
pub const MAX_GRID_DIM_Y: u32 = 65535;

/// `CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z` (45). Never raised, unlike the x axis.
pub const MAX_GRID_DIM_Z: u32 = 65535;

/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` (8).
///
/// The dynamic shared memory a launch may request without opting in through
/// `cuFuncSetAttribute`. 48 KiB on every architecture; see the module documentation for
/// why the opt-in ceiling is not what is checked here. The count is checked against this
/// by [`crate::shared_memory::SharedMemoryBytes`].
pub const MAX_SHARED_MEMORY_BYTES: u32 = 49152;
