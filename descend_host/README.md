# descend_host

Host-side interface for launching Descend kernels from Rust: contexts, streams,
device buffers, and an asynchronous, `unsafe` low-level kernel launcher. Safe
functions generated from Descend kernels synchronize before returning. The CUDA
backend (`cuda-core`) is private and re-exported internally through the thin
facade in `src/backend.rs`.

## Running the tests

The tests need an NVIDIA driver, a GPU, and a CUDA toolkit: the procedural macro loads
NVRTC from it. `DESCEND_CUDA_PATH`, or `CUDA_PATH` when that is unset, has to name the
toolkit root -- the directory holding both `lib/libnvrtc.so` and `include/`. The target
architecture is probed from `nvidia-smi`; `DESCEND_NVRTC_FLAGS=-arch=compute_86`
overrides it.

```bash
DESCEND_CUDA_PATH=/usr/local/cuda cargo test --locked --offline -p descend_host
```

The compile-fail fixtures in `tests/host_api_safety/` run through `trybuild`.

### Multi-GPU tests

`tests/test_genuine_context_mismatch.rs` checks that a buffer allocated on one
device is rejected by a stream on another device. It needs at least two GPUs. On a
single-GPU machine it prints `Skipping genuine context mismatch test: device 1
not available.` and passes without exercising the mismatch path. Run it on
multi-GPU hardware with output enabled to confirm it did not skip:

```bash
cargo test --locked --offline -p descend_host --test test_genuine_context_mismatch -- --nocapture
```

### Device out-of-memory test

`gpu_buffer::tests::device_oom_on_upload_is_recoverable` requests a 20 GiB upload
and expects a recoverable allocation error. The host array is reserved as lazily
mapped zero pages. The test skips with a message if the device has at least
20 GiB free or the host cannot reserve the address space.
