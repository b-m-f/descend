#include "descend.cuh"
/*
function declarations
*/
extern "C" __global__ auto gpu_vec_add(descend::i32 *const a_array,
                            const descend::i32 *const b_array) -> void;
__host__ auto dummy_main(descend::i32 *const arg0, const descend::i32 *const arg1)
    -> void;
/*
function definitions
*/
extern "C" __global__ auto gpu_vec_add(descend::i32 *const a_array,
                            const descend::i32 *const b_array) -> void {
  extern __shared__ descend::byte $buffer[];
  {
    {
      const auto a_ref =
          (&a_array[(((blockIdx.x - 0) * 256) + (threadIdx.x - 0))]);
      (*a_ref) =
          ((*a_ref) + b_array[(((blockIdx.x - 0) * 256) + (threadIdx.x - 0))]);
    }
  }
}

__host__ auto dummy_main(descend::i32 *const arg0, const descend::i32 *const arg1)
    -> void {
  auto gpu = descend::gpu_device(0);
  auto arg0_gpu = descend::gpu_alloc_copy<descend::array<descend::i32, 16384>>(
      (&gpu), (&(*arg0)));
  auto arg1_gpu = descend::gpu_alloc_copy<descend::array<descend::i32, 16384>>(
      (&gpu), (&(*arg1)));
  gpu_vec_add<<<dim3(64, 1, 1), dim3(256, 1, 1), 0>>>((&arg0_gpu), (&arg1_gpu));
}
