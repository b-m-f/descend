#include "descend.cuh"
/*
function declarations
*/
extern "C" __global__ auto gpu_vec_add(descend::i32 *const a_array,
                            const descend::i32 *const b_array) -> void;
/*
function definitions
*/
extern "C" __global__ auto gpu_vec_add(descend::i32 *const a_array,
                            const descend::i32 *const b_array) -> void {
  extern __shared__ descend::byte _buffer[];
  {
    {
      const auto a_ref =
          (&a_array[(((blockIdx.x - 0) * 256) + (threadIdx.x - 0))]);
      (*a_ref) =
          ((*a_ref) + b_array[(((blockIdx.x - 0) * 256) + (threadIdx.x - 0))]);
    }
  }
}

