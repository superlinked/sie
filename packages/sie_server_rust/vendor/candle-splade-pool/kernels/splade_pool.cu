#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdint.h>

__global__ void splade_segmented_max_log1p_f16_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ offsets,
    half* __restrict__ output,
    int64_t total_tokens,
    int32_t vocab_size) {
  const int32_t batch = static_cast<int32_t>(blockIdx.y);
  const int32_t vocab =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vocab >= vocab_size) {
    return;
  }

  const uint32_t start = offsets[batch];
  const uint32_t end = offsets[batch + 1];
  if (start >= end || static_cast<int64_t>(end) > total_tokens ||
      (batch == 0 && start != 0) ||
      (batch + 1 == static_cast<int32_t>(gridDim.y) &&
       static_cast<int64_t>(end) != total_tokens)) {
    output[static_cast<int64_t>(batch) * vocab_size + vocab] =
        __float2half(CUDART_NAN_F);
    return;
  }

  // Keep the reduction in half precision, but evaluate true log1p in float
  // before rounding the pooled result to half. Performing 1 + x in half first
  // erases small positive SPLADE weights that PyTorch log1p preserves.
  // __hmax_nan also preserves the existing non-finite-output rejection path.
  half max_value = __float2half(-INFINITY);
  for (uint32_t token = start; token < end; ++token) {
    max_value = __hmax_nan(
        max_value,
        input[static_cast<int64_t>(token) * vocab_size + vocab]);
  }
  output[static_cast<int64_t>(batch) * vocab_size + vocab] =
      __float2half_rn(log1pf(__half2float(max_value)));
}

extern "C" int splade_segmented_max_log1p_f16(
    const void* input,
    const void* offsets,
    void* output,
    int64_t total_tokens,
    int32_t vocab_size,
    int32_t batch_size,
    cudaStream_t stream) {
  if (input == nullptr || offsets == nullptr || output == nullptr ||
      total_tokens <= 0 || vocab_size <= 0 || batch_size <= 0 ||
      batch_size > 65535) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  constexpr int32_t threads = 256;
  const uint32_t vocab_tiles =
      (static_cast<uint32_t>(vocab_size) + threads - 1) / threads;
  const dim3 grid(vocab_tiles, static_cast<uint32_t>(batch_size));
  const dim3 block(threads);
  splade_segmented_max_log1p_f16_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(input),
      reinterpret_cast<const uint32_t*>(offsets),
      reinterpret_cast<half*>(output),
      total_tokens,
      vocab_size);
  return static_cast<int>(cudaGetLastError());
}
