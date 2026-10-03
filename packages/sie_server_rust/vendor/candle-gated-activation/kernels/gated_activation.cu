#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename scalar_t>
__device__ inline float to_float(scalar_t value);

template <>
__device__ inline float to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__device__ inline float to_float<float>(float value) {
  return value;
}

template <typename scalar_t>
__device__ inline scalar_t from_float(float value);

template <>
__device__ inline half from_float<half>(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ inline float from_float<float>(float value) {
  return value;
}

__device__ inline float gelu_tanh(float x) {
  constexpr float kAlpha = 0.7978845608028654f;
  constexpr float kBeta = 0.044715f;
  return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x * x * x)));
}

template <typename scalar_t>
__global__ void gelu_gate_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t rows,
    int32_t intermediate_size) {
  const int64_t total = rows * static_cast<int64_t>(intermediate_size);
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += blockDim.x * gridDim.x) {
    const int64_t row = idx / intermediate_size;
    const int32_t col = idx - row * intermediate_size;
    const int64_t row_offset = row * static_cast<int64_t>(intermediate_size) * 2;
    const float up = to_float(input[row_offset + col]);
    const float gate = to_float(input[row_offset + intermediate_size + col]);
    output[idx] = from_float<scalar_t>(gelu_tanh(gate) * up);
  }
}

#define CALL_GELU_GATE(T)                                                     \
  gelu_gate_kernel<T><<<grid, block, 0, stream>>>(                            \
      reinterpret_cast<const T*>(input),                                      \
      reinterpret_cast<T*>(output),                                           \
      rows,                                                                   \
      intermediate_size)

extern "C" int gelu_gate(
    const void* input,
    void* output,
    int64_t rows,
    int32_t intermediate_size,
    uint32_t dtype,
    cudaStream_t stream) {
  if (rows <= 0 || intermediate_size <= 0) {
    return 0;
  }

  const int threads = 256;
  const int64_t total = rows * static_cast<int64_t>(intermediate_size);
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  dim3 grid(blocks);
  dim3 block(threads);

  if (dtype == 0) {
    CALL_GELU_GATE(half);
  } else if (dtype == 1) {
    CALL_GELU_GATE(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_GELU_GATE(float);
  } else {
    return 1;
  }

  return static_cast<int>(cudaGetLastError());
}
