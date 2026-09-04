#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_erf(scalar_t x);

template <>
__device__ inline half gelu_erf<half>(half x) {
  return x * __float2half(normcdff(__half2float(x)));
}

template <>
__device__ inline __nv_bfloat16 gelu_erf<__nv_bfloat16>(__nv_bfloat16 x) {
  return x * __float2bfloat16(normcdff(__bfloat162float(x)));
}

template <>
__device__ inline float gelu_erf<float>(float x) {
  return x * normcdff(x);
}

template <typename scalar_t>
__global__ void gelu_erf_gate_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t rows,
    int32_t intermediate_size) {
  const int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= intermediate_size) {
    return;
  }
  for (int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
       row < rows;
       row += static_cast<int64_t>(gridDim.y) * blockDim.y) {
    const int64_t row_offset = row * static_cast<int64_t>(intermediate_size) * 2;
    const scalar_t gate = input[row_offset + col];
    const scalar_t up = input[row_offset + intermediate_size + col];
    output[row * static_cast<int64_t>(intermediate_size) + col] =
        gelu_erf(gate) * up;
  }
}

__global__ void init_gelu_erf_bf16_lut_kernel(
    __nv_bfloat16* __restrict__ lut) {
  const uint32_t bits = blockIdx.x * blockDim.x + threadIdx.x;
  if (bits < (1u << 16)) {
    const __nv_bfloat16 value = __ushort_as_bfloat16(
        static_cast<unsigned short>(bits));
    lut[bits] = gelu_erf<__nv_bfloat16>(value);
  }
}

__global__ void gelu_erf_gate_bf16_lut_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ lut,
    int64_t rows,
    int32_t intermediate_size) {
  const int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= intermediate_size) {
    return;
  }
  for (int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
       row < rows;
       row += static_cast<int64_t>(gridDim.y) * blockDim.y) {
    const int64_t row_offset = row * static_cast<int64_t>(intermediate_size) * 2;
    const __nv_bfloat16 gate = input[row_offset + col];
    const __nv_bfloat16 up = input[row_offset + intermediate_size + col];
    output[row * static_cast<int64_t>(intermediate_size) + col] =
        lut[__bfloat16_as_ushort(gate)] * up;
  }
}

__global__ void gelu_erf_gate_bf16_lut_vec2_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ lut,
    int64_t rows,
    int32_t intermediate_size) {
  const int32_t vec_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t vec_intermediate_size = intermediate_size / 2;
  if (vec_col >= vec_intermediate_size) {
    return;
  }
  for (int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
       row < rows;
       row += static_cast<int64_t>(gridDim.y) * blockDim.y) {
    const int64_t row_offset =
        row * static_cast<int64_t>(vec_intermediate_size) * 2;
    const __nv_bfloat162 gate =
        reinterpret_cast<const __nv_bfloat162*>(input)[row_offset + vec_col];
    const __nv_bfloat162 up =
        reinterpret_cast<const __nv_bfloat162*>(input)[
            row_offset + vec_intermediate_size + vec_col];
    const __nv_bfloat162 activated = __halves2bfloat162(
        lut[__bfloat16_as_ushort(__low2bfloat16(gate))],
        lut[__bfloat16_as_ushort(__high2bfloat16(gate))]);
    reinterpret_cast<__nv_bfloat162*>(output)[
        row * static_cast<int64_t>(vec_intermediate_size) + vec_col] =
        __hmul2_rn(activated, up);
  }
}

#define CALL_GELU_ERF_GATE(T)                                                 \
  gelu_erf_gate_kernel<T><<<grid, block, 0, stream>>>(                        \
      reinterpret_cast<const T*>(input),                                      \
      reinterpret_cast<T*>(output),                                           \
      rows,                                                                   \
      intermediate_size)

extern "C" int gelu_erf_gate(
    const void* input,
    void* output,
    int64_t rows,
    int32_t intermediate_size,
    uint32_t dtype,
    cudaStream_t stream) {
  if (rows <= 0 || intermediate_size <= 0) {
    return 0;
  }

  constexpr int threads_x = 32;
  constexpr int threads_y = 8;
  const int64_t row_blocks = (rows - 1) / threads_y + 1;
  dim3 grid(
      static_cast<unsigned int>(
          (static_cast<int64_t>(intermediate_size) + threads_x - 1) /
          threads_x),
      static_cast<unsigned int>(row_blocks < 65535 ? row_blocks : 65535));
  dim3 block(threads_x, threads_y);

  if (dtype == 0) {
    CALL_GELU_ERF_GATE(half);
  } else if (dtype == 1) {
    CALL_GELU_ERF_GATE(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_GELU_ERF_GATE(float);
  } else {
    return 1;
  }

  return static_cast<int>(cudaGetLastError());
}

extern "C" int init_gelu_erf_bf16_lut(
    void* lut,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int entries = 1 << 16;
  init_gelu_erf_bf16_lut_kernel<<<entries / threads, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(lut));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int gelu_erf_gate_bf16_lut(
    const void* input,
    void* output,
    const void* lut,
    int64_t rows,
    int32_t intermediate_size,
    cudaStream_t stream) {
  if (rows <= 0 || intermediate_size <= 0) {
    return 0;
  }

  constexpr int threads_x = 32;
  constexpr int threads_y = 8;
  const int64_t row_blocks = (rows - 1) / threads_y + 1;
  dim3 block(threads_x, threads_y);
  const bool use_vec2 =
      (reinterpret_cast<uintptr_t>(input) & 0x3u) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 0x3u) == 0 &&
      intermediate_size % 2 == 0;
  if (use_vec2) {
    const int32_t vec_intermediate_size = intermediate_size / 2;
    dim3 grid(
        static_cast<unsigned int>(
            (static_cast<int64_t>(vec_intermediate_size) + threads_x - 1) /
            threads_x),
        static_cast<unsigned int>(row_blocks < 65535 ? row_blocks : 65535));
    gelu_erf_gate_bf16_lut_vec2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_bfloat16*>(output),
        reinterpret_cast<const __nv_bfloat16*>(lut),
        rows,
        intermediate_size);
  } else {
    dim3 grid(
        static_cast<unsigned int>(
            (static_cast<int64_t>(intermediate_size) + threads_x - 1) /
            threads_x),
        static_cast<unsigned int>(row_blocks < 65535 ? row_blocks : 65535));
    gelu_erf_gate_bf16_lut_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_bfloat16*>(output),
        reinterpret_cast<const __nv_bfloat16*>(lut),
        rows,
        intermediate_size);
  }
  return static_cast<int>(cudaGetLastError());
}
