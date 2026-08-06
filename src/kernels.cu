#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../tester/utils.h"

template <typename T>
__device__ __host__ inline float to_float(T value) { return static_cast<float>(value); }

#if defined(PLATFORM_ILUVATAR)
template <>
__device__ __host__ inline float to_float<half>(half value) {
  const unsigned short bits = *reinterpret_cast<unsigned short*>(&value);
  const unsigned sign = static_cast<unsigned>(bits & 0x8000u) << 16;
  unsigned exponent = (bits >> 10) & 0x1fu;
  unsigned mantissa = bits & 0x03ffu;
  unsigned result;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      result = sign | (exponent << 23) | ((mantissa & 0x03ffu) << 13);
    }
  } else if (exponent == 31) {
    result = sign | 0x7f800000u | (mantissa << 13);
  } else {
    result = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  return *reinterpret_cast<float*>(&result);
}
#else
template <>
__device__ __host__ inline float to_float<half>(half value) { return __half2float(value); }
#endif

template <typename T>
__device__ __host__ inline T from_float(float value) { return static_cast<T>(value); }

#if defined(PLATFORM_ILUVATAR)
template <>
__device__ __host__ inline half from_float<half>(float value) {
  const unsigned bits = *reinterpret_cast<unsigned*>(&value);
  const unsigned sign = (bits >> 16) & 0x8000u;
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 112;
  const unsigned mantissa = bits & 0x007fffffu;
  unsigned short result;
  if (exponent <= 0) {
    if (exponent < -10) {
      result = static_cast<unsigned short>(sign);
    } else {
      const unsigned rounded = (mantissa | 0x00800000u) >> (1 - exponent);
      result = static_cast<unsigned short>(sign | ((rounded + 0x00001000u) >> 13));
    }
  } else if (exponent >= 31) {
    result = static_cast<unsigned short>(sign | 0x7c00u);
  } else {
    unsigned rounded = mantissa + 0x00001000u;
    if (rounded & 0x00800000u) {
      rounded = 0;
      ++exponent;
    }
    result = static_cast<unsigned short>(
        sign | (exponent >= 31 ? 0x7c00u : (static_cast<unsigned>(exponent) << 10) | (rounded >> 13)));
  }
  half output;
  *reinterpret_cast<unsigned short*>(&output) = result;
  return output;
}
#else
template <>
__device__ __host__ inline half from_float<half>(float value) { return __float2half(value); }
#endif

#if defined(PLATFORM_ILUVATAR)
#define RMS_NORM_KERNEL iluvatarRmsNormKernel
#else
#define RMS_NORM_KERNEL rmsNormKernel
#endif

template <typename T>
__global__ void RMS_NORM_KERNEL(const T* input, const T* weight, T* output,
                              size_t rows, size_t hidden_dim, float eps) {
  extern __shared__ float sums[];
  const size_t row = blockIdx.x;
  const unsigned tid = threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  const size_t base = row * hidden_dim;
  for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
    const float x = to_float(input[base + col]);
    sum += x * x;
  }
  sums[tid] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sums[0] / static_cast<float>(hidden_dim) + eps);
  for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
    output[base + col] = from_float<T>(to_float(input[base + col]) * inv_rms * to_float(weight[col]));
  }
}

#if defined(PLATFORM_ILUVATAR)
#define FLASH_ATTENTION_KERNEL iluvatarFlashAttentionKernel
#else
#define FLASH_ATTENTION_KERNEL flashAttentionKernel
#endif

template <typename T>
__global__ void FLASH_ATTENTION_KERNEL(const T* q, const T* k, const T* v, T* o,
                                     int batch_size, int target_len, int source_len,
                                     int query_heads, int kv_heads, int head_dim,
                                     bool is_causal) {
  const int linear = blockIdx.x;
  const int dim = threadIdx.x;
  if (dim >= head_dim) return;
  const int q_head = linear % query_heads;
  const int target = (linear / query_heads) % target_len;
  const int batch = linear / (query_heads * target_len);
  const int kv_head = q_head / (query_heads / kv_heads);
  const int q_base = ((batch * target_len + target) * query_heads + q_head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float max_score = -1.0e30f;
  for (int source = 0; source < source_len; ++source) {
    if (is_causal && source > target) continue;
    const int kv_base = ((batch * source_len + source) * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d)
      dot += to_float(q[q_base + d]) * to_float(k[kv_base + d]);
    max_score = fmaxf(max_score, dot * scale);
  }
  float denominator = 0.0f;
  for (int source = 0; source < source_len; ++source) {
    if (is_causal && source > target) continue;
    const int kv_base = ((batch * source_len + source) * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d)
      dot += to_float(q[q_base + d]) * to_float(k[kv_base + d]);
    denominator += expf(dot * scale - max_score);
  }
  float result = 0.0f;
  for (int source = 0; source < source_len; ++source) {
    if (is_causal && source > target) continue;
    const int kv_base = ((batch * source_len + source) * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d)
      dot += to_float(q[q_base + d]) * to_float(k[kv_base + d]);
    result += expf(dot * scale - max_score) / denominator * to_float(v[kv_base + dim]);
  }
  o[q_base + dim] = from_float<T>(result);
}

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  if (rows == 0 || hidden_dim == 0) return;
  T *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_weight, h_weight.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_output, rows * hidden_dim * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(T), cudaMemcpyHostToDevice));
  RMS_NORM_KERNEL<T><<<static_cast<unsigned>(rows), 256, 256 * sizeof(float)>>>(
      d_input, d_weight, d_output, rows, hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, rows * hidden_dim * sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_output));
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      query_heads % kv_heads != 0 || head_dim > 1024) return;
  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, h_o.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(T), cudaMemcpyHostToDevice));
  const int output_rows = batch_size * target_seq_len * query_heads;
  FLASH_ATTENTION_KERNEL<T><<<output_rows, head_dim>>>(
      d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
