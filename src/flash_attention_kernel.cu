#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr float kEps = 1e-20f;

struct FlashAttentionDeviceCache {
  float* d_q = nullptr;
  float* d_k_t = nullptr;
  float* d_v = nullptr;
  float* d_out = nullptr;
  float* d_lse = nullptr;
  size_t qkv_capacity_bytes = 0;
  size_t lse_capacity_bytes = 0;
};

FlashAttentionDeviceCache g_flash_cache;

void free_qkv_cache() {
  if (g_flash_cache.d_q) cudaFree(g_flash_cache.d_q);
  if (g_flash_cache.d_k_t) cudaFree(g_flash_cache.d_k_t);
  if (g_flash_cache.d_v) cudaFree(g_flash_cache.d_v);
  if (g_flash_cache.d_out) cudaFree(g_flash_cache.d_out);
  g_flash_cache.d_q = nullptr;
  g_flash_cache.d_k_t = nullptr;
  g_flash_cache.d_v = nullptr;
  g_flash_cache.d_out = nullptr;
  g_flash_cache.qkv_capacity_bytes = 0;
}

void free_lse_cache() {
  if (g_flash_cache.d_lse) cudaFree(g_flash_cache.d_lse);
  g_flash_cache.d_lse = nullptr;
  g_flash_cache.lse_capacity_bytes = 0;
}

void ensure_flash_cache_capacity(size_t qkv_bytes, size_t lse_bytes) {
  if (qkv_bytes > g_flash_cache.qkv_capacity_bytes) {
    free_qkv_cache();
    cudaMalloc((void**)&g_flash_cache.d_q, qkv_bytes);
    cudaMalloc((void**)&g_flash_cache.d_k_t, qkv_bytes);
    cudaMalloc((void**)&g_flash_cache.d_v, qkv_bytes);
    cudaMalloc((void**)&g_flash_cache.d_out, qkv_bytes);
    g_flash_cache.qkv_capacity_bytes = qkv_bytes;
  }

  if (lse_bytes > g_flash_cache.lse_capacity_bytes) {
    free_lse_cache();
    cudaMalloc((void**)&g_flash_cache.d_lse, lse_bytes);
    g_flash_cache.lse_capacity_bytes = lse_bytes;
  }
}

__global__ void flash_attention_fw_kernel(
    const float* q,
    const float* k_t,
    const float* v,
    float* out,
    float* lse,
    int batch_size,
    int nhead,
    int seq_len,
    int head_dim,
    bool causal) {
  int q_idx = blockIdx.x;
  int h_idx = blockIdx.y;
  int b_idx = blockIdx.z;
  int tid = threadIdx.x;

  if (q_idx >= seq_len || h_idx >= nhead || b_idx >= batch_size) {
    return;
  }

  extern __shared__ float shared_mem[];
  float* s_q = shared_mem;
  float* s_o = s_q + head_dim;
  float* s_reduce = s_o + head_dim;

  __shared__ float s_m;
  __shared__ float s_l;
  __shared__ float s_coeff_old;
  __shared__ float s_coeff_new;

  if (tid == 0) {
    s_m = -INFINITY;
    s_l = 0.0f;
  }

  int bh_base_qv = (b_idx * nhead + h_idx) * seq_len;
  int bh_base_kt = (b_idx * nhead + h_idx) * head_dim;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    int q_pos = (bh_base_qv + q_idx) * head_dim + d;
    s_q[d] = q[q_pos];
    s_o[d] = 0.0f;
  }
  __syncthreads();

  const float inv_sqrt_d = rsqrtf((float)head_dim);

  for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
    if (causal && k_idx > q_idx) {
      continue;
    }

    float partial = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
      int k_pos = (bh_base_kt + d) * seq_len + k_idx;
      partial += s_q[d] * k_t[k_pos];
    }

    s_reduce[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_reduce[tid] += s_reduce[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      float score = s_reduce[0] * inv_sqrt_d;
      float m_new = fmaxf(s_m, score);
      float alpha = expf(s_m - m_new);
      float beta = expf(score - m_new);
      float l_new = alpha * s_l + beta;

      if (l_new <= kEps) {
        s_coeff_old = 0.0f;
        s_coeff_new = 0.0f;
      } else {
        s_coeff_old = (alpha * s_l) / l_new;
        s_coeff_new = beta / l_new;
      }

      s_m = m_new;
      s_l = l_new;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
      int v_pos = (bh_base_qv + k_idx) * head_dim + d;
      float v_val = v[v_pos];
      s_o[d] = s_coeff_old * s_o[d] + s_coeff_new * v_val;
    }
    __syncthreads();
  }

  for (int d = tid; d < head_dim; d += blockDim.x) {
    int out_pos = (bh_base_qv + q_idx) * head_dim + d;
    out[out_pos] = s_o[d];
  }

  if (tid == 0) {
    lse[(b_idx * nhead + h_idx) * seq_len + q_idx] = s_m + logf(fmaxf(s_l, kEps));
  }
}

}  // namespace

extern "C" {

void launch_flash_attention_fw(
    float* out,
    float* lse,
    const float* q,
    const float* k_t,
    const float* v,
    int batch_size,
    int nhead,
    int seq_len,
    int head_dim,
    bool causal,
    cudaStream_t stream) {
  size_t qkv_elems =
      static_cast<size_t>(batch_size) * static_cast<size_t>(nhead) *
      static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim);
  size_t lse_elems =
      static_cast<size_t>(batch_size) * static_cast<size_t>(nhead) *
      static_cast<size_t>(seq_len);

  size_t qkv_bytes = qkv_elems * sizeof(float);
  size_t lse_bytes = lse_elems * sizeof(float);

  ensure_flash_cache_capacity(qkv_bytes, lse_bytes);

  cudaMemcpyAsync(g_flash_cache.d_q, q, qkv_bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(g_flash_cache.d_k_t, k_t, qkv_bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(g_flash_cache.d_v, v, qkv_bytes, cudaMemcpyHostToDevice, stream);

  int threads = 32;
  if (head_dim > 32 && head_dim <= 64) {
    threads = 64;
  } else if (head_dim > 64 && head_dim <= 128) {
    threads = 128;
  } else if (head_dim > 128) {
    threads = 256;
  }

  dim3 grid(seq_len, nhead, batch_size);
  size_t shared_mem_bytes = static_cast<size_t>(2 * head_dim + threads) * sizeof(float);

  flash_attention_fw_kernel<<<grid, threads, shared_mem_bytes, stream>>>(
    g_flash_cache.d_q,
    g_flash_cache.d_k_t,
    g_flash_cache.d_v,
    g_flash_cache.d_out,
    g_flash_cache.d_lse,
      batch_size,
      nhead,
      seq_len,
      head_dim,
      causal);

  cudaMemcpyAsync(out, g_flash_cache.d_out, qkv_bytes, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(lse, g_flash_cache.d_lse, lse_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_flash_attention_fw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

}

void release_flash_attention_buffers() {
  free_qkv_cache();
  free_lse_cache();
}

}