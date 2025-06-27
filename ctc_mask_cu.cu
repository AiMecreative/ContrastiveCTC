#include "ATen/core/TensorBody.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define MAX_TARGET_LENGTH 256
#define BLANK 0

template <typename scalar_t> __device__ __forceinline__ void cuda_swap(scalar_t *a, scalar_t *b, const long size) {
  for (long i = threadIdx.x; i < size; i += blockDim.x) {
    scalar_t temp = a[i];
    a[i] = b[i];
    b[i] = temp;
  }
}

template <typename scalar_t> __device__ __forceinline__ scalar_t logsumexp(scalar_t a, scalar_t b) {
  if (a == -INFINITY && b == -INFINITY) {
    return -INFINITY;
  }
  scalar_t max_v = max(a, b);
  return max_v + log(exp(a - max_v) + exp(b - max_v));
}

template <typename scalar_t>
__device__ __forceinline__ void safe_reciprocal(scalar_t *mask, const long batch_idx, const long timesteps,
                                                const long num_classes) {
  for (long t = 0; t < timesteps; t += 1) {
    for (long s = threadIdx.x; s < num_classes; s += blockDim.x) {
      long idx = batch_idx * timesteps * num_classes + t * num_classes + s;
      mask[idx] = mask[idx] == 0.0f ? 0.0f : 1.0f / mask[idx];
    }
  }
}

template <typename scalar_t>
__global__ void ctc_mask_kernel(const scalar_t *log_probs, const long *targets, const long *input_lengths,
                                const long *target_lengths, scalar_t *losses, scalar_t *alpha_masks,
                                scalar_t *beta_masks, scalar_t *alphas, scalar_t *betas, const long batch_size,
                                const long timesteps, const long num_classes, const long max_target_length) {
  // Use one block to handle one batch
  const long batch_idx = blockIdx.x;
  const long target_length = target_lengths[batch_idx];
  const long input_length = input_lengths[batch_idx];

  // Initialize masks with zeros
  for (long t = 0; t < timesteps; t += 1) {
    for (long s = threadIdx.x; s < num_classes; s += blockDim.x) {
      alpha_masks[batch_idx * timesteps * num_classes + t * num_classes + s] = 0;
      beta_masks[batch_idx * timesteps * num_classes + t * num_classes + s] = 0;
    }
  }
  __syncthreads();

  // Copy the targets for this batch into shared mem, and extend it with blanks
  const long ext_target_length = 2 * target_length + 1;
  __shared__ long shared_ext_targets[2 * MAX_TARGET_LENGTH + 1];
  for (long i = threadIdx.x; i < ext_target_length; i += blockDim.x) {
    shared_ext_targets[i] = i % 2 == 0 ? BLANK : targets[batch_idx * max_target_length + i / 2];
  }
  __syncthreads();

  /* Forward calculation */
  __shared__ scalar_t alpha_prev[2 * MAX_TARGET_LENGTH + 1];
  __shared__ scalar_t alpha_curr[2 * MAX_TARGET_LENGTH + 1];

  if (threadIdx.x == 0) {
    alpha_prev[0] = log_probs[batch_idx * timesteps * num_classes + 0 * num_classes + shared_ext_targets[0]];
    alpha_prev[1] = log_probs[batch_idx * timesteps * num_classes + 0 * num_classes + shared_ext_targets[1]];
    for (long s = 2; s < ext_target_length; s += 1) {
      alpha_prev[s] = -INFINITY;
    }
    alpha_masks[batch_idx * timesteps * num_classes + shared_ext_targets[0]] += 1.0;
    alpha_masks[batch_idx * timesteps * num_classes + shared_ext_targets[1]] += 1.0;
  }
  __syncthreads();

  for (long t = 1; t < input_length; t += 1) {
    for (long s = threadIdx.x; s < ext_target_length; s += blockDim.x) {
      scalar_t prev_sum = alpha_prev[s];
      if (s > 0) {
        prev_sum = logsumexp(prev_sum, alpha_prev[s - 1]);
      }
      if (s > 1 && shared_ext_targets[s] != shared_ext_targets[s - 2]) {
        prev_sum = logsumexp(prev_sum, alpha_prev[s - 2]);
      }

      alpha_curr[s] =
          prev_sum + log_probs[batch_idx * timesteps * num_classes + t * num_classes + shared_ext_targets[s]];
    }
    __syncthreads();

    long s = threadIdx.x;
    if (s < ext_target_length && s % 2 != 0 && alpha_curr[s] != 0 && alpha_curr[s] != -INFINITY) {
      atomicAdd(&alpha_masks[batch_idx * timesteps * num_classes + t * num_classes + shared_ext_targets[s]], 1.0);
    }
    __syncthreads();

    if (s < ext_target_length) {
      alphas[batch_idx * timesteps * (2 * max_target_length + 1) + t * (2 * max_target_length + 1) + s] = alpha_curr[s];
    }
    __syncthreads();
    cuda_swap(alpha_prev, alpha_curr, 2 * MAX_TARGET_LENGTH + 1);
  }

  /* Backword calculation */
  __shared__ scalar_t beta_prev[2 * MAX_TARGET_LENGTH + 1];
  __shared__ scalar_t beta_curr[2 * MAX_TARGET_LENGTH + 1];

  if (threadIdx.x == 0) {
    beta_prev[ext_target_length - 1] = 1;
    beta_prev[ext_target_length - 2] = 1;
    for (long s = 0; s < ext_target_length - 2; s += 1) {
      beta_prev[s] = -INFINITY;
    }
    beta_masks[batch_idx * timesteps * num_classes + (input_length - 1) * num_classes +
               shared_ext_targets[ext_target_length - 1]] += 1.0;
    beta_masks[batch_idx * timesteps * num_classes + (input_length - 1) * num_classes +
               shared_ext_targets[ext_target_length - 2]] += 1.0;
  }
  __syncthreads();

  for (long t = input_length - 2; t >= 0; t -= 1) {
    for (long s = threadIdx.x; s < ext_target_length; s += blockDim.x) {
      scalar_t next_sum = beta_prev[s];
      if (s < ext_target_length - 1) {
        next_sum = logsumexp(next_sum, beta_prev[s + 1]);
      }
      if (s < ext_target_length - 2 && shared_ext_targets[s] != shared_ext_targets[s + 2]) {
        next_sum = logsumexp(next_sum, beta_prev[s + 2]);
      }

      beta_curr[s] =
          next_sum + log_probs[batch_idx * timesteps * num_classes + t * num_classes + shared_ext_targets[s]];
    }
    __syncthreads();

    long s = threadIdx.x;
    if (s < ext_target_length && s % 2 != 0 && beta_curr[s] != 0 && beta_curr[s] != -INFINITY) {
      atomicAdd(&beta_masks[batch_idx * timesteps * num_classes + t * num_classes + shared_ext_targets[s]], 1.0);
    }
    __syncthreads();

    if (s < ext_target_length) {
      betas[batch_idx * timesteps * (2 * max_target_length + 1) + t * (2 * max_target_length + 1) + s] = beta_curr[s];
    }
    __syncthreads();
    cuda_swap(beta_prev, beta_curr, 2 * MAX_TARGET_LENGTH + 1);
  }

  /* Post processing */
  safe_reciprocal(alpha_masks, batch_idx, timesteps, num_classes);
  safe_reciprocal(beta_masks, batch_idx, timesteps, num_classes);

  if (threadIdx.x == 0) {
    scalar_t log_prob = logsumexp(alpha_prev[ext_target_length - 1] + beta_prev[ext_target_length - 1],
                                  alpha_prev[ext_target_length - 2] + beta_prev[ext_target_length - 2]);
    losses[batch_idx] = -log_prob;
  }
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
ctc_decode_mask_forward_cu(torch::Tensor log_probs, torch::Tensor targets, torch::Tensor input_lengths,
                           torch::Tensor target_lengths) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(log_probs.get_device());
  c10::cuda::CUDAGuard device_guard(log_probs.device());
  const long batch_size = log_probs.size(0);
  const long timesteps = log_probs.size(1);
  const long num_classes = log_probs.size(2);
  const long max_target_length = targets.size(1);

  auto losses = torch::zeros_like(input_lengths, log_probs.options());
  auto masks = torch::zeros_like(log_probs, log_probs.options());
  auto alpha_masks = torch::zeros_like(log_probs, log_probs.options());
  auto beta_masks = torch::zeros_like(log_probs, log_probs.options());
  auto alphas = torch::zeros({batch_size, timesteps, (2 * max_target_length + 1)}, log_probs.options());
  auto betas = torch::zeros({batch_size, timesteps, (2 * max_target_length + 1)}, log_probs.options());
  const dim3 blocks(batch_size);
  const dim3 threads(BLOCK_SIZE);

  AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_mask_forward_cu", ([&] {
                               ctc_mask_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                   log_probs.data_ptr<scalar_t>(), targets.data_ptr<long>(),
                                   input_lengths.data_ptr<long>(), target_lengths.data_ptr<long>(),
                                   losses.data_ptr<scalar_t>(), alpha_masks.data_ptr<scalar_t>(),
                                   beta_masks.data_ptr<scalar_t>(), alphas.data_ptr<scalar_t>(),
                                   betas.data_ptr<scalar_t>(), batch_size, timesteps, num_classes, max_target_length);
                             }));
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }
  masks = at::sqrt(alpha_masks * beta_masks);
  return {losses, {alphas, betas}, {masks, alpha_masks, beta_masks}};
}