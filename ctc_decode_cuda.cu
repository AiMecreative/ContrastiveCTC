#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <cmath>
#include <cstdint>
#include <pybind11/detail/common.h>
#include <torch/extension.h>
#include <vector>

// Return m + log(exp(a - m) + exp(b - m))
__device__ float logsumexp(float a, float b) {
  if (a == -INFINITY)
    return b;
  if (b == -INFINITY)
    return a;
  auto m = fmax(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

__global__ void ctc_forward_kernel(const float *log_probs, const int64_t *targets, const int64_t *input_lengths,
                                   const int64_t *target_lengths, float *losses, float *decode_map, const int64_t T,
                                   const int64_t B, const int64_t C, const int64_t S_max, const int64_t blank) {
  // Per batch per block
  const int64_t b = blockIdx.x;
  if (b >= B)
    return;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t S = 2 * target_length + 1;
  int64_t T_max = input_length;

  // Size: T * S
  extern __shared__ float dp[];

  for (int64_t i = 0; i < S; i += 1) {
    dp[i] = -INFINITY;
  }
  __syncthreads();

  dp[0] = log_probs[0 * B * C + b * C + blank];
  if (S > 1) {
    dp[1] = log_probs[0 * B * C + b * C + targets[b * target_length + 0]];
  }

  decode_map[b * T * S_max + 0 * S_max + 0] = dp[0];
  if (S > 1) {
    decode_map[b * T * S_max + 0 * S_max + 1] = dp[1];
  }

  for (int64_t t = 1; t < T_max; t += 1) {
    for (int64_t s = 0; s < S; s += 1) {
      float prev1 = dp[(t - 1) * S + s];
      float prev2 = (s > 0) ? dp[(t - 1) * S + s - 1] : -INFINITY;
      float prev3 =
          (s > 1 && (s % 2 == 1) && targets[b * target_length + s / 2] != targets[b * target_length + s / 2 - 1])
              ? dp[(t - 1) * S + s - 2]
              : -INFINITY;
      int64_t c = (s % 2 == 0) ? blank : targets[b * target_length + s / 2];
      float emit = log_probs[t * B * C + b * C + c];
      dp[t * S + s] = logsumexp(logsumexp(prev1, prev2), prev3) + emit;
      decode_map[b * T * S_max + t * S_max + s] = dp[t * S + s];
    }
  }

  float log_likelihood = logsumexp(dp[(T_max - 1) * S + S - 1], (S > 1) ? dp[(T_max - 1) * S + S - 2] : -INFINITY);
  losses[b] = -log_likelihood;
}

std::vector<torch::Tensor> ctc_decode_cuda(torch::Tensor log_probs, torch::Tensor targets, torch::Tensor input_lengths,
                                           torch::Tensor target_lengths, const int64_t blank) {
  const int64_t T = log_probs.size(0);
  const int64_t B = log_probs.size(1);
  const int64_t C = log_probs.size(2);

  auto losses = torch::zeros({B}, log_probs.options());
  auto grad_probs = torch::zeros_like(log_probs);

  int64_t S_max = 2 * target_lengths.max().item<int64_t>() + 1;
  auto decode_map = torch::full({B, T, S_max}, -INFINITY, log_probs.options());

  size_t shared_size = T * S_max * sizeof(float);

  ctc_forward_kernel<<<B, 1, shared_size>>>(log_probs.data_ptr<float>(), targets.data_ptr<int64_t>(),
                                            input_lengths.data_ptr<int64_t>(), target_lengths.data_ptr<int64_t>(),
                                            losses.data_ptr<float>(), decode_map.data_ptr<float>(), T, B, C, S_max,
                                            blank);

  return {losses, decode_map};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ctc_decode_cuda", &ctc_decode_cuda, "CTCLoss with decode map (CUDA)");
}
