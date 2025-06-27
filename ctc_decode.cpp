#include <c10/util/Exception.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <vector>

std::vector<torch::Tensor> ctc_decode_cuda(torch::Tensor log_probs, torch::Tensor targets, torch::Tensor input_lengths,
                                           torch::Tensor target_lengths, const int blank);

std::vector<torch::Tensor> ctc_decode(torch::Tensor log_probs, torch::Tensor targets, torch::Tensor input_lengths,
                                      torch::Tensor target_lengths, const int blank) {
  TORCH_CHECK(log_probs.is_cuda(), "log_probs must be CUDA tensor");
  return ctc_decode(log_probs, targets, input_lengths, target_lengths, blank);
}

TORCH_LIBRARY(ctc_decode, m) { m.def("forward", &ctc_decode); }