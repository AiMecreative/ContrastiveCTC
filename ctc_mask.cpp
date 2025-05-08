#include <torch/extension.h>
#include <vector>

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
ctc_decode_mask_forward_cu(torch::Tensor log_probs, torch::Tensor targets, torch::Tensor input_lengths,
                           torch::Tensor target_lengths);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
ctc_mask_forward(torch::Tensor log_probs, torch::Tensor targets, torch::Tensor input_lengths,
                 torch::Tensor target_lengths) {
  CHECK_INPUT(log_probs);
  CHECK_INPUT(targets);
  CHECK_INPUT(input_lengths);
  CHECK_INPUT(target_lengths);

  return ctc_decode_mask_forward_cu(log_probs, targets, input_lengths, target_lengths);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &ctc_mask_forward, "CTC Mask Forward (CUDA)"); }
