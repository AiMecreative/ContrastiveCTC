import torch
import ctc_decode_module


T, B, C = 50, 4, 37
log_probs = torch.randn(T, B, C, device="cuda").log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, C - 1, (B * 20,), dtype=torch.int64, device="cuda")
input_lengths = torch.full((B,), T, dtype=torch.int64, device="cuda")
target_lengths = torch.randint(5, 20, (B,), dtype=torch.int64, device="cuda")

loss, decode_map = ctc_decode_module.ctc_decode_fn(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=C - 1,
)
print("Loss:", loss.item())
print("Decode Map shape:", decode_map.shape)
loss.backward()
