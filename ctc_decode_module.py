import torch
import ctc_decode


class CTCDecodeLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int,
    ):
        losses: torch.Tensor
        decode_map: torch.Tensor
        losses, decode_map = ctc_decode.ctc_decode(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank,
        )
        ctx.save_for_backward(log_probs, decode_map)
        return losses.mean(), decode_map

    @staticmethod
    def backward(
        ctx,
        grad_loss: torch.Tensor,
        grad_decode_map: torch.Tensor,
    ):
        log_probs, decode_map = ctx.saved_tensors
        # grad_input = torch.autograd.grad(
        #     outputs=log_probs.log_softmax(dim=2),
        #     inputs=log_probs,
        #     grad_outputs=grad_loss.expand_as(log_probs),
        #     retain_graph=True,
        #     allow_unused=True,
        # )[0]
        return grad_loss * torch.ones_like(log_probs), None, None, None, None


def ctc_decode_fn(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
):
    return CTCDecodeLoss.apply(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        torch.tensor(blank, dtype=torch.long),
    )
