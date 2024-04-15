import torch
from torch.utils.cpp_extension import load

monotonic_rnnt_cpp = load(name="monotonic_rnnt_cpp", sources=["pytorch_binding/monotonic_rnnt.cu"], extra_cflags=["-DRNNT_ENABLE_GPU", "-O2"], extra_include_paths=["include/"])

class MonotonicRNNTFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        acts: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
        blank_label: int = 0,
    ) -> torch.Tensor:
        assert monotonic_rnnt_cpp is not None
        is_cuda = acts.is_cuda
        loss_func = monotonic_rnnt_cpp.cpu_monotonic_rnnt if is_cuda else monotonic_rnnt_cpp.gpu_monotonic_rnnt
        grads = torch.zeros_like(acts) if acts.requires_grad else torch.zeros(0).to(acts)
        costs = torch.zeros(labels.size(0), dtype=acts.dtype)

        loss_func(
            acts,
            labels,
            input_lengths,
            label_lengths,
            blank_label,
            0
        )

        costs = costs.to(acts.device)
        ctx.save_for_backward(grads, input_lengths, label_lengths)

        return costs

    @staticmethod
    def backward(ctx, grad_outputs):
        # If an activation at position i belongs to sample b in the minibatch, it
        # gets grad[i] * grad_output[b] as gradient

        # Shape [T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1)].
        grad = ctx.saved_tensors[0]
        # [T_1, T_2, ..., T_B]
        input_lengths = ctx.saved_tensors[1]
        # [S_1, S_2, ..., S_B]
        label_lengths = ctx.saved_tensors[2]

        # [T_1*(S_1+1), T_2*(S_2+1), ..., T_B*(S_B+1)]
        repeats = input_lengths * (label_lengths + 1)

        # grad_output has shape [B] -> extend to shape of activations by repetition
        # [T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1)]
        grad_outputs = grad_outputs.repeat_interleave(repeats)
        # [T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1), 1]
        grad_outputs = grad_outputs.unsqueeze(1)
        # [T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1), 1]
        grad_outputs = grad_outputs * grad
        return [grad_outputs, None, None, None]



def monotonic_rnnt_loss(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    blank_label: int = 0,
) -> torch.Tensor:
    """Computes the RNNT loss between a sequence of activations and a
    ground truth labeling.
    Args:
        acts:            A packed 2-D Tensor of logits*. The dimensions should be
                         (T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1), V), where B is the minibatch index,
                         T_i and S_i are the lengths of feature- and label-sequence i respectively and V indexes
                         over activations for each symbol in the alphabet. The packing is assumed to be done in
                         row-major order, i.e. the order of activations for sample b of the batch should be:
                         (z_{1,1}, .., z_{1,S_b+1},
                         z_{2, 1}, .., z_{2, S_b+1}, ..,
                         z_{T_1,1}, .., z_{T_b, S_b+1}) with all samples occuring consecutively.
        labels:          A 2-D Tensor of ints with shape [B, max_b(S_b)], a padded label sequences.
        input_lengths:   A 1-D Tensor of ints, [T_1, T_2, ..., T_B], the number of time steps for each sequence in the
                         minibatch.
        label_lengths:   A 1-D Tensor of ints, [S_1, S_2, ..., S_B], the length of each label sequence for each example
                         in the minibatch.
        blank_label:     the label value/index that the RNNT calculation should use as the blank label
    Returns:
        1-D float Tensor of shape [B], the cost of each example in the minibatch
        (as negative log probabilities).
    * This class performs the softmax operation internally.
    """

    result = MonotonicRNNTFunction.apply(acts, labels, input_lengths, label_lengths, blank_label)
    assert result is not None
    return result

class MonotonicRNNTLoss(torch.nn.Module):
    """Computes the RNNT loss between a sequence of activations and a
    ground truth labeling.
    Args:
        blank_label:     the label value/index that the RNNT calculation should use as the blank label
    * This module performs the softmax operation internally.
    """
    def __init__(self, blank: int = 0) -> None:
        super().__init__()
        self.blank = blank
        self.loss = MonotonicRNNTFunction.apply

    def forward(self, acts: torch.Tensor, labels: torch.Tensor, input_lengths: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            acts:            A packed 2-D Tensor of logits*. The dimensions should be
                             (T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1), V), where B is the minibatch index,
                             T_i and S_i are the lengths of feature- and label-sequence i respectively and V indexes
                             over activations for each symbol in the alphabet. The packing is assumed to be done in
                             row-major order, i.e. the order of activations for sample b of the batch should be:
                             (z_{1,1}, .., z_{1,S_b+1},
                             z_{2, 1}, .., z_{2, S_b+1}, ..,
                             z_{T_1,1}, .., z_{T_b, S_b+1}) with all samples occuring consecutively.
            labels:          A 2-D Tensor of ints with shape [B, max_b(S_b)], a padded label sequences.
            input_lengths:   A 1-D Tensor of ints, [T_1, T_2, ..., T_B], the number of time steps for each sequence in the
                             minibatch.
            label_lengths:   A 1-D Tensor of ints, [S_1, S_2, ..., S_B], the length of each label sequence for each example
                             in the minibatch.
        Returns:
            1-D float Tensor of shape [B], the cost of each example in the minibatch
            (as negative log probabilities).
        """
        loss = self.loss(acts, labels, input_lengths, label_lengths, self.blank)
        assert loss is not None
        return loss

