__globals__ = ["register_op", "rnnt_loss", "_RNNTLossGrad"]

import os

import tensorflow as tf

from tensorflow.python.framework import ops

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

_monotonic_rnnt = None


def register_op(library_path: str) -> None:
    global _monotonic_rnnt
    _monotonic_rnnt = tf.load_op_library(library_path)


def monotonic_rnnt_loss(
    acts: tf.Tensor,
    labels: tf.Tensor,
    input_lengths: tf.Tensor,
    label_lengths: tf.Tensor,
    blank_label: int = 0,
) -> tf.Tensor:
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
    assert (
        _monotonic_rnnt is not None
    ), "Call `register_op` to register the operation before calling `rnnt_loss`."
    loss, _ = _monotonic_rnnt.monotonic_rnnt(
        acts,
        labels,
        input_lengths,
        label_lengths,
        blank_label,
    )
    return loss


# Computes gradient of operation with respect to each input
# Only the activations (input 0) have a gradient, all others do not
@ops.RegisterGradient("MonotonicRNNT")
def _RNNTLossGrad(op, grad_loss, _):
    """
    Args:
    op: Executed operation. Can be used to retreive input/output values.
    grad_loss: gradient with respect to first operation output (loss).
               Usually just [1, 1, ..., 1] (shape [B]).
    _: gradient with respect to second operation output (grad). Unused.

    Returns:
        Gradient with respect to each input. Only the activations (input 0)
        have a gradient, all others have None.
    """
    # If an activation at position i belongs to sample b in the minibatch, it
    # gets grad[i] * grad_loss[b] as gradient

    # Shape [T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1)].
    grad = op.outputs[1]
    # NOTE since here we are batch first, cannot use _BroadcastMul
    # [T_1, T_2, ..., T_B]
    input_lengths = op.inputs[2]
    # [S_1, S_2, ..., S_B]
    label_lengths = op.inputs[3]

    # [T_1*(S_1+1), T_2*(S_2+1), ..., T_B*(S_B+1)]
    repeats = tf.math.multiply(input_lengths, label_lengths + 1)

    # grad_loss has shape [B] -> extend to shape of activations by repetition
    # [T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1)]
    grad_loss = tf.repeat(grad_loss, repeats, axis=0)
    # [T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1), 1]
    grad_loss = tf.expand_dims(grad_loss, axis=1)
    # [T_1*(S_1+1) + T_2*(S_2+1) + ... + T_B*(S_B+1), 1]
    grad_loss = tf.math.multiply(grad_loss, grad)
    return [grad_loss, None, None, None]
