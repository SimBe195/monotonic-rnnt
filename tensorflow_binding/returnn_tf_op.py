"""
Provides a RETURNN wrapper around `monotonic-rnnt`:
  https://github.com/SimBe195/monotonic-rnnt.git

Importing this module immediately compiles the library and TF module.
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import os
from returnn.tf.util.basic import OpCodeCompiler


tensorflow_binding_dir = os.path.dirname(os.path.abspath(__file__))
_tf_mod = None


def is_checked_out():
    """Checks if the git submodule is checkout out."""
    return os.path.isfile(f"{tensorflow_binding_dir}/monotonic_rnnt_op.cu")


def init_monotonic_rnnt(verbose=False):
    """
    Initialiazes and compiles the library. Caches the TF module.
    :param bool verbose:
    """
    global _tf_mod
    assert (
        is_checked_out()
    ), "submodule not checked out? Run `git submodule update --init --recursive`"

    import sys

    enable_gpu = "CUDA_HOME" in os.environ
    if not enable_gpu:
        print(
            "CUDA_HOME not found in the environment so building "
            "without GPU support. To build with GPU support "
            "please define the CUDA_HOME environment variable. "
            "This should be a path which contains include/cuda.h",
            file=sys.stderr,
        )

    src_files = [
        f"{tensorflow_binding_dir}/monotonic_rnnt_op.cu",
    ]
    assert all(
        [os.path.isfile(f) for f in src_files]
    ), f"some of the files {src_files} do not exist."
    src_code = ""
    for fn in src_files:
        f_code = open(fn).read()
        filename = os.path.basename(fn)
        src_code += f"\n// ------------ {filename} : BEGIN {{ ------------\n"
        # https://gcc.gnu.org/onlinedocs/cpp/Line-Control.html#Line-Control
        src_code += f'#line 1 "{filename}"\n'
        src_code += f_code
        src_code += f"\n// ------------ {filename} : END }} --------------\n\n"

    # Available macro definitions:
    # USE_NAIVE_KERNEL: use simpler unoptimized alpha-beta kernel
    # DEBUG_TIME: output execution time of kernel
    # DEBUG_SPACE: output reserved workspace size
    # DEBUG_LOG_SOFTMAX: output log softmax denominator
    # DEBUG_FWDBWD: output alphas and betas
    # DEBUG_GRADS: output gradients
    # RNNT_ENABLE_GPU: compile with cuda kernels
    # RNNT_DISABLE_OMP: Disable usage of OpenMP for CPU parallelization
    compiler = OpCodeCompiler(
        base_name="monotonic_rnnt_kernels",
        code_version=1,
        code=src_code,
        include_paths=(f"{tensorflow_binding_dir}/../include",),
        c_macro_defines={"RNNT_ENABLE_GPU": enable_gpu},
        ld_flags=["-Xcompiler", "-fopenmp"],
        is_cpp=True,
        use_cuda_if_available=True,
        verbose=verbose,
    )
    tf_mod = compiler.load_tf_module()
    assert hasattr(tf_mod, "MonotonicRNNT"), f"content of mod: {dir(tf_mod)}"
    _tf_mod = tf_mod
    return tf_mod


"""
Copied from monotonic-rnnt/tensorflow_binding/register_op.py
"""


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
        _tf_mod is not None
    ), "Call `init_monotonic_rnnt` to register the operation before calling `rnnt_loss`."
    loss, _ = _tf_mod.monotonic_rnnt(
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


init_monotonic_rnnt(True)
