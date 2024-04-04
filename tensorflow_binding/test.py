from register_op import rnnt_loss
import tensorflow as tf
import numpy as np


def run_test():
    acts = tf.Tensor(
        [
            [0.6, 0.3, 0.1],
            [0.7, 0.1, 0.2],
            [0.5, 0.1, 0.4],

            [0.5, 0.4, 0.1],
            [0.5, 0.1, 0.4],
            [0.8, 0.1, 0.1],

            [0.4, 0.3, 0.3],
            [0.5, 0.1, 0.4],
            [0.7, 0.2, 0.1],

            [0.8, 0.1, 0.1],
            [0.3, 0.1, 0.6],
            [0.8, 0.1, 0.1],
        ], shape=(4 * 3, 3), dtype=tf.float32
    )

    labels = tf.Tensor([[1, 2]], shape=(1, 2), dtype=tf.int32)
    input_lengths = tf.Tensor([4], shape=(1,), dtype=tf.int32)
    label_lengths = tf.Tensor([2], shape=(1,), dtype=tf.int32)

    costs = rnnt_loss(acts, labels, input_lengths, label_lengths)

    assert costs[0] == -1.01, costs
