from register_op import rnnt_loss
import tensorflow as tf


def run_test():
    acts = tf.constant([
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
    acts = tf.math.log(acts)

    labels = tf.constant([[1, 2]], shape=(1, 2), dtype=tf.int32)
    input_lengths = tf.constant([4], shape=(1,), dtype=tf.int32)
    label_lengths = tf.constant([2], shape=(1,), dtype=tf.int32)

    costs = rnnt_loss(acts, labels, input_lengths, label_lengths)
    cost = costs[0]
    print()
    if abs(cost - 1.01) < 1e-02:
        print("Test passed")
    else:
        print(f"Test failed. Got value {cost}, should be close to 1.01")
    print()


if __name__ == "__main__":
    run_test()
