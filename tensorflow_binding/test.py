import tensorflow as tf
import time

from register_op import rnnt_loss


def run_test():
    print()
    print("Run small value test")
    acts = tf.constant(
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
        ],
        shape=(4 * 3, 3),
        dtype=tf.float32,
    )
    acts = tf.math.log(acts)

    labels = tf.constant([[1, 2]], shape=(1, 2), dtype=tf.int32)
    lengths = tf.constant([4], shape=(1,), dtype=tf.int32)
    label_lengths = tf.constant([2], shape=(1,), dtype=tf.int32)

    with tf.GradientTape() as g:
        g.watch(acts)
        costs = rnnt_loss(acts, labels, lengths, label_lengths)
    cost = costs[0]
    grads = g.gradient(costs, acts)

    expected_grads = tf.constant(
        [
            [0.04, -0.14, 0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.13, -0.19, 0.06],
            [-0.04, 0.04, -0.01],
            [0.0, 0.0, 0.0],
            [0.06, -0.1, 0.04],
            [0.01, 0.07, -0.08],
            [-0.06, 0.04, 0.02],
            [0.0, 0.0, 0.0],
            [0.14, 0.05, -0.19],
            [-0.11, 0.05, 0.05],
        ]
    )

    test_result = True

    if abs(cost - 1.01) > 1e-02:
        print(f"Loss test failed. Got loss value {cost}, should be close to 1.01")
        test_result = False
    else:
        print("Loss test passed.")

    try:
        tf.debugging.assert_near(expected_grads, grads, atol=1e-02)
        print("Gradient test passed.")
    except tf.errors.InvalidArgumentError as e:
        print(f"Gradient test failed. {e}")
        test_result = False

    if test_result is True:
        print("Tests passed")
    else:
        print("Tests failed.")
    print()


def run_size_test(B: int, T: int, S: int, V: int, num_iters: int = 1):
    print()
    print(f"Run size test with B={B}, T={T}, S={S}, V={V} for {num_iters} iterations")
    acts = tf.random.uniform((B * T * (S + 1), V), dtype=tf.float32)
    labels = tf.random.uniform((B, S), minval=1, maxval=V, dtype=tf.int32)
    lengths = tf.fill((B), T)
    label_lengths = tf.fill((B), S)

    times = []

    with tf.GradientTape() as g:
        g.watch(acts)
        for i in range(num_iters):
            start = time.perf_counter()
            costs = rnnt_loss(acts, labels, lengths, label_lengths)
            elapsed = time.perf_counter() - start
            print(f"Iteration {i} took {elapsed*1000:.2f} ms.")
            times.append(elapsed * 1000)

    avg = sum(times) / len(times)
    variance = sum((t - avg) * (t - avg) for t in times) / len(times)
    print(
        f"All iterations completed. Average time: {avg:.2f} ms, variance: {variance:.4f}."
    )

    grads = g.gradient(costs, acts)
    if (
        tf.math.reduce_any(tf.math.is_inf(costs))
        or tf.math.reduce_any(tf.math.is_nan(costs))
        or tf.math.reduce_any(tf.math.is_inf(grads))
        or tf.math.reduce_any(tf.math.is_nan(grads))
    ):
        print("Size test failed. Inf/nan value encountered.")
    else:
        print("Size test passed.")
    print()


if __name__ == "__main__":
    run_test()
    run_size_test(1, 150, 20, 50, 10)
    run_size_test(1, 150, 20, 5000, 10)
    run_size_test(16, 150, 20, 50, 10)
    run_size_test(16, 150, 20, 5000, 10)
