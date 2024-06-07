import argparse
import time

import tensorflow as tf

from register_op import register_op, monotonic_rnnt_loss


def test_cost_grad_values() -> None:
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
    acts = tf.math.log(acts)  # type: ignore

    labels = tf.constant([[1, 2]], shape=(1, 2), dtype=tf.int32)
    lengths = tf.constant([4], shape=(1,), dtype=tf.int32)
    label_lengths = tf.constant([2], shape=(1,), dtype=tf.int32)

    with tf.GradientTape() as g:
        g.watch(acts)
        costs = monotonic_rnnt_loss(
            acts=acts,
            labels=labels,
            input_lengths=lengths,
            label_lengths=label_lengths,
            blank_label=0,
        )  # type: ignore
    cost = costs.numpy()[0]  # type: ignore
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

    assert abs(cost - 1.01) < 1e-02

    tf.debugging.assert_near(expected_grads, grads, atol=1e-02)

    print("Cost and grad test successful")


def test_align_restrict_values() -> None:
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
    acts = tf.math.log(acts)  # type: ignore

    labels = tf.constant([[1, 2]], shape=(1, 2), dtype=tf.int32)
    lengths = tf.constant([4], shape=(1,), dtype=tf.int32)
    label_lengths = tf.constant([2], shape=(1,), dtype=tf.int32)
    alignment = tf.constant([[0, 1, 0, 2]], dtype=tf.int32)

    costs = monotonic_rnnt_loss(
        acts=acts,
        labels=labels,
        input_lengths=lengths,
        label_lengths=label_lengths,
        alignment=alignment,
        max_distance_from_alignment=1,
        blank_label=0,
    )  # type: ignore
    cost = costs.numpy()[0]  # type: ignore

    assert abs(cost - 1.22) < 1e-02

    alignment = tf.constant([[1, 2, 0, 0]], dtype=tf.int32)

    costs = monotonic_rnnt_loss(
        acts=acts,
        labels=labels,
        input_lengths=lengths,
        label_lengths=label_lengths,
        alignment=alignment,
        max_distance_from_alignment=0,
        blank_label=0,
    )  # type: ignore
    cost = costs.numpy()[0]  # type: ignore

    assert abs(cost - 2.7) < 1e-02

    print("Align restrict test successful")


def run_size_test(B: int, T: int, S: int, V: int, num_iters: int = 1) -> None:
    assert num_iters > 0

    acts = tf.random.uniform((B * T * (S + 1), V), dtype=tf.float32)
    labels = tf.random.uniform((B, S), minval=1, maxval=V, dtype=tf.int32)
    lengths = tf.fill((B), T)
    label_lengths = tf.fill((B), S)

    times = []

    costs = None
    with tf.GradientTape() as g:
        g.watch(acts)
        for _ in range(num_iters):
            start = time.perf_counter()
            costs = monotonic_rnnt_loss(acts, labels, lengths, label_lengths)  # type: ignore
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

    avg = sum(times) / len(times)
    variance = sum((t - avg) * (t - avg) for t in times) / len(times)
    print(
        f"All iterations for size test (B={B}, T={T}, S={S}, V={V}) completed. Average time: {avg:.2f} ms, variance: {variance:.4f}."
    )

    grads = g.gradient(costs, acts)
    assert not tf.math.reduce_any(tf.math.is_inf(costs))  # type: ignore
    assert not tf.math.reduce_any(tf.math.is_nan(costs))  # type: ignore
    assert not tf.math.reduce_any(tf.math.is_inf(grads))  # type: ignore
    assert not tf.math.reduce_any(tf.math.is_nan(grads))  # type: ignore


def test_size_1() -> None:
    run_size_test(1, 150, 20, 50, 20)


def test_size_2() -> None:
    run_size_test(1, 150, 20, 5000, 20)


def test_size_3() -> None:
    run_size_test(16, 150, 20, 50, 20)


def test_size_4() -> None:
    run_size_test(16, 150, 20, 5000, 10)


def test_size_5() -> None:
    run_size_test(2, 391, 300, 79, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tensorflow op test suite")
    parser.add_argument(
        "lib_file",
        type=str,
        help="Path to compiled `libmonotonic_rnnt_tf_op.so` library file",
    )
    args = parser.parse_args()
    register_op(args.lib_file)
    test_cost_grad_values()
    test_size_1()
    test_size_2()
    test_size_3()
    test_size_4()
    test_size_5()
    test_align_restrict_values()
    print("All tests ran successfully!")
