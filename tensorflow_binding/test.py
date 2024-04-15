import logging
import os
import time

from _pytest.config import Notset
import pytest
import tensorflow as tf

from tensorflow_binding.register_op import register_op, rnnt_loss


@pytest.fixture(scope="session", autouse=True)
def setup_lib(pytestconfig: pytest.Config) -> None:
    lib_path = pytestconfig.getoption("lib_path")
    assert not isinstance(lib_path, Notset)
    if not os.path.isfile(lib_path):
        pytest.exit(f"Provided library file {lib_path} does not exist.", 1)

    register_op(lib_path)


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
        costs = rnnt_loss(acts, labels, lengths, label_lengths)  # type: ignore
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
            costs = rnnt_loss(acts, labels, lengths, label_lengths)  # type: ignore
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

    avg = sum(times) / len(times)
    variance = sum((t - avg) * (t - avg) for t in times) / len(times)
    logging.info(
        f"All iterations for size test (B={B}, T={T}, S={S}, V={V}) completed. Average time: {avg:.2f} ms, variance: {variance:.4f}."
    )

    grads = g.gradient(costs, acts)
    assert not tf.math.reduce_any(tf.math.is_inf(costs))  # type: ignore
    assert not tf.math.reduce_any(tf.math.is_nan(costs))  # type: ignore
    assert not tf.math.reduce_any(tf.math.is_inf(grads))  # type: ignore
    assert not tf.math.reduce_any(tf.math.is_nan(grads))  # type: ignore


def test_size_1() -> None:
    run_size_test(1, 150, 20, 50, 10)


def test_size_2() -> None:
    run_size_test(1, 150, 20, 5000, 10)


def test_size_3() -> None:
    run_size_test(16, 150, 20, 50, 10)


def test_size_4() -> None:
    run_size_test(16, 150, 20, 5000, 10)
