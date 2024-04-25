import torch

from _pytest.config import Notset
import pytest

from pytorch_binding import monotonic_rnnt_loss



def test_cost_grad_values() -> None:
    acts = torch.tensor(
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
        dtype=torch.float32,
    )
    acts = torch.log(acts)  # type: ignore

    labels = torch.tensor([[1, 2]], dtype=torch.int32)
    lengths = torch.tensor([4], dtype=torch.int32)
    label_lengths = torch.tensor([2], dtype=torch.int32)

    acts.requires_grad_(True)

    costs = monotonic_rnnt_loss(acts, labels, lengths, label_lengths)  # type: ignore

    cost = costs.detach().numpy()[0]

    costs.backward()
    grads = acts.grad

    expected_grads = torch.tensor(
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
        ],
        dtype=torch.float32,
    )

    assert abs(cost - 1.01) < 1e-02

    assert torch.allclose(grads, expected_grads, atol=1e-02)
