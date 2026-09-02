import numpy as np
import torch
from src.interventions import center_projection_transform, zero_projection_transform, addition_transform


def test_zero_projection_removes_last_token_component():
    h = torch.tensor([[[0.0, 2.0], [3.0, 4.0]]])
    out = zero_projection_transform(np.array([1.0, 0.0]))(h)
    assert torch.allclose(out[:, 0, :], h[:, 0, :])
    assert torch.allclose(out[:, -1, :], torch.tensor([[0.0, 4.0]]))


def test_center_projection_sets_projection_to_center():
    h = torch.tensor([[[3.0, 4.0]]])
    out = center_projection_transform(np.array([1.0, 0.0]), center=1.25)(h)
    assert torch.allclose(out[:, -1, 0], torch.tensor([1.25]))


def test_addition_changes_last_token_only():
    h = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    out = addition_transform(np.array([1.0, 0.0]), amount=2.0)(h)
    assert torch.allclose(out[:, 0, :], h[:, 0, :])
    assert torch.allclose(out[:, -1, :], torch.tensor([[4.0, 2.0]]))
