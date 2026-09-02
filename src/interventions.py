from __future__ import annotations

import torch


def center_projection_transform(direction, center: float):
    d = torch.as_tensor(direction, dtype=torch.float32)

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        local_d = d.to(device=hidden.device, dtype=hidden.dtype)
        out = hidden.clone()
        x = out[:, -1, :]
        projection = (x * local_d).sum(dim=-1, keepdim=True)
        out[:, -1, :] = x - (projection - float(center)) * local_d
        return out

    return transform


def zero_projection_transform(direction):
    d = torch.as_tensor(direction, dtype=torch.float32)

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        local_d = d.to(device=hidden.device, dtype=hidden.dtype)
        out = hidden.clone()
        x = out[:, -1, :]
        projection = (x * local_d).sum(dim=-1, keepdim=True)
        out[:, -1, :] = x - projection * local_d
        return out

    return transform


def addition_transform(direction, amount: float):
    d = torch.as_tensor(direction, dtype=torch.float32)

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        local_d = d.to(device=hidden.device, dtype=hidden.dtype)
        out = hidden.clone()
        out[:, -1, :] = out[:, -1, :] + float(amount) * local_d
        return out

    return transform
