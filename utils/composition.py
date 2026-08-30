from typing import Tuple

import torch
import torch.nn.functional as F


LatentBBox = Tuple[int, int, int, int]
ACTIVITY_SMOOTH_RADIUS = 1


def _smooth(x: torch.Tensor) -> torch.Tensor:
    radius = ACTIVITY_SMOOTH_RADIUS
    kernel = 2 * radius + 1
    height, width = x.shape[-2], x.shape[-1]
    shape = x.shape
    flat = x.reshape(-1, 1, height, width)
    flat = F.avg_pool2d(
        flat, kernel_size=kernel, stride=1, padding=radius, count_include_pad=False
    )
    return flat.reshape(shape)


def activity(delta: torch.Tensor) -> torch.Tensor:
    """Per-cell edit magnitude of one branch, against its own reference."""
    return _smooth(delta.float().pow(2).sum(dim=1, keepdim=True).sqrt())


class RegionComposer:
    """Resolve overlapping regions by their edit magnitude."""

    def __init__(self, reference: torch.Tensor):
        self._canvas = reference.clone()
        owner_shape = list(reference.shape)
        owner_shape[1] = 1
        self._best = torch.full(
            owner_shape, -1.0, dtype=torch.float32, device=reference.device
        )
        self._owner = torch.full(
            owner_shape, 2**30, dtype=torch.int32, device=reference.device
        )

    def add(
        self,
        latents: torch.Tensor,
        latent_bbox: LatentBBox,
        delta: torch.Tensor,
        branch_id: int,
    ) -> None:
        y1, y2, x1, x2 = latent_bbox
        claim = activity(delta)

        best = self._best[..., y1:y2, x1:x2]
        owner = self._owner[..., y1:y2, x1:x2]
        window = self._canvas[..., y1:y2, x1:x2]

        takes = (claim > best) | ((claim == best) & (branch_id < owner))

        self._best[..., y1:y2, x1:x2] = torch.where(takes, claim, best)
        self._owner[..., y1:y2, x1:x2] = torch.where(
            takes, torch.full_like(owner, branch_id), owner
        )
        self._canvas[..., y1:y2, x1:x2] = torch.where(
            takes, latents.to(window.dtype), window
        )

    def compose(self) -> torch.Tensor:
        return self._canvas
