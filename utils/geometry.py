import math
from typing import Dict, List, Optional, Tuple, Union

import PIL.Image
import torch

from diffusers.utils.torch_utils import randn_tensor


MIN_BRANCH_PIXELS = 64


def coerce_bbox(bbox: Union[Dict[str, int], Tuple[int, int, int, int], List[int]]):
    if isinstance(bbox, dict):
        return int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def bbox_to_latent_coords(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    latent_hw: Tuple[int, int],
):
    """Return the latent cells touched by an image-space box."""
    x1, y1, x2, y2 = bbox
    image_w, image_h = image_size
    latent_h, latent_w = latent_hw

    x1_l = int(math.floor(x1 * latent_w / image_w))
    x2_l = int(math.ceil(x2 * latent_w / image_w))
    y1_l = int(math.floor(y1 * latent_h / image_h))
    y2_l = int(math.ceil(y2 * latent_h / image_h))

    x1_l = max(0, min(x1_l, latent_w - 1))
    y1_l = max(0, min(y1_l, latent_h - 1))
    x2_l = max(x1_l + 1, min(x2_l, latent_w))
    y2_l = max(y1_l + 1, min(y2_l, latent_h))

    return y1_l, y2_l, x1_l, x2_l


def _grow_to(low, high, need, limit):
    """Widen [low, high) to at least ``need`` cells, symmetrically, inside ``limit``."""
    if high - low >= need:
        return low, high
    extra = need - (high - low)
    low -= extra // 2
    high += extra - extra // 2
    if low < 0:
        high -= low
        low = 0
    if high > limit:
        low -= high - limit
        high = limit
    return max(0, low), min(limit, high)


def min_size_latent_bbox(latent_bbox, full_size, latent_hw):
    """Grow a latent box to the image processor's minimum input size."""
    full_width, full_height = full_size
    latent_height, latent_width = latent_hw
    stride_y = max(1, full_height // max(1, latent_height))
    stride_x = max(1, full_width // max(1, latent_width))
    need_y = -(-MIN_BRANCH_PIXELS // stride_y)
    need_x = -(-MIN_BRANCH_PIXELS // stride_x)

    y1, y2, x1, x2 = latent_bbox
    y1, y2 = _grow_to(y1, y2, need_y, latent_height)
    x1, x2 = _grow_to(x1, x2, need_x, latent_width)
    return y1, y2, x1, x2


def aligned_crop(full_image, full_size, latent_hw, latent_bbox):
    """Cut a crop on latent-cell boundaries."""
    full_width, full_height = full_size
    latent_height, latent_width = latent_hw
    stride_y = max(1, full_height // max(1, latent_height))
    stride_x = max(1, full_width // max(1, latent_width))

    y1, y2, x1, x2 = latent_bbox
    box = (x1 * stride_x, y1 * stride_y, x2 * stride_x, y2 * stride_y)

    resized = full_image.convert("RGB")
    if resized.size != (full_width, full_height):
        resized = resized.resize((full_width, full_height), PIL.Image.LANCZOS)
    return resized.crop(box), (box[2] - box[0], box[3] - box[1])


def region_write_weight(
    latent_hw: Tuple[int, int],
    latent_bboxes: List[Tuple[int, int, int, int]],
    margin: int,
    device,
    dtype,
) -> torch.Tensor:
    """Build a linear write-weight ramp around the union of region boxes."""
    latent_height, latent_width = latent_hw
    weight = torch.zeros(1, 1, latent_height, latent_width, dtype=torch.float32, device=device)
    for y1, y2, x1, x2 in latent_bboxes:
        weight[..., y1:y2, x1:x2] = 1.0

    if margin > 0:
        dilated = weight
        for step in range(1, margin + 1):
            dilated = torch.nn.functional.max_pool2d(
                dilated, kernel_size=3, stride=1, padding=1
            )
            weight = torch.maximum(weight, dilated * (1.0 - step / (margin + 1)))

    return weight.to(dtype)


def make_noise_like(latents: torch.Tensor, generator: Optional[torch.Generator]):
    if generator is None:
        return torch.randn_like(latents)
    return randn_tensor(
        latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
    )


def add_noise_like(
    scheduler,
    sample: torch.Tensor,
    noise: torch.Tensor,
    timestep: Union[int, float, torch.Tensor],
):
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor(timestep, device=sample.device)
    else:
        timestep = timestep.to(sample.device)
    if timestep.ndim == 0:
        timestep = timestep.expand(sample.shape[0])
    return scheduler.scale_noise(sample=sample, timestep=timestep, noise=noise)
