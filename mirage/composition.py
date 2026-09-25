import torch
import torch.nn.functional as F


def edit_strength(delta):
    """Per-cell norm of a branch's deviation from its reference, smoothed over 3x3 cells."""
    norm = delta.float().pow(2).sum(dim=1, keepdim=True).sqrt()
    return F.avg_pool2d(norm, kernel_size=3, stride=1, padding=1, count_include_pad=False)


def compose(canvas, branches):
    """Paste (latents, box, delta) branches onto the canvas; overlaps go to the stronger edit."""
    canvas = canvas.clone()
    best = torch.full_like(canvas[:, :1], -1.0, dtype=torch.float32)
    for latents, (y1, y2, x1, x2), delta in branches:
        strength = edit_strength(delta)
        region = best[..., y1:y2, x1:x2]
        wins = strength > region
        best[..., y1:y2, x1:x2] = torch.where(wins, strength, region)
        canvas[..., y1:y2, x1:x2] = torch.where(wins, latents.to(canvas.dtype), canvas[..., y1:y2, x1:x2])
    return canvas
