import math

from PIL import Image

# FLUX.2's image processor rejects inputs smaller than 64 px on either side.
MIN_CROP_SIZE = 64


def bbox_to_latent(bbox, image_size, latent_hw):
    """Latent cells (y1, y2, x1, x2) covered by a pixel bbox."""
    x1, y1, x2, y2 = (int(bbox[k]) for k in ("x1", "y1", "x2", "y2"))
    width, height = image_size
    latent_h, latent_w = latent_hw
    x1 = max(0, min(math.floor(x1 * latent_w / width), latent_w - 1))
    y1 = max(0, min(math.floor(y1 * latent_h / height), latent_h - 1))
    x2 = max(x1 + 1, min(math.ceil(x2 * latent_w / width), latent_w))
    y2 = max(y1 + 1, min(math.ceil(y2 * latent_h / height), latent_h))
    return y1, y2, x1, x2


def grow(lo, hi, size, limit):
    """Widen [lo, hi) to at least `size` around its center, staying inside [0, limit)."""
    if hi - lo >= size:
        return lo, hi
    extra = size - (hi - lo)
    lo -= extra // 2
    hi += extra - extra // 2
    if lo < 0:
        hi -= lo
        lo = 0
    if hi > limit:
        lo -= hi - limit
        hi = limit
    return max(lo, 0), hi


def min_size_box(box, image_size, latent_hw):
    """Grow a latent box until its crop is at least MIN_CROP_SIZE pixels on each side."""
    width, height = image_size
    latent_h, latent_w = latent_hw
    y1, y2, x1, x2 = box
    y1, y2 = grow(y1, y2, math.ceil(MIN_CROP_SIZE / (height // latent_h)), latent_h)
    x1, x2 = grow(x1, x2, math.ceil(MIN_CROP_SIZE / (width // latent_w)), latent_w)
    return y1, y2, x1, x2


def crop_cells(image, image_size, latent_hw, box):
    """Crop the pixels under the latent cells `box` of the image resized to `image_size`."""
    width, height = image_size
    stride_y, stride_x = height // latent_hw[0], width // latent_hw[1]
    y1, y2, x1, x2 = box
    if image.size != image_size:
        image = image.resize(image_size, Image.Resampling.LANCZOS)
    return image.crop((x1 * stride_x, y1 * stride_y, x2 * stride_x, y2 * stride_y))
