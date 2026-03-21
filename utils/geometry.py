from typing import Dict, List, Tuple, Union


def coerce_bbox(bbox: Union[Dict[str, int], Tuple[int, int, int, int], List[int]]):
    if isinstance(bbox, dict):
        return int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

def bbox_to_latent_coords(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    latent_hw: Tuple[int, int],
):
    x1, y1, x2, y2 = bbox
    image_w, image_h = image_size
    latent_h, latent_w = latent_hw

    x1_l = int(round(x1 * latent_w / image_w))
    x2_l = int(round(x2 * latent_w / image_w))
    y1_l = int(round(y1 * latent_h / image_h))
    y2_l = int(round(y2 * latent_h / image_h))

    x1_l = max(0, min(x1_l, latent_w - 1))
    y1_l = max(0, min(y1_l, latent_h - 1))
    x2_l = max(x1_l + 1, min(x2_l, latent_w))
    y2_l = max(y1_l + 1, min(y2_l, latent_h))

    return y1_l, y2_l, x1_l, x2_l
