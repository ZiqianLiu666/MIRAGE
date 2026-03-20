import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam2.sam2_image_predictor import SAM2ImagePredictor
import utils.qwen_utils as qwen

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate bbox and mask using Qwen referring-expression "
            "localization plus SAM2, then update the JSONL in place."
        )
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing full images.",
    )
    parser.add_argument(
        "--jsonl",
        default="synthesis_pipeline/instruction.fixed.jsonl",
        help="Instruction JSONL to update in-place with fields `bbox` and `mask`.",
    )
    parser.add_argument(
        "--sam2-model-id",
        default="facebook/sam2-hiera-large",
        help="Hugging Face SAM2 model id, e.g. facebook/sam2-hiera-large",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for SAM2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples per Qwen bbox batch. SAM2 still runs image-by-image.",
    )
    parser.add_argument(
        "--expand-mask-pixels",
        type=int,
        default=0,
        help="Dilate each mask by N pixels before polygon conversion.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="If set, save bbox and mask visualization images into this dir.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def chunk(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def clamp_box(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def norm1000_to_pixel(v: Any, total: int) -> int:
    val = max(0.0, min(float(v), 1000.0))
    return int(round(val / 1000.0 * total))


def bbox_to_pixel(
    bbox: Dict[str, Any], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    try:
        if "bbox_2d" in bbox:
            coords = bbox["bbox_2d"]
            if not isinstance(coords, list) or len(coords) != 4:
                return None
            x1 = norm1000_to_pixel(coords[0], width)
            y1 = norm1000_to_pixel(coords[1], height)
            x2 = norm1000_to_pixel(coords[2], width)
            y2 = norm1000_to_pixel(coords[3], height)
        else:
            x1 = int(round(float(bbox["x1"])))
            y1 = int(round(float(bbox["y1"])))
            x2 = int(round(float(bbox["x2"])))
            y2 = int(round(float(bbox["y2"])))
    except (KeyError, TypeError, ValueError):
        return None
    return clamp_box(x1, y1, x2, y2, width, height)


def pixel_box_to_dict(
    pixel_box: Optional[Tuple[int, int, int, int]], label: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    if pixel_box is None:
        return None
    x1, y1, x2, y2 = pixel_box
    out: Dict[str, Any] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    if label:
        out["label"] = label
    return out


def is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(x, str) and x.strip() for x in value
    )

def has_valid_bbox_list(value: Any, expected_len: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_len:
        return False
    return all(
        isinstance(item, dict) and (
            ("bbox_2d" in item and isinstance(item.get("bbox_2d"), list))
            or all(k in item for k in ("x1", "y1", "x2", "y2"))
        )
        for item in value
    )

def has_valid_mask_list(value: Any, expected_len: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_len:
        return False
    return all(isinstance(item, list) for item in value)


def select_best_mask(
    masks: np.ndarray, iou_scores: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    if masks is None:
        return None
    if masks.ndim == 2:
        return masks.astype(bool)
    if masks.ndim != 3 or masks.shape[0] == 0:
        return None

    if (
        iou_scores is not None
        and iou_scores.ndim == 1
        and iou_scores.size == masks.shape[0]
    ):
        idx = int(np.argmax(iou_scores))
    else:
        idx = 0
    return masks[idx].astype(bool)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    k = int(pixels) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return out.astype(bool)


def mask_to_polygon(mask: np.ndarray) -> List[int]:
    if mask is None:
        return []
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return []
    pts = contour.reshape(-1, 2)
    poly: List[int] = []
    for x, y in pts:
        poly.extend([int(x), int(y)])
    return poly


def _palette(i: int, alpha: int = 120):
    colors = [
        (255, 0, 0, alpha),
        (0, 255, 0, alpha),
        (0, 128, 255, alpha),
        (255, 128, 0, alpha),
        (255, 0, 255, alpha),
        (0, 255, 255, alpha),
        (128, 255, 0, alpha),
        (255, 0, 128, alpha),
        (0, 255, 128, alpha),
    ]
    return colors[i % len(colors)]


def draw_bboxes(
    img: Image.Image, bboxes: List[Optional[Tuple[int, int, int, int]]]
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for i, box in enumerate(bboxes):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = _palette(i, alpha=255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1), str(i + 1), fill=color)
    return out


def poly_to_points(poly: List[int]) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for i in range(0, len(poly), 2):
        if i + 1 >= len(poly):
            break
        pts.append((int(poly[i]), int(poly[i + 1])))
    return pts


def draw_masks(img: Image.Image, polygons: List[List[int]]) -> Image.Image:
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i, poly in enumerate(polygons):
        if not poly or len(poly) < 6:
            continue
        pts = poly_to_points(poly)
        if len(pts) < 3:
            continue
        color = _palette(i, alpha=120)
        draw.polygon(pts, fill=color, outline=(color[0], color[1], color[2], 255))
    return Image.alpha_composite(base, overlay).convert("RGB")
def detect_bboxes_batch(batch: List[Dict[str, Any]]) -> None:
    image_inputs: List[Image.Image] = []
    refer_inputs: List[str] = []
    mapping: List[Tuple[Dict[str, Any], int, str]] = []

    for item in batch:
        image = item["_image"]
        if item["need_bbox"]:
            item["_boxes_vis"] = [None] * len(item["refer_objects"])
            for idx, refer_object in enumerate(item["refer_objects"]):
                image_inputs.append(image)
                refer_inputs.append(refer_object)
                mapping.append((item, idx, refer_object))
        else:
            item["_boxes_vis"] = [
                bbox_to_pixel(bbox, image.width, image.height)
                for bbox in item["record"]["bbox"]
            ]

    if not image_inputs:
        return

    raw_outputs = qwen.locate_refer_object_batch(image_inputs, refer_inputs)
    for out_idx, (item, ref_idx, refer_object) in enumerate(mapping):
        image = item["_image"]
        raw_list = raw_outputs[out_idx] if out_idx < len(raw_outputs) else None
        raw_box = raw_list[0] if isinstance(raw_list, list) and raw_list else None
        pixel_box = bbox_to_pixel(raw_box, image.width, image.height)
        item["_boxes_vis"][ref_idx] = pixel_box

    for item in batch:
        if item["need_bbox"]:
            item["record"]["bbox"] = [
                pixel_box_to_dict(box, label=refer_object)
                for box, refer_object in zip(item["_boxes_vis"], item["refer_objects"])
            ]


def infer_one(
    item: Dict[str, Any],
    predictor,
    expand_mask_pixels: int,
    vis_dir: Optional[Path],
) -> None:
    device_type = str(predictor.device).split(":")[0]
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    image = item["_image"]
    image_np = np.array(image)
    boxes_vis = item["_boxes_vis"]

    mask_polygons: List[List[int]] = []

    with torch.inference_mode():
        with autocast_ctx:
            predictor.set_image(image_np)
            for pixel_box in boxes_vis:
                if pixel_box is None:
                    mask_polygons.append([])
                    continue

                box_np = np.array(pixel_box, dtype=np.float32)
                try:
                    masks, iou_scores, _ = predictor.predict(
                        box=box_np,
                        multimask_output=False,
                        normalize_coords=True,
                    )
                except Exception as exc:
                    print(
                        f"[WARN] SAM2 predict failed: image={item['image_name']} box={pixel_box} err={exc}"
                    )
                    mask_polygons.append([])
                    continue

                best_mask = select_best_mask(masks, iou_scores)
                if best_mask is None:
                    mask_polygons.append([])
                    continue
                best_mask = dilate_mask(best_mask, expand_mask_pixels)
                mask_polygons.append(mask_to_polygon(best_mask))

    predictor.reset_predictor()
    item["record"]["mask"] = mask_polygons

    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(item["image_name"]).stem
        draw_bboxes(image, boxes_vis).save(vis_dir / f"{stem}_bbox.png")
        draw_masks(image, mask_polygons).save(vis_dir / f"{stem}_mask.png")

    item.pop("_boxes_vis", None)
    item.pop("_image", None)


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    jsonl_path = Path(args.jsonl)

    records = load_jsonl(jsonl_path)
    records_to_process = (
        records if args.max_items is None else records[: args.max_items]
    )

    tasks: List[Dict[str, Any]] = []
    for rec in records_to_process:
        image_name = rec.get("image")
        image_path = image_dir / image_name
        refer_objects = rec.get("refer_object")

        need_bbox = not has_valid_bbox_list(rec.get("bbox"), len(refer_objects))
        need_mask = need_bbox or not has_valid_mask_list(
            rec.get("mask"), len(refer_objects)
        )
        if not need_bbox and not need_mask:
            continue

        tasks.append(
            {
                "record": rec,
                "image_name": image_name,
                "image_path": image_path,
                "refer_objects": refer_objects,
                "need_bbox": need_bbox,
            }
        )

    if not tasks:
        print("No records to process.")
        write_jsonl(jsonl_path, records)
        print(f"JSONL updated: {jsonl_path}")
        return

    predictor = SAM2ImagePredictor.from_pretrained(
        model_id=args.sam2_model_id,
        device=args.device,
    )

    vis_dir = Path(args.vis_dir) if args.vis_dir else None

    updated = 0
    batch_size = max(1, int(args.batch_size))
    num_batches = (len(tasks) + batch_size - 1) // batch_size
    for batch in tqdm(
        chunk(tasks, batch_size),
        total=num_batches,
        desc="qwen_bbox_sam2_mask",
    ):
        for item in batch:
            item["_image"] = load_image(item["image_path"])
        detect_bboxes_batch(batch)
        for item in batch:
            infer_one(
                item=item,
                predictor=predictor,
                expand_mask_pixels=max(0, int(args.expand_mask_pixels)),
                vis_dir=vis_dir,
            )
            updated += 1
            write_jsonl(jsonl_path, records)

    write_jsonl(jsonl_path, records)
    print(f"BBox/mask updated records: {updated}")
    print(f"JSONL updated: {jsonl_path}")


if __name__ == "__main__":
    main()
