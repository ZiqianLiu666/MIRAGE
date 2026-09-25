import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage.vlm import BACKENDS, load_vlm, to_pixels  # noqa: E402

COLORS = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 128, 0), (255, 0, 255), (0, 255, 255)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Localize the referring expressions with a VLM and segment them with SAM2 (Sec. A.4). "
        "Adds `bbox` and `mask` to every record of --jsonl in place."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--vlm", default="qwen8b", choices=BACKENDS)
    parser.add_argument("--sam2-model-id", default="facebook/sam2-hiera-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4, help="images per VLM batch")
    parser.add_argument("--vis-dir", help="save box and mask overlays here")
    return parser.parse_args()


def mask_to_polygon(mask):
    """Largest outer contour of a binary mask as [x1, y1, x2, y2, ...]."""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [int(v) for v in max(contours, key=cv2.contourArea).reshape(-1)]


def visualize(image, boxes, polygons, path):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i, (box, polygon) in enumerate(zip(boxes, polygons)):
        color = COLORS[i % len(COLORS)]
        draw.polygon(polygon, fill=color + (120,), outline=color + (255,))
        draw.rectangle(box, outline=color + (255,), width=2)
        draw.text(box[:2], str(i + 1), fill=color + (255,))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(path)


def main():
    args = parse_args()
    with open(args.jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    vlm = load_vlm(args.vlm, device=args.device, dtype=torch.bfloat16)
    predictor = SAM2ImagePredictor.from_pretrained(args.sam2_model_id, device=args.device)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    for start in tqdm(range(0, len(records), args.batch_size)):
        batch = records[start : start + args.batch_size]
        images = [Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB") for r in batch]
        queries = [(image, phrase) for record, image in zip(batch, images) for phrase in record["refer_object"]]
        found = iter(vlm.locate([image for image, _ in queries], [phrase for _, phrase in queries]))

        for record, image in zip(batch, images):
            boxes = []
            for phrase in record["refer_object"]:
                box = next(found)
                if box is None:
                    raise ValueError(f"{args.vlm} found no box for '{phrase}' in {record['image']}")
                boxes.append(to_pixels(box, image.size))

            polygons = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(np.array(image))
                for box in boxes:
                    masks, _, _ = predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
                    polygons.append(mask_to_polygon(masks[0]))

            record["bbox"] = [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": phrase}
                for (x1, y1, x2, y2), phrase in zip(boxes, record["refer_object"])
            ]
            record["mask"] = polygons
            if args.vis_dir:
                visualize(image, boxes, polygons, os.path.join(args.vis_dir, Path(record["image"]).stem + ".png"))

    with open(args.jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
