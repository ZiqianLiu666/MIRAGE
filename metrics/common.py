import csv
import json
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

MASK_ALIGNMENT_POLICIES = ["source-frame", "full-frame", "flux2-crop"]
MASK_ALIGNMENT_HELP = (
    "how masks drawn on the source image map onto the edited image: source-frame if the edited image "
    "has the source resolution, full-frame if it shows the whole source frame at another resolution "
    "(MIRAGE, Qwen), flux2-crop for the official FLUX.2 pipelines, which downscale inputs to 1 MP and "
    "center-crop them to multiples of 16"
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_samples(annotations_jsonl, crop_instruction_jsonl):
    """(image name, instruction, masks, {edit index: local instruction}) for each benchmark image."""
    local = {}
    for record in load_jsonl(crop_instruction_jsonl):
        index = int(re.search(r"\d+", record["image"])[0]) - 1
        local.setdefault(record["original_image"], {})[index] = record["new_instruction"].strip()
    return [
        (r["image"], r["editing_instruction"], r["mask"], dict(sorted(local[r["image"]].items())))
        for r in load_jsonl(annotations_jsonl)
    ]


def polygon_mask(mask, size):
    polygons = mask if isinstance(mask[0], list) else [mask]
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        draw.polygon(polygon, outline=1, fill=1)
    return np.array(canvas)


def project_mask(mask, size, policy):
    if policy == "source-frame":
        return mask
    if policy == "full-frame":
        return mask if mask.size == size else mask.resize(size, Image.Resampling.NEAREST)
    width, height = mask.size
    if width * height > 1024 * 1024:
        scale = math.sqrt(1024 * 1024 / (width * height))
        mask = mask.resize((int(width * scale), int(height * scale)), Image.Resampling.NEAREST)
    left = (mask.width - size[0]) // 2
    top = (mask.height - size[1]) // 2
    return mask.crop((left, top, left + size[0], top + size[1]))


def masked_pair(source, edited, mask, policy):
    source_mask = Image.fromarray(polygon_mask(mask, source.size).astype(np.uint8))
    edited_mask = project_mask(source_mask, edited.size, policy)
    return (
        Image.fromarray(np.array(source) * np.array(source_mask)[:, :, None]),
        Image.fromarray(np.array(edited) * np.array(edited_mask)[:, :, None]),
    )


def _mean(rows, key):
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else None


def _overall(metrics):
    return math.sqrt(min(metrics["prompt_following"], metrics["consistency"]) * metrics["perceptual_quality"])


def evaluate(args, score_pq=None, score_sc=None):
    """Score all edited images and write per-edit, per-image and summary results.

    score_pq(source, edited) -> (score, reasoning)
    score_sc([(index, instruction, masked source, masked edited)]) -> {index: (PF, Cons, reasoning)}
    """
    samples = load_samples(args.annotations_jsonl, args.crop_instruction_jsonl)
    os.makedirs(args.result_dir, exist_ok=True)
    edit_rows, image_rows = [], []
    with (
        open(os.path.join(args.result_dir, "per_edit_results.jsonl"), "w", encoding="utf-8") as edit_file,
        open(os.path.join(args.result_dir, "per_image_results.jsonl"), "w", encoding="utf-8") as image_file,
    ):
        for key, (name, _, masks, edits) in enumerate(tqdm(samples)):
            source = Image.open(os.path.join(args.input_image_root, name)).convert("RGB")
            edited = Image.open(os.path.join(args.edited_image_root, name)).convert("RGB")
            row = {"key": str(key), "image": name, "num_crops": len(edits)}

            if score_pq:
                row["perceptual_quality"], row["PQ_reasoning"] = score_pq(source, edited)
            if score_sc:
                items = [
                    (i, text, *masked_pair(source, edited, masks[i], args.mask_alignment_policy))
                    for i, text in edits.items()
                ]
                scores = score_sc(items)
                rows = []
                for i in edits:
                    prompt_following, consistency, reasoning = scores[i]
                    rows.append(
                        {
                            "key": f"{key}__{i}",
                            "image_key": str(key),
                            "crop_index": i,
                            "prompt_following": prompt_following,
                            "consistency": consistency,
                            "SC_reasoning": reasoning,
                        }
                    )
                    edit_file.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                edit_file.flush()
                edit_rows += rows
                row["prompt_following"] = _mean(rows, "prompt_following")
                row["consistency"] = _mean(rows, "consistency")
                row["SC_reasoning"] = [r["SC_reasoning"] for r in rows]
            if score_pq and score_sc:
                row["overall"] = _overall(row)

            image_rows.append(row)
            image_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            image_file.flush()

    perceptual_quality = _mean(image_rows, "perceptual_quality")
    summary = {"num_images": len(image_rows), "num_edits": len(edit_rows)}
    for granularity, rows in (("per_edit", edit_rows), ("per_image", image_rows)):
        metrics = {
            "prompt_following": _mean(rows, "prompt_following"),
            "consistency": _mean(rows, "consistency"),
            "perceptual_quality": perceptual_quality if rows else None,
        }
        metrics["overall"] = None if None in metrics.values() else _overall(metrics)
        rounded = {k: None if v is None else round(v, 4) for k, v in metrics.items()}
        summary[granularity] = {"count": len(rows), **rounded}

    with open(os.path.join(args.result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.result_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        fields = ["granularity", "count", "prompt_following", "consistency", "perceptual_quality", "overall"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for granularity in ("per_edit", "per_image"):
            writer.writerow({"granularity": granularity, **summary[granularity]})
    print(json.dumps(summary, indent=2))
