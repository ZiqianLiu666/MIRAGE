import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _parse_flat_polygon(values: Any) -> List[int]:
    if (
        isinstance(values, list)
        and len(values) == 1
        and isinstance(values[0], list)
    ):
        values = values[0]
    return [int(round(float(v))) for v in values]


def normalize_polygons(mask_value: Any) -> List[List[Any]]:
    if all(isinstance(v, (int, float)) for v in mask_value):
        return [_parse_flat_polygon(mask_value)]
    return [_parse_flat_polygon(poly) for poly in mask_value]


def load_crop_instruction_map(path: str) -> Dict[str, List[Dict[str, Any]]]:
    records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            image_name = item["original_image"]
            records[image_name].append(
                {
                    "image": item["image"],
                    "new_instruction": item["new_instruction"],
                }
            )

    for image_name in records:
        records[image_name].sort(
            key=lambda x: int(re.search(r"(\d+)", x["image"]).group(1))
        )
    return records


def load_annotations(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            base_image_path = item["image"]
            editing_prompt = item["editing_instruction"]
            stem = os.path.splitext(os.path.basename(base_image_path))[0]
            entries.append(
                {
                    "file_id": str(item.get("id", stem)),
                    "base_image_path": base_image_path,
                    "image_name": os.path.basename(base_image_path),
                    "editing_prompt": editing_prompt,
                    "editing_type": str(item.get("editing_type_id", "")),
                    "mask_raw": item["mask"],
                }
            )
    return entries


def mask_decode(encoded_mask: List[Any], image_size: Tuple[int, int]) -> np.ndarray:
    new_mask = Image.new("L", image_size, 0)
    new_draw = ImageDraw.Draw(new_mask)
    new_draw.polygon(encoded_mask, outline=1, fill=1)
    return np.array(new_mask)


def calculate_metric(
    metrics_calculator: Any,
    metric: str,
    src_image: Image.Image,
    tgt_image: Image.Image,
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    tgt_prompt: str,
):
    src_image = src_image.resize(
        (tgt_image.size[0], tgt_image.size[1]),
        resample=Image.Resampling.BILINEAR,
    )

    image_metric_fns = {
        "psnr": metrics_calculator.calculate_psnr,
        "lpips": metrics_calculator.calculate_lpips,
        "mse": metrics_calculator.calculate_mse,
        "ssim": metrics_calculator.calculate_ssim,
        "structure_distance": metrics_calculator.calculate_structure_distance,
    }

    if metric in image_metric_fns:
        return image_metric_fns[metric](src_image, tgt_image, None, None)

    unedit_suffix = "_unedit_part"
    if metric.endswith(unedit_suffix):
        base_metric = metric[: -len(unedit_suffix)]
        return image_metric_fns[base_metric](
            src_image, tgt_image, 1 - src_mask, 1 - tgt_mask
        )

    edit_suffix = "_edit_part"
    if metric.endswith(edit_suffix):
        base_metric = metric[: -len(edit_suffix)]
        if base_metric in image_metric_fns:
            return image_metric_fns[base_metric](
                src_image, tgt_image, src_mask, tgt_mask
            )

    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        return metrics_calculator.calculate_clip_similarity(
            tgt_image, tgt_prompt, tgt_mask
        )

    raise ValueError(f"Unsupported metric: {metric}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_mapping_file",
        type=str,
        default="RefEdit-Bench/annotations.jsonl",
    )
    parser.add_argument(
        "--src_image_folder",
        type=str,
        default="RefEdit-Bench",
    )
    parser.add_argument(
        "--tgt_methods",
        nargs="+",
        type=str,
        required=True,
        help="One or more folders that contain edited results.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="metrics/traditional/table5_summary.csv",
        help="Aggregated Table-5-style CSV output path.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--per_image_result_path",
        type=str,
        default="",
        help="Optional per-image CSV output path. Leave empty to skip per-image dump.",
    )
    parser.add_argument(
        "--crop-instruction-jsonl",
        type=str,
        required=True,
        help="Path to crop_instruction.jsonl containing new_instruction per crop.",
    )
    return parser.parse_args()


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _append_summary(
    output_csv: str,
    methods: Sequence[Tuple[str, str]],
    summary_metrics: Sequence[Tuple[str, str]],
    method_metric_values: Dict[Tuple[str, str], List[float]],
) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    write_header = (not os.path.exists(output_csv)) or os.path.getsize(output_csv) == 0

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["split", "model", *[label for _, label in summary_metrics]])

        for method_key, _ in methods:
            row_values: List[Any] = ["all", method_key]
            has_value = False
            for metric_name, _ in summary_metrics:
                avg = _mean(method_metric_values.get((method_key, metric_name), []))
                if avg is None:
                    row_values.append("")
                else:
                    row_values.append(f"{avg:.4f}")
                    has_value = True
            if has_value:
                writer.writerow(row_values)


def main() -> None:
    args = parse_args()
    from matrics_calculator import MetricsCalculator

    annotations = load_annotations(args.annotation_mapping_file)
    methods = [
        (os.path.basename(os.path.normpath(method_path)), method_path)
        for method_path in args.tgt_methods
    ]
    crop_instruction_map = load_crop_instruction_map(args.crop_instruction_jsonl)

    metrics_calculator = MetricsCalculator(args.device)
    summary_metrics = (
        ("structure_distance", "Structure Distance"),
        ("psnr_unedit_part", "PSNR"),
        ("lpips_unedit_part", "LPIPS"),
        ("mse_unedit_part", "MSE"),
        ("ssim_unedit_part", "SSIM"),
        ("clip_similarity_target_image", "CLIP Whole"),
        ("clip_similarity_target_image_edit_part", "CLIP Edited"),
    )
    metric_names = tuple(metric_name for metric_name, _ in summary_metrics)
    method_metric_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    per_image_context = nullcontext(None)
    if args.per_image_result_path:
        os.makedirs(os.path.dirname(args.per_image_result_path) or ".", exist_ok=True)
        per_image_context = open(
            args.per_image_result_path, "w", newline="", encoding="utf-8"
        )
    
    with per_image_context as per_image_file:
        per_image_writer = None
        if per_image_file is not None:
            per_image_writer = csv.writer(per_image_file)
            csv_head = []
            for method_key, _ in methods:
                for metric in metric_names:
                    csv_head.append(f"{method_key}|{metric}")
            per_image_writer.writerow(["file_id", "split", "editing_type"] + csv_head)

        for idx, ann in enumerate(annotations):
            file_id = ann["file_id"]
            editing_type = ann["editing_type"]
            base_image_path = ann["base_image_path"]

            src_image_path = os.path.join(args.src_image_folder, base_image_path)
            src_image = Image.open(src_image_path)
            editing_prompt = ann["editing_prompt"]
            image_name = ann["image_name"]

            edit_masks: List[np.ndarray] = []
            edit_prompts: List[str] = []
            polygons = normalize_polygons(ann["mask_raw"])
            crop_items = crop_instruction_map.get(image_name)

            for edit_idx, (poly, crop_item) in enumerate(zip(polygons, crop_items)):
                edit_prompt = crop_item["new_instruction"].strip()
                edit_mask = mask_decode(poly, src_image.size)
                edit_masks.append(edit_mask[:, :, np.newaxis].repeat(3, axis=2))
                edit_prompts.append(edit_prompt)

            union_mask = np.maximum.reduce(np.stack(edit_masks, axis=0))

            row_values: List[Any] = [file_id, "all", editing_type]

            for method_key, method_path in methods:
                tgt_image_path = os.path.join(method_path, base_image_path)
                tgt_image = Image.open(tgt_image_path)

                for metric in metric_names:
                    if metric == "clip_similarity_target_image_edit_part":
                        per_edit_scores: List[float] = []
                        for edit_mask, edit_prompt in zip(edit_masks, edit_prompts):
                            score = calculate_metric(
                                metrics_calculator,
                                metric,
                                src_image,
                                tgt_image,
                                edit_mask,
                                edit_mask,
                                edit_prompt,
                            )
                            numeric_score = _finite_float_or_none(score)
                            per_edit_scores.append(numeric_score)
                        metric_score = _mean(per_edit_scores)
                    else:
                        score = calculate_metric(
                            metrics_calculator,
                            metric,
                            src_image,
                            tgt_image,
                            union_mask,
                            union_mask,
                            editing_prompt,
                        )
                        metric_score = _finite_float_or_none(score)
                        
                    row_values.append(metric_score)
                    method_metric_values[(method_key, metric)].append(metric_score)

            if per_image_writer is not None:
                per_image_writer.writerow(row_values)

            if (idx + 1) % 20 == 0 or idx + 1 == len(annotations):
                print(f"Processed {idx + 1}/{len(annotations)}")

    _append_summary(
        output_csv=args.result_path,
        methods=methods,
        summary_metrics=summary_metrics,
        method_metric_values=method_metric_values,
    )
    print(f"Saved aggregated metrics to: {args.result_path}")
    if args.per_image_result_path:
        print(f"Saved per-image metrics to: {args.per_image_result_path}")


if __name__ == "__main__":
    main()
