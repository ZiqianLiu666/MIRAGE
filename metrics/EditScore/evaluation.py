import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from editscore import EditScore

@dataclass
class CropEdit:
    crop_index: int
    instruction: str


@dataclass
class EvalSample:
    key: str
    input_path: Path
    edited_path: Path
    masks: list[Any]
    crop_edits: list[CropEdit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EditScore on mybench. SC uses local Qwen; PQ uses API."
    )
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--crop-instruction-jsonl", type=Path, required=True)
    parser.add_argument("--input-image-root", type=Path, required=True)
    parser.add_argument("--edited-image-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)

    parser.add_argument(
        "--sc-backbone",
        type=str,
        default="qwen3vllm",
        choices=["qwen25vl", "qwen25vl_vllm", "qwen3vl", "qwen3vl_vllm"],
    )
    parser.add_argument(
        "--sc-model-name-or-path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--sc-lora-path", type=str, default=None)

    parser.add_argument("--pq-model-name-or-path", type=str, default="gpt-4.1")
    parser.add_argument(
        "--pq-openai-url",
        type=str,
        default="https://api.openai.com/v1/chat/completions",
    )
    parser.add_argument("--pq-key", type=str, required=True)

    parser.add_argument("--num-pass", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--score-range", type=int, default=25)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--cache-dir", type=str, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_crop_instruction_map(crop_instruction_jsonl: Path) -> dict[str, list[str]]:
    raw_map: dict[str, list[tuple[int, str]]] = {}
    for row in load_jsonl(crop_instruction_jsonl):
        original_image = str(row["original_image"])
        crop_match = re.search(r"(\d+)", Path(str(row["image"])).stem)
        crop_index = int(crop_match.group(1)) - 1
        instruction = str(row["new_instruction"]).strip()
        raw_map.setdefault(original_image, []).append((crop_index, instruction))

    crop_map: dict[str, list[str]] = {}
    for original_image, items in raw_map.items():
        crop_map[original_image] = [
            instruction for _, instruction in sorted(items, key=lambda item: item[0])
        ]
    return crop_map


def decode_polygon(masks: list[Any], crop_index: int) -> list[int | float]:
    polygon = masks[crop_index]
    if (
        isinstance(polygon, list)
        and len(polygon) == 1
        and isinstance(polygon[0], list)
        and len(polygon[0]) >= 6
    ):
        polygon = polygon[0]
    return polygon


def mask_decode(
    encoded_mask: list[int | float], image_size: tuple[int, int]
) -> np.ndarray:
    mask_image = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask_image).polygon(encoded_mask, outline=1, fill=1)
    return np.array(mask_image)


def build_mybench_samples(
    annotations: list[dict[str, Any]],
    crop_instruction_jsonl: Path,
    input_root: Path,
    edited_root: Path,
) -> list[EvalSample]:
    crop_map = load_crop_instruction_map(crop_instruction_jsonl)
    samples: list[EvalSample] = []
    for image_index, row in enumerate(annotations):
        image_name = str(row["image"])
        input_path = input_root / image_name
        edited_path = edited_root / image_name
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input image: {input_path}")
        if not edited_path.exists():
            raise FileNotFoundError(f"Missing edited image: {edited_path}")

        masks = row["mask"]
        crop_instructions = crop_map.get(image_name)
        if not isinstance(masks, list) or not masks:
            raise ValueError(f"Invalid mask format for image {image_name}")
        if crop_instructions is None:
            raise ValueError(f"Missing crop instructions for image {image_name}")
        if len(masks) != len(crop_instructions):
            raise ValueError(
                f"Mask / crop instruction mismatch for image {image_name}: "
                f"{len(masks)} masks vs {len(crop_instructions)} crop instructions"
            )

        crop_edits = [
            CropEdit(crop_index=crop_index, instruction=instruction)
            for crop_index, instruction in enumerate(crop_instructions)
        ]
        samples.append(
            EvalSample(
                key=str(image_index),
                input_path=input_path,
                edited_path=edited_path,
                masks=masks,
                crop_edits=crop_edits,
            )
        )
    return samples


def open_aligned_pair(
    input_path: Path,
    edited_path: Path,
) -> tuple[Image.Image, Image.Image]:
    with Image.open(input_path) as input_img:
        input_image = input_img.convert("RGB")
    with Image.open(edited_path) as edited_img:
        edited_image = edited_img.convert("RGB")

    if input_image.size != edited_image.size:
        raise ValueError(
            f"Input / edited size mismatch: {input_path}={input_image.size}, "
            f"{edited_path}={edited_image.size}"
        )

    return input_image, edited_image


def build_sc_pair(
    input_image: Image.Image,
    edited_image: Image.Image,
    masks: list[Any],
    crop_index: int,
) -> tuple[Image.Image, Image.Image]:
    polygon = decode_polygon(masks, crop_index)
    mask = mask_decode(polygon, input_image.size).astype(np.uint8)
    masked_input = Image.fromarray(np.array(input_image) * mask[:, :, None])
    masked_edited = Image.fromarray(np.array(edited_image) * mask[:, :, None])
    return masked_input, masked_edited


def metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows))


def round_metrics(
    metrics: dict[str, float | None], ndigits: int = 4
) -> dict[str, float | None]:
    return {
        key: None if value is None else round(float(value), ndigits)
        for key, value in metrics.items()
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sc_scorer(args: argparse.Namespace) -> EditScore:
    return EditScore(
        backbone=args.sc_backbone,
        model_name_or_path=args.sc_model_name_or_path,
        score_range=args.score_range,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_pass=args.num_pass,
        lora_path=args.sc_lora_path,
        cache_dir=args.cache_dir,
    )


def build_pq_scorer(args: argparse.Namespace) -> EditScore:
    return EditScore(
        backbone="openai",
        key=args.pq_key,
        openai_url=args.pq_openai_url,
        model_name_or_path=args.pq_model_name_or_path,
        score_range=args.score_range,
        temperature=args.temperature,
        num_pass=args.num_pass,
    )


def evaluate_mybench(
    samples: list[EvalSample],
    sc_scorer: EditScore,
    pq_scorer: EditScore,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_edit_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="mybench-eval"):
        input_image, edited_image = open_aligned_pair(
            sample.input_path,
            sample.edited_path,
        )

        pq_result = pq_scorer.evaluate(
            [input_image, edited_image],
            "",
            only_pq=True,
        )
        image_perceptual_quality = float(pq_result["perceptual_quality"])
        image_pq_reasoning = str(pq_result.get("PQ_reasoning", ""))

        this_image_edit_rows: list[dict[str, Any]] = []
        for crop_edit in sample.crop_edits:
            sc_input, sc_edited = build_sc_pair(
                input_image=input_image,
                edited_image=edited_image,
                masks=sample.masks,
                crop_index=crop_edit.crop_index,
            )

            sc_result = sc_scorer.evaluate(
                [sc_input, sc_edited],
                crop_edit.instruction,
                only_sc=True,
            )
            per_edit_row = {
                "key": f"{sample.key}__{crop_edit.crop_index}",
                "image_key": sample.key,
                "crop_index": crop_edit.crop_index,
                "prompt_following": float(sc_result["prompt_following"]),
                "consistency": float(sc_result["consistency"]),
                "SC_reasoning": str(sc_result.get("SC_reasoning", "")),
            }
            per_edit_rows.append(per_edit_row)
            this_image_edit_rows.append(per_edit_row)

        image_prompt_following = metric_mean(this_image_edit_rows, "prompt_following")
        image_consistency = metric_mean(this_image_edit_rows, "consistency")
        image_overall = math.sqrt(
            min(image_prompt_following, image_consistency) * image_perceptual_quality
        )

        per_image_rows.append(
            {
                "key": sample.key,
                "num_crops": len(sample.crop_edits),
                "num_sc_evals": len(this_image_edit_rows),
                "prompt_following": image_prompt_following,
                "consistency": image_consistency,
                "perceptual_quality": image_perceptual_quality,
                "overall": image_overall,
                "SC_reasoning": [row["SC_reasoning"] for row in this_image_edit_rows],
                "PQ_reasoning": image_pq_reasoning,
            }
        )

    def summarize_sc(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "prompt_following": metric_mean(rows, "prompt_following"),
            "consistency": metric_mean(rows, "consistency"),
        }

    def add_overall(
        metric: dict[str, float], perceptual_quality: float
    ) -> dict[str, float]:
        return {
            "prompt_following": metric["prompt_following"],
            "consistency": metric["consistency"],
            "perceptual_quality": perceptual_quality,
            "overall": math.sqrt(
                min(metric["prompt_following"], metric["consistency"])
                * perceptual_quality
            ),
        }

    dataset_pq = metric_mean(per_image_rows, "perceptual_quality")
    per_edit_metric = add_overall(summarize_sc(per_edit_rows), dataset_pq)
    per_image_metric = add_overall(summarize_sc(per_image_rows), dataset_pq)

    write_jsonl(output_dir / "per_edit_results.jsonl", per_edit_rows)
    write_jsonl(output_dir / "per_image_results.jsonl", per_image_rows)

    summary = {
        "dataset_type": "mybench",
        "num_images": len(per_image_rows),
        "num_edits": len(per_edit_rows),
        "per_edit": {"count": len(per_edit_rows), **round_metrics(per_edit_metric)},
        "per_image": {"count": len(per_image_rows), **round_metrics(per_image_metric)},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "granularity",
                "count",
                "prompt_following",
                "consistency",
                "perceptual_quality",
                "overall",
            ],
        )
        writer.writeheader()
        writer.writerow({"granularity": "per_edit", **summary["per_edit"]})
        writer.writerow({"granularity": "per_image", **summary["per_image"]})


def main() -> None:
    args = parse_args()
    annotations = load_jsonl(args.annotations_jsonl)
    if not annotations:
        raise ValueError(f"No annotations loaded from: {args.annotations_jsonl}")

    samples = build_mybench_samples(
        annotations=annotations,
        crop_instruction_jsonl=args.crop_instruction_jsonl,
        input_root=args.input_image_root,
        edited_root=args.edited_image_root,
    )
    if not samples:
        raise ValueError("No samples after filtering")

    sc_scorer = build_sc_scorer(args)
    pq_scorer = build_pq_scorer(args)
    output_dir = args.result_dir

    evaluate_mybench(
        samples=samples,
        sc_scorer=sc_scorer,
        pq_scorer=pq_scorer,
        output_dir=output_dir,
    )
    print(f"Done. Results are saved to: {output_dir}")


if __name__ == "__main__":
    main()
