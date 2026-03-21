try:
    import dotenv

    dotenv.load_dotenv(override=True)
except Exception:
    pass

import argparse
import csv
import json
import math
import os
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
    mask_index: int


@dataclass
class EvalSample:
    key: str
    image_index: int
    split: str
    instruction: str
    input_path: Path
    edited_path: Path
    mask_field: Any
    crop_edits: list[CropEdit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluation for EditScore on RefEdit-Bench and "
            "generate_benchmark/filtered_benchmark."
        )
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["auto", "refedit", "mybench"],
        help="Dataset mode. auto infers from inputs.",
    )
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument(
        "--crop-instruction-jsonl",
        type=Path,
        default=None,
        help="Required for per-crop SC instruction alignment.",
    )
    parser.add_argument("--input-image-root", type=Path, required=True)
    parser.add_argument("--edited-image-root", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--instruction-count",
        type=int,
        required=True,
        help=(
            "Required for instruction-count ablation. "
            "Evaluate only the first N instructions/masks per image."
        ),
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="openai",
        choices=[
            "openai",
            "qwen25vl",
            "qwen25vl_vllm",
            "internvl3_5",
            "qwen3vl",
            "qwen3vl_vllm",
        ],
    )
    parser.add_argument("--model_name_or_path", type=str, default="gpt-4.1")
    parser.add_argument(
        "--openai_url", type=str, default="https://api.openai.com/v1/chat/completions"
    )
    parser.add_argument("--key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--num_pass", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--score_range", type=int, default=25)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--lora_path", type=str, default="EditScore/EditScore-7B")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument(
        "--save-masked-visuals",
        action="store_true",
        help="Save SC input pairs actually sent to model (input / edited / pair).",
    )
    parser.add_argument(
        "--masked-visual-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for saved SC input visuals. "
            "Default: <result_dir>/<backbone>/sc_inputs"
        ),
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_dataset_type(
    requested: str,
    annotations: list[dict[str, Any]],
    crop_instruction_jsonl: Path | None,
) -> str:
    if requested != "auto":
        return requested
    first = annotations[0] if annotations else {}
    if "edit_instruction_single" in first:
        return "refedit"
    return "mybench"


def _candidate_names(image_name: str, image_index: int) -> list[str]:
    suffixes = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    p = Path(image_name)
    stem = p.stem
    out: list[str] = [image_name]
    out.extend([f"{stem}{suf}" for suf in suffixes])

    if stem.isdigit():
        numeric_stem = int(stem)
        out.extend([f"{numeric_stem}{suf}" for suf in suffixes])
        out.extend([f"{numeric_stem:05d}{suf}" for suf in suffixes])

    out.extend([f"{image_index}{suf}" for suf in suffixes])
    out.extend([f"{image_index:05d}{suf}" for suf in suffixes])

    dedup: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in seen:
            dedup.append(name)
            seen.add(name)
    return dedup


def resolve_image_path(root: Path, image_name: str, image_index: int, kind: str) -> Path:
    candidates = _candidate_names(image_name, image_index)
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot resolve {kind} image for {image_name} (index={image_index}) under "
        f"{root}. Tried: {candidates}"
    )


def _parse_crop_index(crop_name: str) -> int:
    m = re.search(r"(\d+)", Path(crop_name).stem)
    if not m:
        raise ValueError(f"Cannot parse crop index from: {crop_name}")
    return int(m.group(1)) - 1


def load_crop_instruction_map(crop_instruction_jsonl: Path) -> dict[str, dict[int, str]]:
    crop_rows = load_jsonl(crop_instruction_jsonl)
    crop_map: dict[str, dict[int, str]] = {}

    for row in crop_rows:
        original_image = str(row["original_image"])
        crop_idx = _parse_crop_index(str(row["image"]))
        instruction_raw = row.get("new_instruction")
        if not isinstance(instruction_raw, str) or not instruction_raw.strip():
            raise ValueError(
                "crop_instruction_jsonl requires non-empty `new_instruction` for "
                f"original_image={original_image}, crop={row.get('image')}"
            )
        crop_map.setdefault(original_image, {})[crop_idx] = instruction_raw.strip()

    return crop_map


def _split_editing_instruction(instruction: str) -> list[str]:
    if not isinstance(instruction, str):
        return []
    text = instruction.strip()
    if not text:
        return []
    text = text.rstrip(".")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    out: list[str] = []
    for part in parts:
        part = re.sub(r"^(and|then)\s+", "", part, flags=re.IGNORECASE).strip()
        if part:
            out.append(part)
    return out


def _join_instruction_clauses(clauses: list[str]) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0].rstrip(".") + "."
    return ", and ".join(clause.rstrip(".") for clause in clauses) + "."


def _truncate_instruction(instruction: str, instruction_count: int | None) -> str:
    if instruction_count is None:
        return instruction.strip() if isinstance(instruction, str) else ""
    clauses = _split_editing_instruction(instruction)
    if not clauses:
        return ""
    keep_count = max(1, min(int(instruction_count), len(clauses)))
    return _join_instruction_clauses(clauses[:keep_count])


def _ensure_mask_list(mask_field: Any) -> list[Any]:
    if isinstance(mask_field, str):
        mask_data = json.loads(mask_field)
    else:
        mask_data = mask_field
    if not isinstance(mask_data, list) or len(mask_data) == 0:
        raise ValueError(f"Invalid mask format: {type(mask_field)}")
    return mask_data


def _decode_polygon(mask_field: Any, polygon_index: int) -> list[int | float]:
    mask_data = _ensure_mask_list(mask_field)
    if polygon_index >= len(mask_data):
        raise IndexError(
            f"Mask polygon index out of range: {polygon_index} >= {len(mask_data)}"
        )
    polygon = mask_data[polygon_index]
    if (
        isinstance(polygon, list)
        and len(polygon) == 1
        and isinstance(polygon[0], list)
        and len(polygon[0]) >= 6
    ):
        polygon = polygon[0]
    if not isinstance(polygon, list) or len(polygon) < 6:
        raise ValueError("Mask polygon is invalid.")
    return polygon


def mask_decode(encoded_mask: list[int | float], image: Image.Image) -> np.ndarray:
    new_mask = Image.new("L", image.size, 0)
    new_draw = ImageDraw.Draw(new_mask)
    new_draw.polygon(encoded_mask, outline=1, fill=1)
    return np.array(new_mask)


def build_refedit_samples(
    annotations: list[dict[str, Any]],
    crop_instruction_jsonl: Path,
    input_root: Path,
    edited_root: Path,
    instruction_count: int | None,
    max_samples: int | None,
) -> list[EvalSample]:
    crop_map = load_crop_instruction_map(crop_instruction_jsonl)
    indexed = list(enumerate(annotations))
    if max_samples is not None:
        indexed = indexed[: max(0, max_samples)]

    samples: list[EvalSample] = []
    for image_index, row in indexed:
        image_name = str(row["image"])
        if image_name not in crop_map:
            raise ValueError(
                f"Missing crop instructions for image {image_name} in {crop_instruction_jsonl}"
            )

        input_path = resolve_image_path(input_root, image_name, image_index, "input")
        edited_path = resolve_image_path(edited_root, image_name, image_index, "edited")
        masks = _ensure_mask_list(row["mask"])
        limit = len(masks)
        if instruction_count is not None:
            limit = min(len(masks), int(instruction_count))
        if limit <= 0:
            raise ValueError(
                f"No instructions selected for image {image_name}. "
                f"instruction_count={instruction_count}, num_masks={len(masks)}"
            )

        crop_edits: list[CropEdit] = []
        for edit_idx in range(limit):
            instruction = crop_map[image_name].get(edit_idx)
            if instruction is None or not str(instruction).strip():
                raise ValueError(
                    f"Missing per-mask instruction for image {image_name}, "
                    f"mask index {edit_idx} in {crop_instruction_jsonl}"
                )
            crop_edits.append(
                CropEdit(
                    crop_index=edit_idx,
                    instruction=str(instruction).strip(),
                    mask_index=edit_idx,
                )
            )

        image_instruction_raw = str(row.get("editing_instruction", "")).strip()
        image_instruction = _truncate_instruction(
            image_instruction_raw, instruction_count=instruction_count
        )
        if not image_instruction:
            raise ValueError(
                f"Missing non-empty editing_instruction in annotations for image {image_name}"
            )
        split = "easy" if image_index < 100 else "hard"
        samples.append(
            EvalSample(
                key=str(image_index),
                image_index=image_index,
                split=split,
                instruction=image_instruction,
                input_path=input_path,
                edited_path=edited_path,
                mask_field=row["mask"],
                crop_edits=crop_edits,
            )
        )
    return samples


def build_mybench_samples(
    annotations: list[dict[str, Any]],
    crop_instruction_jsonl: Path,
    input_root: Path,
    edited_root: Path,
    instruction_count: int | None,
    max_samples: int | None,
) -> list[EvalSample]:
    crop_map = load_crop_instruction_map(crop_instruction_jsonl)

    indexed = list(enumerate(annotations))
    if max_samples is not None:
        indexed = indexed[: max(0, max_samples)]

    samples: list[EvalSample] = []
    for image_index, row in indexed:
        image_name = str(row["image"])
        if image_name not in crop_map:
            raise ValueError(
                f"Missing crop instructions for image {image_name} in {crop_instruction_jsonl}"
            )

        input_path = resolve_image_path(input_root, image_name, image_index, "input")
        edited_path = resolve_image_path(edited_root, image_name, image_index, "edited")
        masks = _ensure_mask_list(row["mask"])

        limit = len(masks)
        if instruction_count is not None:
            limit = min(len(masks), int(instruction_count))
        if limit <= 0:
            raise ValueError(
                f"No instructions selected for image {image_name}. "
                f"instruction_count={instruction_count}, num_masks={len(masks)}"
            )

        crop_edits: list[CropEdit] = []
        for edit_idx in range(limit):
            instruction = crop_map[image_name].get(edit_idx)
            if instruction is None or not str(instruction).strip():
                raise ValueError(
                    f"Missing per-mask instruction for image {image_name}, "
                    f"mask index {edit_idx} in {crop_instruction_jsonl}"
                )
            crop_edits.append(
                CropEdit(
                    crop_index=edit_idx,
                    instruction=str(instruction).strip(),
                    mask_index=edit_idx,
                )
            )

        image_instruction_raw = str(row.get("editing_instruction", "")).strip()
        image_instruction = _truncate_instruction(
            image_instruction_raw, instruction_count=instruction_count
        )
        if not image_instruction:
            raise ValueError(
                f"Missing non-empty editing_instruction in annotations for image {image_name}"
            )

        samples.append(
            EvalSample(
                key=str(image_index),
                image_index=image_index,
                split="overall",
                instruction=image_instruction,
                input_path=input_path,
                edited_path=edited_path,
                mask_field=row["mask"],
                crop_edits=crop_edits,
            )
        )

    return samples


def open_aligned_pair(input_path: Path, edited_path: Path) -> tuple[Image.Image, Image.Image]:
    input_image = Image.open(input_path).convert("RGB")
    edited_image = Image.open(edited_path).convert("RGB")
    if edited_image.size != input_image.size:
        edited_image = edited_image.resize(input_image.size)
    return input_image, edited_image


def build_sc_pair(
    input_image: Image.Image,
    edited_image: Image.Image,
    mask_field: Any,
    mask_index: int,
) -> tuple[Image.Image, Image.Image]:
    polygon = _decode_polygon(mask_field, mask_index)
    mask = mask_decode(polygon, input_image)
    masked_input = Image.fromarray(np.array(input_image) * mask[:, :, None])
    masked_edited = Image.fromarray(np.array(edited_image) * mask[:, :, None])
    return masked_input, masked_edited


def _metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(r[key]) for r in rows) / len(rows))


def _round_metrics(metrics: dict[str, float], ndigits: int = 4) -> dict[str, float]:
    return {k: round(float(v), ndigits) for k, v in metrics.items()}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_sc_visual_pair(
    sc_input: Image.Image,
    sc_edited: Image.Image,
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / f"{prefix}_input.png"
    edited_path = out_dir / f"{prefix}_edited.png"
    pair_path = out_dir / f"{prefix}_pair.png"

    if not input_path.exists():
        sc_input.save(input_path)
    if not edited_path.exists():
        sc_edited.save(edited_path)
    if not pair_path.exists():
        panel = Image.new(
            "RGB",
            (sc_input.width + sc_edited.width, max(sc_input.height, sc_edited.height)),
            (0, 0, 0),
        )
        panel.paste(sc_input, (0, 0))
        panel.paste(sc_edited, (sc_input.width, 0))
        panel.save(pair_path)


def evaluate_refedit(
    samples: list[EvalSample],
    scorer: EditScore,
    output_dir: Path,
    save_masked_visuals: bool,
    masked_visual_dir: Path,
) -> None:
    per_image_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "per_image_results.jsonl"
    if per_image_path.exists():
        per_image_path.unlink()

    for sample in tqdm(samples, desc="refedit-eval"):
        if not sample.crop_edits:
            raise ValueError(f"No crop edits found for image index {sample.image_index}")

        input_image, edited_image = open_aligned_pair(sample.input_path, sample.edited_path)

        image_pq_result = scorer.evaluate(
            [input_image, edited_image],
            "",
            only_pq=True,
        )
        image_perceptual_quality = float(image_pq_result["perceptual_quality"])
        image_pq_reasoning = str(image_pq_result.get("PQ_reasoning", ""))

        this_image_edit_rows: list[dict[str, Any]] = []
        for crop_edit in sample.crop_edits:
            sc_input, sc_edited = build_sc_pair(
                input_image=input_image,
                edited_image=edited_image,
                mask_field=sample.mask_field,
                mask_index=crop_edit.mask_index,
            )

            if save_masked_visuals:
                save_sc_visual_pair(
                    sc_input=sc_input,
                    sc_edited=sc_edited,
                    out_dir=masked_visual_dir / "mask" / f"image_{sample.key}",
                    prefix=f"{sample.key}__{crop_edit.crop_index}",
                )

            edit_result = scorer.evaluate(
                [sc_input, sc_edited],
                crop_edit.instruction,
                only_sc=True,
            )
            this_image_edit_rows.append(
                {
                    "prompt_following": float(edit_result["prompt_following"]),
                    "consistency": float(edit_result["consistency"]),
                    "SC_reasoning": str(edit_result.get("SC_reasoning", "")),
                }
            )

        prompt_following = _metric_mean(this_image_edit_rows, "prompt_following")
        consistency = _metric_mean(this_image_edit_rows, "consistency")
        perceptual_quality = image_perceptual_quality
        overall = math.sqrt(min(prompt_following, consistency) * perceptual_quality)
        sc_reasoning: list[str] = [r["SC_reasoning"] for r in this_image_edit_rows]

        summary_rows.append(
            {
                "split": sample.split,
                "prompt_following": prompt_following,
                "consistency": consistency,
                "perceptual_quality": perceptual_quality,
                "overall": overall,
            }
        )

        per_image_row = {
            "key": sample.key,
            "prompt_following": prompt_following,
            "consistency": consistency,
            "perceptual_quality": perceptual_quality,
            "overall": overall,
            "num_crops": len(sample.crop_edits),
            "num_sc_evals": len(this_image_edit_rows),
            "SC_reasoning": sc_reasoning,
            "PQ_reasoning": image_pq_reasoning,
        }
        per_image_rows.append(per_image_row)
        append_jsonl(per_image_path, per_image_row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
        pf = _metric_mean(rows, "prompt_following")
        cons = _metric_mean(rows, "consistency")
        pq = _metric_mean(rows, "perceptual_quality")
        return {
            "prompt_following": pf,
            "consistency": cons,
            "perceptual_quality": pq,
            # Paper-aligned aggregation: compute overall from aggregated PF/Cons/PQ.
            "overall": math.sqrt(min(pf, cons) * pq),
        }

    easy_rows = [r for r in summary_rows if r["split"] == "easy"]
    hard_rows = [r for r in summary_rows if r["split"] == "hard"]

    summary = {
        "dataset_type": "refedit",
        "num_images": len(per_image_rows),
        "easy": {"count": len(easy_rows), **_round_metrics(summarize(easy_rows))},
        "hard": {"count": len(hard_rows), **_round_metrics(summarize(hard_rows))},
        "overall": {"count": len(per_image_rows), **_round_metrics(summarize(per_image_rows))},
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "split",
            "count",
            "prompt_following",
            "consistency",
            "perceptual_quality",
            "overall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ["easy", "hard", "overall"]:
            row = summary[split]
            writer.writerow({"split": split, **row})


def evaluate_mybench(
    samples: list[EvalSample],
    scorer: EditScore,
    output_dir: Path,
    save_masked_visuals: bool,
    masked_visual_dir: Path,
) -> None:
    per_edit_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    per_edit_path = output_dir / "per_edit_results.jsonl"
    per_image_path = output_dir / "per_image_results.jsonl"
    if per_image_path.exists():
        per_image_path.unlink()
    if per_edit_path.exists():
        per_edit_path.unlink()

    for sample in tqdm(samples, desc="mybench-eval"):
        if not sample.crop_edits:
            raise ValueError(f"No crop edits found for image index {sample.image_index}")

        input_image, edited_image = open_aligned_pair(sample.input_path, sample.edited_path)

        image_pq_result = scorer.evaluate(
            [input_image, edited_image],
            "",
            only_pq=True,
        )
        image_perceptual_quality = float(image_pq_result["perceptual_quality"])
        image_pq_reasoning = str(image_pq_result.get("PQ_reasoning", ""))

        this_image_edit_rows: list[dict[str, Any]] = []

        for crop_edit in sample.crop_edits:
            sc_input, sc_edited = build_sc_pair(
                input_image=input_image,
                edited_image=edited_image,
                mask_field=sample.mask_field,
                mask_index=crop_edit.mask_index,
            )

            if save_masked_visuals:
                save_sc_visual_pair(
                    sc_input=sc_input,
                    sc_edited=sc_edited,
                    out_dir=masked_visual_dir / "mask" / f"image_{sample.key}",
                    prefix=f"{sample.key}__{crop_edit.crop_index}",
                )

            edit_result = scorer.evaluate(
                [sc_input, sc_edited],
                crop_edit.instruction,
                only_sc=True,
            )

            per_edit_row = {
                "key": f"{sample.key}__{crop_edit.crop_index}",
                "prompt_following": float(edit_result["prompt_following"]),
                "consistency": float(edit_result["consistency"]),
                "SC_reasoning": str(edit_result.get("SC_reasoning", "")),
            }
            per_edit_rows.append(per_edit_row)
            append_jsonl(per_edit_path, per_edit_row)
            this_image_edit_rows.append(per_edit_row)

        image_prompt_following = _metric_mean(this_image_edit_rows, "prompt_following")
        image_consistency = _metric_mean(this_image_edit_rows, "consistency")
        image_overall = math.sqrt(
            min(image_prompt_following, image_consistency) * image_perceptual_quality
        )
        image_sc_reasoning: list[str] = [r["SC_reasoning"] for r in this_image_edit_rows]

        per_image_row = {
            "key": sample.key,
            "num_crops": len(sample.crop_edits),
            "num_sc_evals": len(this_image_edit_rows),
            "prompt_following": image_prompt_following,
            "consistency": image_consistency,
            "perceptual_quality": image_perceptual_quality,
            "overall": image_overall,
            "SC_reasoning": image_sc_reasoning,
            "PQ_reasoning": image_pq_reasoning,
        }
        per_image_rows.append(per_image_row)
        append_jsonl(per_image_path, per_image_row)

    def summarize_sc(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "prompt_following": _metric_mean(rows, "prompt_following"),
            "consistency": _metric_mean(rows, "consistency"),
        }

    def add_overall(metric: dict[str, float], perceptual_quality: float) -> dict[str, float]:
        m = dict(metric)
        m["perceptual_quality"] = perceptual_quality
        m["overall"] = math.sqrt(
            min(m["prompt_following"], m["consistency"]) * perceptual_quality
        )
        return m

    dataset_pq = _metric_mean(per_image_rows, "perceptual_quality")
    per_edit_metric = add_overall(summarize_sc(per_edit_rows), dataset_pq)
    per_image_metric = add_overall(summarize_sc(per_image_rows), dataset_pq)

    summary = {
        "dataset_type": "mybench",
        "num_images": len(per_image_rows),
        "num_edits": len(per_edit_rows),
        "per_edit": {"count": len(per_edit_rows), **_round_metrics(per_edit_metric)},
        "per_image": {"count": len(per_image_rows), **_round_metrics(per_image_metric)},
        "note": (
            "For mybench, PQ is evaluated once per image using original+edited images (without instruction text), "
            "and stored at per-image/summary level. "
            "SC (PF/Consistency) uses per-crop mask regions with per-crop instructions."
            + " Overall is computed after aggregation with sqrt(min(PF, Consistency) * PQ)."
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "granularity",
            "count",
            "prompt_following",
            "consistency",
            "perceptual_quality",
            "overall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"granularity": "per_edit", **summary["per_edit"]})
        writer.writerow({"granularity": "per_image", **summary["per_image"]})


def build_scorer(args: argparse.Namespace) -> EditScore:
    return EditScore(
        backbone=args.backbone,
        key=args.key,
        openai_url=args.openai_url,
        model_name_or_path=args.model_name_or_path,
        score_range=args.score_range,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_pass=args.num_pass,
        lora_path=args.lora_path,
        cache_dir=args.cache_dir,
    )


def main() -> None:
    args = parse_args()

    annotations = load_jsonl(args.annotations_jsonl)
    if not annotations:
        raise ValueError(f"No annotations loaded from: {args.annotations_jsonl}")

    dataset_type = infer_dataset_type(
        requested=args.dataset_type,
        annotations=annotations,
        crop_instruction_jsonl=args.crop_instruction_jsonl,
    )

    if dataset_type in {"mybench", "refedit"} and args.crop_instruction_jsonl is None:
        raise ValueError(f"{dataset_type} requires --crop-instruction-jsonl")

    instruction_count: int | None = None
    if dataset_type in {"mybench", "refedit"}:
        instruction_count = int(args.instruction_count)
        if instruction_count <= 0:
            raise ValueError("--instruction-count must be >= 1")
        print(
            f"[info] instruction-count ablation enabled: using first {instruction_count} "
            "instruction(s)/mask(s) per image."
        )

    if dataset_type == "refedit":
        samples = build_refedit_samples(
            annotations=annotations,
            crop_instruction_jsonl=args.crop_instruction_jsonl,
            input_root=args.input_image_root,
            edited_root=args.edited_image_root,
            instruction_count=instruction_count,
            max_samples=args.max_samples,
        )
    else:
        samples = build_mybench_samples(
            annotations=annotations,
            crop_instruction_jsonl=args.crop_instruction_jsonl,
            input_root=args.input_image_root,
            edited_root=args.edited_image_root,
            instruction_count=instruction_count,
            max_samples=args.max_samples,
        )

    if not samples:
        raise ValueError("No samples after filtering")

    scorer = build_scorer(args)

    output_dir = args.result_dir / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_type == "refedit":
        evaluate_refedit(
            samples=samples,
            scorer=scorer,
            output_dir=output_dir,
            save_masked_visuals=args.save_masked_visuals,
            masked_visual_dir=(
                args.masked_visual_dir
                if args.masked_visual_dir is not None
                else output_dir / "sc_inputs"
            ),
        )
    else:
        evaluate_mybench(
            samples=samples,
            scorer=scorer,
            output_dir=output_dir,
            save_masked_visuals=args.save_masked_visuals,
            masked_visual_dir=(
                args.masked_visual_dir
                if args.masked_visual_dir is not None
                else output_dir / "sc_inputs"
            ),
        )

    print(f"Done. Results are saved to: {output_dir}")


if __name__ == "__main__":
    main()
