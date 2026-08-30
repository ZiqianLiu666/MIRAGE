from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    from .prompts import (
        CONTEXT,
        PQ_RULE,
        SC_BATCH_CONTEXT,
        SC_BATCH_RULE,
    )
except ImportError:
    from prompts import (
        CONTEXT,
        PQ_RULE,
        SC_BATCH_CONTEXT,
        SC_BATCH_RULE,
    )


FLUX2_MAX_INPUT_AREA = 1024 * 1024
NEAREST = getattr(Image, "Resampling", Image).NEAREST


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


@dataclass
class SCBatchItem:
    crop_index: int
    instruction: str
    input_image: Image.Image
    edited_image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PF, consistency, and PQ with configurable OpenAI judges. "
            "All PF/consistency items for one image share one batched request; "
            "PQ uses one separate request per image."
        )
    )
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--crop-instruction-jsonl", type=Path, required=True)
    parser.add_argument("--input-image-root", type=Path, required=True)
    parser.add_argument("--edited-image-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help=(
            "Fallback model used for both SC and PQ. --sc-model and --pq-model "
            "override it independently."
        ),
    )
    parser.add_argument(
        "--sc-model",
        default=None,
        help="Model used for prompt-following and consistency evaluation.",
    )
    parser.add_argument(
        "--pq-model",
        default=None,
        help="Model used for perceptual-quality evaluation.",
    )
    parser.add_argument(
        "--mask-alignment-policy",
        choices=["source-frame", "full-frame", "flux2-crop"],
        required=True,
        help=(
            "How source-coordinate masks map to the native edited image. Use "
            "source-frame when preprocessing has restored the edited image to "
            "the exact source dimensions (the GPT Image adapter). Use "
            "full-frame when the edited image contains the complete source frame "
            "at another resolution (Qwen and the custom VLM/oracle runners). Use "
            "flux2-crop for outputs from the official Flux2/Klein or Flux2/DEV "
            "pipeline, which caps inputs at 1 MP and center-crops dimensions to "
            "multiples of 16. Images themselves are never resized or padded."
        ),
    )
    parser.add_argument(
        "--image-detail",
        choices=["auto", "low", "high"],
        default="high",
        help="OpenAI vision detail level. high is the evaluation default.",
    )
    parser.add_argument(
        "--openai-url",
        default="https://api.openai.com/v1/chat/completions",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument("--score-range", type=int, default=25)
    parser.add_argument(
        "--metrics",
        choices=["all", "sc", "pq"],
        default="all",
        help=(
            "Metrics to evaluate: all runs SC (prompt following and "
            "consistency) plus PQ; sc runs only SC; pq runs only PQ. "
            "Defaults to all."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Resume from result-dir. An image is skipped when the metrics "
            "selected by --metrics are already present."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.exists() else []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def parse_masks(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = json.loads(value)
    return value


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

        masks = parse_masks(row["mask"])
        crop_instructions = crop_map[image_name]

        samples.append(
            EvalSample(
                key=str(image_index),
                input_path=input_path,
                edited_path=edited_path,
                masks=masks,
                crop_edits=[
                    CropEdit(crop_index=index, instruction=instruction)
                    for index, instruction in enumerate(crop_instructions)
                ],
            )
        )
    return samples


def decode_polygons(
    masks: list[Any], crop_index: int
) -> list[list[int | float]]:
    value = masks[crop_index]
    pending = [value]
    polygons: list[list[int | float]] = []
    while pending:
        candidate = pending.pop(0)
        if candidate and all(
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            for coordinate in candidate
        ):
            polygons.append(candidate)
        else:
            pending[0:0] = candidate
    return polygons


def mask_decode(
    encoded_masks: list[list[int | float]], image_size: tuple[int, int]
) -> np.ndarray:
    mask_image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in encoded_masks:
        draw.polygon(polygon, outline=1, fill=1)
    return np.array(mask_image)


def open_image_pair(
    input_path: Path, edited_path: Path
) -> tuple[Image.Image, Image.Image]:
    with Image.open(input_path) as input_img:
        input_image = input_img.convert("RGB")
    with Image.open(edited_path) as edited_img:
        edited_image = edited_img.convert("RGB")
    return input_image, edited_image


def flux2_pre_crop_size(image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    if width * height <= FLUX2_MAX_INPUT_AREA:
        return image_size

    scale = math.sqrt(FLUX2_MAX_INPUT_AREA / (width * height))
    return int(width * scale), int(height * scale)


def project_mask_to_edited(
    source_mask: Image.Image,
    edited_size: tuple[int, int],
    policy: str,
) -> Image.Image:
    if policy == "source-frame":
        return source_mask.copy()

    if policy == "full-frame":
        if source_mask.size == edited_size:
            return source_mask.copy()
        return source_mask.resize(edited_size, resample=NEAREST)

    pre_crop_size = flux2_pre_crop_size(source_mask.size)
    projected = source_mask
    if projected.size != pre_crop_size:
        projected = projected.resize(pre_crop_size, resample=NEAREST)

    left = (pre_crop_size[0] - edited_size[0]) // 2
    top = (pre_crop_size[1] - edited_size[1]) // 2
    return projected.crop(
        (left, top, left + edited_size[0], top + edited_size[1])
    )


def apply_binary_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    mask_array = np.array(mask, dtype=np.uint8)
    return Image.fromarray(np.array(image) * mask_array[:, :, None])


def build_sc_pair(
    input_image: Image.Image,
    edited_image: Image.Image,
    masks: list[Any],
    crop_index: int,
    mask_alignment_policy: str,
) -> tuple[Image.Image, Image.Image]:
    polygons = decode_polygons(masks, crop_index)
    source_mask = Image.fromarray(
        mask_decode(polygons, input_image.size).astype(np.uint8)
    )
    edited_mask = project_mask_to_edited(
        source_mask=source_mask,
        edited_size=edited_image.size,
        policy=mask_alignment_policy,
    )
    return (
        apply_binary_mask(input_image, source_mask),
        apply_binary_mask(edited_image, edited_mask),
    )


def encode_pil_image(image: Image.Image) -> str:
    image_stream = BytesIO()
    image.save(image_stream, format="PNG")
    encoded = base64.b64encode(image_stream.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class GPTScorer:
    def __init__(
        self,
        key: str,
        url: str,
        model_name: str,
        score_range: int,
        image_detail: str,
        sc_model_name: str | None = None,
        pq_model_name: str | None = None,
    ) -> None:
        self.key = key
        self.url = url
        self.model_name = model_name
        self.sc_model_name = sc_model_name or model_name
        self.pq_model_name = pq_model_name or model_name
        self.score_range = score_range
        self.image_detail = image_detail
        self.sc_request_count = 0
        self.pq_request_count = 0
        self.pq_prompt = "\n".join(
            [CONTEXT, PQ_RULE.replace("10", str(score_range))]
        )
        self.sc_batch_prompt = "\n".join(
            [
                SC_BATCH_CONTEXT,
                SC_BATCH_RULE.replace("<score_range>", str(score_range)),
            ]
        )

    def evaluate_pq(self, images: list[Image.Image]) -> dict[str, Any]:
        result = self._call(
            images,
            self.pq_prompt,
            request_kind="pq",
        )
        scale = self.score_range / 10
        return {
            "perceptual_quality": min(result["score"]) / scale,
            "PQ_reasoning": result["reasoning"],
        }

    def evaluate_sc_batch(
        self, items: list[SCBatchItem]
    ) -> dict[int, dict[str, Any]]:
        crop_indices = [item.crop_index for item in items]

        prompt_content: list[dict[str, Any]] = [
            {"type": "text", "text": self.sc_batch_prompt}
        ]
        for item in items:
            prompt_content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"EDIT ITEM crop_index={item.crop_index}\n"
                            f"Editing instruction: {item.instruction}\n"
                            "The next image is this item's original masked image."
                        ),
                    },
                    self._image_content(item.input_image),
                    {
                        "type": "text",
                        "text": (
                            f"The next image is crop_index={item.crop_index}'s "
                            "edited masked image."
                        ),
                    },
                    self._image_content(item.edited_image),
                ]
            )

        item_schema = {
            "type": "object",
            "properties": {
                "crop_index": {"type": "integer", "enum": crop_indices},
                "prompt_following": {"type": "number"},
                "consistency": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": [
                "crop_index",
                "prompt_following",
                "consistency",
                "reasoning",
            ],
            "additionalProperties": False,
        }
        schema = {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": item_schema,
                }
            },
            "required": ["edits"],
            "additionalProperties": False,
        }
        output = self._request(
            prompt_content=prompt_content,
            schema=schema,
            schema_name="batched_edit_evaluation",
            request_kind="sc",
        )

        raw_edits = output["edits"]

        parsed: dict[int, dict[str, Any]] = {}
        scale = self.score_range / 10
        for row in raw_edits:
            crop_index = int(row["crop_index"])
            prompt_following = float(row["prompt_following"])
            consistency = float(row["consistency"])
            parsed[crop_index] = {
                "prompt_following": prompt_following / scale,
                "consistency": consistency / scale,
                "SC_reasoning": str(row["reasoning"]),
            }

        return parsed

    def _image_content(self, image: Image.Image) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {
                "url": encode_pil_image(image),
                "detail": self.image_detail,
            },
        }

    def _call(
        self,
        images: list[Image.Image],
        prompt: str,
        request_kind: str = "sc",
    ) -> dict[str, Any]:
        prompt_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]
        prompt_content.extend(self._image_content(image) for image in images)
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "score": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        }
        output = self._request(
            prompt_content=prompt_content,
            schema=schema,
            schema_name="edit_evaluation",
            request_kind=request_kind,
        )
        output["score"] = [float(score) for score in output["score"]]
        return output

    def _request(
        self,
        prompt_content: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str,
        request_kind: str,
    ) -> dict[str, Any]:
        if request_kind == "sc":
            self.sc_request_count += 1
            request_model = self.sc_model_name
        else:
            self.pq_request_count += 1
            request_model = self.pq_model_name

        payload = {
            "model": request_model,
            "messages": [{"role": "user", "content": prompt_content}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        response = requests.post(
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            },
            timeout=180,
        )
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])


def metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows))


def round_metrics(
    metrics: dict[str, float | None], ndigits: int = 4
) -> dict[str, float | None]:
    return {
        key: None if value is None else round(float(value), ndigits)
        for key, value in metrics.items()
    }


def row_has_sc(row: dict[str, Any]) -> bool:
    return all(
        row.get(key) is not None
        for key in ("prompt_following", "consistency", "SC_reasoning")
    )


def row_has_pq(row: dict[str, Any]) -> bool:
    return all(
        row.get(key) is not None
        for key in ("perceptual_quality", "PQ_reasoning")
    )


def row_has_selected_metrics(row: dict[str, Any], metrics: str) -> bool:
    if metrics == "sc":
        return row_has_sc(row)
    if metrics == "pq":
        return row_has_pq(row)
    return row_has_sc(row) and row_has_pq(row)


def evaluate_mybench(
    samples: list[EvalSample],
    scorer: GPTScorer,
    output_dir: Path,
    mask_alignment_policy: str,
    skip_existing: bool = False,
    metrics: str = "all",
) -> None:
    evaluate_sc = metrics in {"all", "sc"}
    evaluate_pq = metrics in {"all", "pq"}
    sc_model_name = getattr(
        scorer, "sc_model_name", getattr(scorer, "model_name", None)
    )
    pq_model_name = getattr(
        scorer, "pq_model_name", getattr(scorer, "model_name", None)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_edit_path = output_dir / "per_edit_results.jsonl"
    per_image_path = output_dir / "per_image_results.jsonl"
    sample_keys = {sample.key for sample in samples}

    if skip_existing:
        per_image_by_key: dict[str, dict[str, Any]] = {}
        for loaded_row in load_jsonl_if_exists(per_image_path):
            key = str(loaded_row["key"])
            if key not in sample_keys:
                continue
            per_image_by_key[key] = {
                **per_image_by_key.get(key, {}),
                **loaded_row,
            }

        sc_completed_keys = {
            key for key, row in per_image_by_key.items() if row_has_sc(row)
        }
        per_edit_rows = [
            row
            for row in load_jsonl_if_exists(per_edit_path)
            if str(row["image_key"]) in sc_completed_keys
            and str(row["image_key"]) in sample_keys
        ]
    else:
        per_edit_rows = []
        per_image_by_key = {}

    completed_keys = {
        key
        for key, row in per_image_by_key.items()
        if row_has_selected_metrics(row, metrics)
    }
    pending_samples = [
        sample for sample in samples if sample.key not in completed_keys
    ]

    write_jsonl(per_edit_path, per_edit_rows)
    write_jsonl(
        per_image_path,
        [
            per_image_by_key[sample.key]
            for sample in samples
            if sample.key in per_image_by_key
        ],
    )
    if skip_existing:
        print(
            f"Resume ({metrics}): skipping {len(completed_keys)}/"
            f"{len(samples)} images; evaluating {len(pending_samples)}."
        )

    for sample in tqdm(pending_samples, desc="gptscore-eval"):
        input_image, edited_image = open_image_pair(
            sample.input_path, sample.edited_path
        )
        existing_image_row = per_image_by_key.get(sample.key, {})
        needs_sc = evaluate_sc and not row_has_sc(existing_image_row)
        needs_pq = evaluate_pq and not row_has_pq(existing_image_row)

        sc_batch: list[SCBatchItem] = []
        if needs_sc:
            for crop_edit in sample.crop_edits:
                sc_input, sc_edited = build_sc_pair(
                    input_image=input_image,
                    edited_image=edited_image,
                    masks=sample.masks,
                    crop_index=crop_edit.crop_index,
                    mask_alignment_policy=mask_alignment_policy,
                )
                sc_batch.append(
                    SCBatchItem(
                        crop_index=crop_edit.crop_index,
                        instruction=crop_edit.instruction,
                        input_image=sc_input,
                        edited_image=sc_edited,
                    )
                )
        pq_result = (
            scorer.evaluate_pq([input_image, edited_image])
            if needs_pq
            else None
        )
        sc_results = scorer.evaluate_sc_batch(sc_batch) if needs_sc else None

        if needs_sc:
            this_image_edit_rows: list[dict[str, Any]] = []
            per_edit_rows = [
                row
                for row in per_edit_rows
                if str(row["image_key"]) != sample.key
            ]
            for crop_edit in sample.crop_edits:
                sc_result = sc_results[crop_edit.crop_index]
                per_edit_row = {
                    "key": f"{sample.key}__{crop_edit.crop_index}",
                    "image_key": sample.key,
                    "crop_index": crop_edit.crop_index,
                    "mask_alignment_policy": mask_alignment_policy,
                    "input_size": list(input_image.size),
                    "edited_size": list(edited_image.size),
                    "sc_model": sc_model_name,
                    "prompt_following": float(sc_result["prompt_following"]),
                    "consistency": float(sc_result["consistency"]),
                    "SC_reasoning": str(sc_result["SC_reasoning"]),
                }
                per_edit_rows.append(per_edit_row)
                this_image_edit_rows.append(per_edit_row)
            write_jsonl(per_edit_path, per_edit_rows)
        else:
            this_image_edit_rows = [
                row
                for row in per_edit_rows
                if str(row["image_key"]) == sample.key
            ]

        per_image_row = {
            **existing_image_row,
            "key": sample.key,
            "num_crops": len(sample.crop_edits),
            "mask_alignment_policy": mask_alignment_policy,
            "input_size": list(input_image.size),
            "edited_size": list(edited_image.size),
        }
        if needs_sc:
            per_image_row.update(
                {
                    "sc_model": sc_model_name,
                    "num_sc_evals": len(this_image_edit_rows),
                    "prompt_following": metric_mean(
                        this_image_edit_rows, "prompt_following"
                    ),
                    "consistency": metric_mean(
                        this_image_edit_rows, "consistency"
                    ),
                    "SC_reasoning": [
                        row["SC_reasoning"] for row in this_image_edit_rows
                    ],
                }
            )
        if needs_pq:
            per_image_row.update(
                {
                    "pq_model": pq_model_name,
                    "perceptual_quality": float(
                        pq_result["perceptual_quality"]
                    ),
                    "PQ_reasoning": str(pq_result["PQ_reasoning"]),
                }
            )
        if row_has_sc(per_image_row) and row_has_pq(per_image_row):
            per_image_row["overall"] = math.sqrt(
                min(
                    float(per_image_row["prompt_following"]),
                    float(per_image_row["consistency"]),
                )
                * float(per_image_row["perceptual_quality"])
            )
        else:
            per_image_row.pop("overall", None)

        per_image_by_key[sample.key] = per_image_row
        write_jsonl(
            per_image_path,
            [
                per_image_by_key[ordered_sample.key]
                for ordered_sample in samples
                if ordered_sample.key in per_image_by_key
            ],
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

    per_image_rows = [
        per_image_by_key[sample.key]
        for sample in samples
        if sample.key in per_image_by_key
    ]
    selected_image_rows = [
        row
        for row in per_image_rows
        if row_has_selected_metrics(row, metrics)
    ]
    sc_image_rows = [row for row in per_image_rows if row_has_sc(row)]
    pq_image_rows = [row for row in per_image_rows if row_has_pq(row)]

    empty_metric: dict[str, float | None] = {
        "prompt_following": None,
        "consistency": None,
        "perceptual_quality": None,
        "overall": None,
    }
    per_edit_metric = dict(empty_metric)
    per_image_metric = dict(empty_metric)
    if evaluate_sc:
        per_edit_metric.update(summarize_sc(per_edit_rows))
        per_image_metric.update(summarize_sc(sc_image_rows))
    if evaluate_pq:
        dataset_pq = metric_mean(pq_image_rows, "perceptual_quality")
        per_image_metric["perceptual_quality"] = dataset_pq
        if evaluate_sc:
            per_edit_metric = add_overall(
                {
                    "prompt_following": float(
                        per_edit_metric["prompt_following"]
                    ),
                    "consistency": float(per_edit_metric["consistency"]),
                },
                dataset_pq,
            )
            per_image_metric = add_overall(
                {
                    "prompt_following": float(
                        per_image_metric["prompt_following"]
                    ),
                    "consistency": float(per_image_metric["consistency"]),
                },
                dataset_pq,
            )

    selected_edit_count = len(per_edit_rows) if evaluate_sc else 0
    summary = {
        "dataset_type": "mybench",
        "metrics": metrics,
        "mask_alignment_policy": mask_alignment_policy,
        "image_encoding": "png",
        "image_detail": scorer.image_detail,
        "models": {
            "sc": sc_model_name if evaluate_sc else None,
            "pq": pq_model_name if evaluate_pq else None,
        },
        "api_requests": {
            "sc": scorer.sc_request_count,
            "pq": scorer.pq_request_count,
            "total": scorer.sc_request_count + scorer.pq_request_count,
        },
        "resume": {
            "enabled": skip_existing,
            "skipped_images": len(completed_keys),
            "evaluated_images": len(pending_samples),
        },
        "num_images": len(selected_image_rows),
        "num_edits": selected_edit_count,
        "per_edit": {
            "count": selected_edit_count,
            **round_metrics(per_edit_metric),
        },
        "per_image": {
            "count": len(selected_image_rows),
            **round_metrics(per_image_metric),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
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

    samples = build_mybench_samples(
        annotations=annotations,
        crop_instruction_jsonl=args.crop_instruction_jsonl,
        input_root=args.input_image_root,
        edited_root=args.edited_image_root,
    )
    key = args.key or os.environ.get("OPENAI_API_KEY")

    scorer = GPTScorer(
        key=key or "",
        url=args.openai_url,
        model_name=args.model,
        score_range=args.score_range,
        image_detail=args.image_detail,
        sc_model_name=args.sc_model,
        pq_model_name=args.pq_model,
    )
    evaluate_mybench(
        samples,
        scorer,
        args.result_dir,
        mask_alignment_policy=args.mask_alignment_policy,
        skip_existing=args.skip_existing,
        metrics=args.metrics,
    )
    print(f"Done. Results are saved to: {args.result_dir}")


if __name__ == "__main__":
    main()
