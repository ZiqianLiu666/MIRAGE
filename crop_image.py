import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image
import utils.vlm_utils as vlm

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_vlm_name(value: str) -> str:
    aliases = {
        "qwen": "qwen8b",
        "qwen3": "qwen8b",
        "qwen8b": "qwen8b",
        "qwen4": "qwen4b",
        "qwen4b": "qwen4b",
        "gemma4": "gemma4",
        "qwen35": "qwen35",
        "regionreasoner": "regionreasoner",
    }
    return aliases[str(value).lower().strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate refer_object/new_instruction, bbox, and crops."
    )

    parser.add_argument(
        "--image-root",
        default=None,
        help="Folder containing images (batch mode).",
    )
    parser.add_argument(
        "--instruction-jsonl",
        default=None,
        help="JSONL file with image -> editing_instruction mapping.",
    )
    parser.add_argument(
        "--out-jsonl",
        required=True,
        help=(
            "Output JSONL path. In batch mode, existing complete records are "
            "used as a resume checkpoint."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from complete crop records and only localize missing "
            "items. Enabled by default; use --no-skip-existing to rebuild "
            "the output JSONL from scratch."
        ),
    )

    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to a single image. If set, single-image mode is used.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction string for single-image mode.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dir for cropped images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for VLM inference in batch mode.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding (pixels) to add around bbox.",
    )

    parser.add_argument(
        "--vlm",
        type=parse_vlm_name,
        choices=["gemma4", "qwen8b", "qwen4b", "qwen35", "regionreasoner"],
        default="gemma4",
        help=(
            "VLM used for instruction parsing and visual grounding. "
            "Options: gemma4, qwen8b (Qwen3-VL-8B), qwen4b "
            "(Qwen3-VL-4B), qwen35, regionreasoner. RegionReasoner uses "
            "the same checkpoint for parsing and grounding. "
            "Default: gemma4 (google/gemma-4-12B-it)."
        ),
    )
    parser.add_argument(
        "--vlm-model-id",
        default=None,
        help=(
            "Optional Hugging Face model id/path override. Defaults to "
            "google/gemma-4-12B-it for gemma4, "
            "Qwen/Qwen3-VL-8B-Instruct for qwen8b, "
            "Qwen/Qwen3-VL-4B-Instruct for qwen4b, "
            "Qwen/Qwen3.5-9B for qwen35, and "
            "lmsdss/RegionReasoner-7B for regionreasoner."
        ),
    )
    parser.add_argument(
        "--vlm-device",
        default="cuda:0",
        help="Torch device used by the VLM. Default: cuda:0.",
    )
    parser.add_argument(
        "--vlm-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="VLM inference dtype. Default: bf16.",
    )
    parser.add_argument(
        "--gemma-visual-tokens",
        type=int,
        choices=[70, 140, 280, 560, 1120],
        default=1120,
        help=(
            "Gemma 4 visual token budget. Higher is better for fine-grained "
            "grounding; default: 1120."
        ),
    )

    return parser.parse_args()

def load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_instruction_data(
    jsonl_path: str,
) -> tuple[Dict[str, str], Dict[str, int]]:
    inst_map: Dict[str, str] = {}
    expected_edit_counts: Dict[str, int] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            image_name = rec["image"]
            inst_map[image_name] = rec["editing_instruction"]
            masks = rec.get("mask")
            if isinstance(masks, list) and masks:
                expected_edit_counts[image_name] = len(masks)
    return inst_map, expected_edit_counts


def write_jsonl(path: str, records: List[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)


def load_existing_records(path: str) -> List[dict]:
    input_path = Path(path)
    if not input_path.exists():
        return []

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _record_has_complete_crop(record: dict, base_crop_dir: str) -> bool:
    image_name = record.get("original_image")
    crop_name = record.get("image")
    bbox = record.get("bbox")
    if not isinstance(image_name, str) or not image_name:
        return False
    if not isinstance(crop_name, str) or not crop_name:
        return False
    if (
        not isinstance(record.get("refer_object"), str)
        or not record["refer_object"]
    ):
        return False
    if (
        not isinstance(record.get("new_instruction"), str)
        or not record["new_instruction"]
    ):
        return False
    if not isinstance(bbox, dict) or not all(
        key in bbox for key in ("x1", "y1", "x2", "y2")
    ):
        return False
    if not all(
        isinstance(bbox[key], (int, float)) and not isinstance(bbox[key], bool)
        for key in ("x1", "y1", "x2", "y2")
    ):
        return False

    crop_path = Path(base_crop_dir) / image_name / crop_name
    return crop_path.is_file()


def prepare_resume_state(
    output_jsonl: str,
    base_crop_dir: str,
    source_image_names: List[str],
    expected_edit_counts: Dict[str, int] | None = None,
) -> tuple[List[dict], set[str]]:
    expected_edit_counts = expected_edit_counts or {}
    existing_records = load_existing_records(output_jsonl)
    if not existing_records:
        return [], set()

    records_by_image: Dict[str, List[dict]] = {}
    for record in existing_records:
        image_name = record["original_image"]
        records_by_image.setdefault(image_name, []).append(record)

    source_images = set(source_image_names)
    completed_images = set()
    incomplete_images = set()
    for image_name, image_records in records_by_image.items():
        if image_name not in source_images:
            continue
        crop_names = [record.get("image") for record in image_records]
        has_unique_crops = all(
            isinstance(crop_name, str) and crop_name for crop_name in crop_names
        ) and len(crop_names) == len(set(crop_names))
        expected_count = expected_edit_counts.get(image_name)
        has_expected_count = (
            expected_count is None or len(image_records) == expected_count
        )
        if has_expected_count and has_unique_crops and all(
            _record_has_complete_crop(record, base_crop_dir)
            for record in image_records
        ):
            completed_images.add(image_name)
        else:
            incomplete_images.add(image_name)

    retained_records = [
        record
        for record in existing_records
        if _record_has_complete_crop(record, base_crop_dir)
    ]

    print(
        f"Resume checkpoint: loaded {len(existing_records)} records; "
        f"skipping {len(completed_images)} complete images."
    )
    if incomplete_images:
        print(
            "Resume checkpoint: filling missing items in incomplete images: "
            + ", ".join(sorted(incomplete_images))
        )

    return retained_records, completed_images


def build_records_from_items(image_name: str, items: List[dict]) -> List[dict]:
    records = []
    for item_idx, item in enumerate(items, start=1):
        refer_object = item["Refer_object"].strip()
        new_inst = item["New_edit_instruction"].strip()
        records.append(
            {
                "original_image": image_name,
                "image": f"crop_{item_idx:02d}.png",
                "bbox": None,
                "refer_object": refer_object,
                "new_instruction": new_inst,
            }
        )
    return records


def parse_single_image_records(
    image_path: str,
    instruction: str,
) -> List[dict]:
    image_name = os.path.basename(image_path)
    print(f"Image: {image_path}")
    print(f"Instruction: {instruction}")
    items = vlm.parse_edit_instruction(instruction)
    print(f"Parsed items: {items}")
    return build_records_from_items(image_name, items)


def locate_and_crop_records(
    image: Image.Image,
    image_name: str,
    records: List[dict],
    base_crop_dir: str,
    padding: int,
) -> None:
    crop_dir = os.path.join(base_crop_dir, image_name)
    os.makedirs(crop_dir, exist_ok=True)

    for record_index, rec in enumerate(records, start=1):
        refer_object = rec["refer_object"]
        bboxes = vlm.locate_refer_object(image, refer_object)

        result = vlm.crop_with_bbox(
            image_input=image,
            bbox=bboxes[0],
            crop_dir=crop_dir,
            index=record_index,
            padding=padding,
        )

        save_path, bbox = result
        print(f"    -> Saved crop: {save_path}")
        rec["image"] = os.path.basename(save_path)
        rec["bbox"] = {
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
        }


def process_single_image(
    image_path: str,
    instruction: str,
    base_crop_dir: str,
    padding: int,
) -> List[dict]:
    records = parse_single_image_records(
        image_path=image_path,
        instruction=instruction,
    )
    image = load_image(image_path)
    image_name = os.path.basename(image_path)
    locate_and_crop_records(
        image=image,
        image_name=image_name,
        records=records,
        base_crop_dir=base_crop_dir,
        padding=padding,
    )
    return records


def process_batch(
    image_root: str,
    inst_map: Dict[str, str],
    base_crop_dir: str,
    output_jsonl: str,
    batch_size: int = 1,
    padding: int = 10,
    expected_edit_counts: Dict[str, int] | None = None,
    skip_existing: bool = True,
) -> List[dict]:
    image_names = sorted(
        [f for f in os.listdir(image_root) if f.lower().endswith(IMAGE_EXTS)]
    )
    print(f"Found {len(image_names)} images in {image_root}")

    if skip_existing:
        all_records, completed_images = prepare_resume_state(
            output_jsonl=output_jsonl,
            base_crop_dir=base_crop_dir,
            source_image_names=image_names,
            expected_edit_counts=expected_edit_counts,
        )
    else:
        all_records, completed_images = [], set()
        print("Existing-record skipping disabled; rebuilding all crop records.")

    existing_by_slot = {
        (record["original_image"], record["image"]): record
        for record in all_records
    }
    pending_image_names = [
        image_name for image_name in image_names if image_name not in completed_images
    ]
    if not pending_image_names:
        print("All images are already complete; nothing to process.")
        return all_records

    print(
        f"Processing {len(pending_image_names)} remaining images "
        f"({len(completed_images)} skipped)."
    )
    total = len(pending_image_names)
    for batch_start in range(0, total, batch_size):
        batch_names = pending_image_names[batch_start : batch_start + batch_size]
        print(
            f"\n=== Remaining batch "
            f"[{batch_start + 1}-{batch_start + len(batch_names)}/{total}] ==="
        )

        instructions = [inst_map[img_name] for img_name in batch_names]
        parse_start = time.perf_counter()
        items_list = vlm.parse_edit_instruction_batch(instructions)
        parse_seconds = time.perf_counter() - parse_start

        batch_records: List[dict] = []
        batch_tasks = []

        for img_name, instruction, items in zip(batch_names, instructions, items_list):
            print(f"\n=== Processing image: {img_name} ===")
            print(f"  Instruction: {instruction}")
            print(f"  Parsed items: {items}")

            image_path = os.path.join(image_root, img_name)
            image = load_image(image_path)

            image_records = build_records_from_items(img_name, items)
            for record_index, rec in enumerate(image_records, start=1):
                record_key = (img_name, rec["image"])
                if skip_existing and record_key in existing_by_slot:
                    print(
                        f"    -> Skipping existing crop: "
                        f"{img_name}/{rec['image']}"
                    )
                    continue

                batch_records.append(rec)
                batch_tasks.append(
                    {
                        "image_name": img_name,
                        "image": image,
                        "record": rec,
                        "record_index": record_index,
                    }
                )

        detect_start = time.perf_counter()
        if batch_tasks:
            bboxes_list = vlm.locate_refer_object_batch(
                [task["image"] for task in batch_tasks],
                [task["record"]["refer_object"] for task in batch_tasks],
            )
        else:
            bboxes_list = []
        detect_seconds = time.perf_counter() - detect_start

        for task, bboxes in zip(batch_tasks, bboxes_list):
            if not bboxes:
                refer_object = task["record"]["refer_object"]
                print(
                    f"    -> WARNING: no bbox found for '{refer_object}' in "
                    f"{task['image_name']}; skipping"
                )
                task["record"]["image"] = None
                task["record"]["bbox"] = None
                continue

            result = vlm.crop_with_bbox(
                image_input=task["image"],
                bbox=bboxes[0],
                crop_dir=os.path.join(base_crop_dir, task["image_name"]),
                index=task["record_index"],
                padding=padding,
            )
            save_path, bbox = result
            print(f"    -> Saved crop: {save_path}")
            task["record"]["image"] = os.path.basename(save_path)
            task["record"]["bbox"] = {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
            }

        all_records.extend(batch_records)
        write_jsonl(output_jsonl, all_records)
        batch_number = batch_start // batch_size + 1
        print(
            f"[Timing] Batch {batch_number}: "
            f"parse={parse_seconds:.3f}s, detect={detect_seconds:.3f}s"
        )

    return all_records


def main():
    args = parse_args()

    vlm.configure_backend(
        name=args.vlm,
        model_id=args.vlm_model_id,
        device=args.vlm_device,
        dtype=args.vlm_dtype,
        gemma_visual_tokens=args.gemma_visual_tokens,
    )
    print(
        f"VLM backend: {args.vlm}; "
        f"model override: {args.vlm_model_id or 'default'}; "
        f"device: {args.vlm_device}; dtype: {args.vlm_dtype}"
    )
    base_crop_dir = str(Path(args.output_dir).resolve())
    os.makedirs(base_crop_dir, exist_ok=True)
    output_jsonl = str(Path(args.out_jsonl).resolve())

    if args.image_path:
        records = process_single_image(
            image_path=args.image_path,
            instruction=args.instruction,
            base_crop_dir=base_crop_dir,
            padding=args.padding,
        )
        write_jsonl(output_jsonl, records)
    else:
        inst_map, expected_edit_counts = load_instruction_data(
            args.instruction_jsonl
        )
        print(f"Loaded {len(inst_map)} instructions from {args.instruction_jsonl}")
        process_batch(
            image_root=args.image_root,
            inst_map=inst_map,
            base_crop_dir=base_crop_dir,
            output_jsonl=output_jsonl,
            batch_size=args.batch_size,
            padding=args.padding,
            expected_edit_counts=expected_edit_counts,
            skip_existing=args.skip_existing,
        )

    print(f"JSONL saved to: {output_jsonl}")


if __name__ == "__main__":
    main()
