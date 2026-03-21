import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from PIL import Image
import utils.qwen_utils as vlm

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

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
        help="Output JSONL path.",
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

    return parser.parse_args()

def load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_instruction_map(jsonl_path: str) -> Dict[str, str]:
    inst_map: Dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            inst_map[rec["image"]] = rec["editing_instruction"]
    return inst_map


def write_jsonl(path: str, records: List[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)


def extract_item_fields(item: dict) -> tuple[str, str]:
    return item["Refer_object"].strip(), item["New_edit_instruction"].strip()


def build_records_from_items(image_name: str, items: List[dict]) -> List[dict]:
    records = []
    for item_idx, item in enumerate(items, start=1):
        refer_object, new_inst = extract_item_fields(item)
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
) -> List[dict]:
    image_names = sorted(
        [f for f in os.listdir(image_root) if f.lower().endswith(IMAGE_EXTS)]
    )
    print(f"Found {len(image_names)} images in {image_root}")

    all_records: List[dict] = []
    total = len(image_names)
    for batch_start in range(0, total, batch_size):
        batch_names = image_names[batch_start : batch_start + batch_size]
        print(
            f"\n=== Batch [{batch_start + 1}-{batch_start + len(batch_names)}/{total}] ==="
        )

        instructions = [inst_map[img_name] for img_name in batch_names]
        items_list = vlm.parse_edit_instruction_batch(instructions)

        batch_records: List[dict] = []
        batch_tasks = []

        for img_name, instruction, items in zip(batch_names, instructions, items_list):
            print(f"\n=== Processing image: {img_name} ===")
            print(f"  Instruction: {instruction}")
            print(f"  Parsed items: {items}")

            image_path = os.path.join(image_root, img_name)
            image = load_image(image_path)

            image_records = build_records_from_items(img_name, items)
            batch_records.extend(image_records)
            for record_index, rec in enumerate(image_records, start=1):
                batch_tasks.append(
                    {
                        "image_name": img_name,
                        "image": image,
                        "record": rec,
                        "record_index": record_index,
                    }
                )

        bboxes_list = vlm.locate_refer_object_batch(
            [task["image"] for task in batch_tasks],
            [task["record"]["refer_object"] for task in batch_tasks],
        )

        for task, bboxes in zip(batch_tasks, bboxes_list):
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

    return all_records


def main():
    args = parse_args()

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
        inst_map = load_instruction_map(args.instruction_jsonl)
        print(f"Loaded {len(inst_map)} instructions from {args.instruction_jsonl}")
        process_batch(
            image_root=args.image_root,
            inst_map=inst_map,
            base_crop_dir=base_crop_dir,
            output_jsonl=output_jsonl,
            batch_size=args.batch_size,
            padding=args.padding,
        )

    print(f"JSONL saved to: {output_jsonl}")


if __name__ == "__main__":
    main()
