import argparse
import json
import os
import tempfile
import time

import torch
from PIL import Image
from tqdm import tqdm

from mirage.vlm import BACKENDS, load_vlm, parse_instructions, to_pixels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split each editing instruction into local edits and localize their targets with a VLM."
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--instruction-jsonl", required=True)
    parser.add_argument("--output-dir", required=True, help="where the crops are saved")
    parser.add_argument("--out-jsonl", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="reuse saved crop records with existing PNG files and recompute missing crops")
    mode.add_argument("--overwrite", dest="resume", action="store_false", help="recompute all images (default)")
    parser.set_defaults(resume=False)
    parser.add_argument("--vlm", default="qwen35", choices=BACKENDS)
    parser.add_argument("--vlm-model-id", help="override the default checkpoint of --vlm")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="random seed for model sampling")
    parser.add_argument("--padding", type=int, default=10, help="pixels added around each box")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def crops_complete(records, expected_count=None):
    """Only skip an image when every expected crop is present."""
    return (
        bool(records)
        and (expected_count is None or len(records) == expected_count)
        and all(record.get("refer_object") and record.get("new_instruction") for record in records)
        and all(record.get("num_crops") == len(records) for record in records)
        and [record["image"] for record in records] == [f"crop_{k:02d}.png" for k in range(1, len(records) + 1)]
    )


def load_saved(out_jsonl, output_dir):
    """Recover records with existing crop files without inspecting image contents."""
    if not os.path.isfile(out_jsonl):
        return {}
    with open(out_jsonl, encoding="utf-8") as f:
        lines = f.readlines()
    grouped = {}
    last_name = None
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if lineno == len(lines):
                print(f"Ignoring incomplete final JSONL line in {out_jsonl}.")
                break
            raise ValueError(f"Invalid JSON in {out_jsonl}:{lineno}") from exc
        last_name = record["original_image"]
        grouped.setdefault(last_name, []).append(record)

    saved = {}
    for name, records in grouped.items():
        crop_names = [f"crop_{k:02d}.png" for k in range(1, len(records) + 1)]
        legacy = all("num_crops" not in r for r in records)
        if legacy and name == last_name:
            print(f"Rechecking final legacy image without a completion count: {name}")
        elif legacy and [r["image"] for r in records] == crop_names:
            # Earlier legacy groups were fully written before the next image began.
            records = [{**record, "num_crops": len(records)} for record in records]
        valid = {}
        for record in records:
            if not os.path.isfile(os.path.join(output_dir, name, record["image"])):
                continue
            valid[record["image"]] = record
        if valid:
            saved[name] = list(valid.values())
    return saved


def save_completed(out_jsonl, saved):
    """Atomically save crop records, including an incomplete final image."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=os.path.dirname(os.path.abspath(out_jsonl)),
            prefix=f".{os.path.basename(out_jsonl)}.", suffix=".tmp", delete=False,
        ) as f:
            temporary_path = f.name
            for name in sorted(saved):
                for record in saved[name]:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, out_jsonl)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    with open(args.instruction_jsonl, encoding="utf-8") as f:
        annotations = {r["image"]: r for r in (json.loads(line) for line in f)}
    instructions = {name: record["editing_instruction"] for name, record in annotations.items()}
    expected_counts = {
        name: len(record["mask"]) if isinstance(record.get("mask"), list) and record["mask"] else None
        for name, record in annotations.items()
    }
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    saved = load_saved(args.out_jsonl, args.output_dir) if args.resume else {}
    saved = {name: records for name, records in saved.items() if name in instructions}
    completed = {name for name, records in saved.items() if crops_complete(records, expected_counts[name])}
    names = sorted(name for name in instructions if name not in completed)
    save_completed(args.out_jsonl, saved)
    print(f"Skipping {len(completed)} completed images; {len(names)} images remaining.")
    if not names:
        return
    torch.manual_seed(args.seed)
    vlm = load_vlm(args.vlm, args.vlm_model_id, args.device, getattr(torch, args.dtype))

    parse_time = locate_time = 0.0
    for start in tqdm(range(0, len(names), args.batch_size)):
        batch = names[start : start + args.batch_size]
        images = {name: Image.open(os.path.join(args.image_root, name)).convert("RGB") for name in batch}

        tic = time.perf_counter()
        parsed = parse_instructions(vlm, [instructions[name] for name in batch])
        edits = []
        for name, pairs in zip(batch, parsed, strict=True):
            expected_count = expected_counts[name]
            if expected_count is not None and len(pairs) != expected_count:
                raise ValueError(
                    f"{args.vlm} extracted {len(pairs)} edits for {name}, "
                    f"but annotations expect {expected_count}."
                )
            existing = {
                record["image"]: record for record in saved.get(name, [])
                if record.get("refer_object") and record.get("new_instruction")
            }
            expected = {f"crop_{k:02d}.png": pair for k, pair in enumerate(pairs, start=1)}
            for crop_name, record in existing.items():
                if (
                    crop_name not in expected
                    or tuple(expected[crop_name]) != (record["refer_object"], record["new_instruction"])
                    or (expected_count is None and record.get("num_crops", len(pairs)) != len(pairs))
                ):
                    raise ValueError(
                        f"Saved {name}/{crop_name} does not match the current instruction decomposition. "
                        "Use a new output directory or --overwrite to avoid mixing results."
                    )
                record["num_crops"] = len(pairs)
            edits.extend(
                (name, k, ref, inst, len(pairs))
                for k, (ref, inst) in enumerate(pairs, start=1)
                if f"crop_{k:02d}.png" not in existing
            )
        toc = time.perf_counter()
        parse_time += toc - tic
        if not edits:
            save_completed(args.out_jsonl, saved)
            continue
        try:
            boxes = vlm.iter_locate([images[name] for name, *_ in edits], [ref for _, _, ref, _, _ in edits])
            for (name, k, ref, inst, total), box in zip(edits, boxes, strict=True):
                x1, y1, x2, y2 = to_pixels(box, images[name].size, args.padding)
                crop_name = f"crop_{k:02d}.png"
                os.makedirs(os.path.join(args.output_dir, name), exist_ok=True)
                images[name].crop((x1, y1, x2, y2)).save(os.path.join(args.output_dir, name, crop_name))
                record = {
                    "original_image": name,
                    "image": crop_name,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "refer_object": ref,
                    "new_instruction": inst,
                    "num_crops": total,
                }
                saved[name] = [row for row in saved.get(name, []) if row["image"] != crop_name] + [record]
                saved[name].sort(key=lambda record: int(record["image"][5:-4]))
                save_completed(args.out_jsonl, saved)
        except ValueError as exc:
            raise ValueError(f"Localization failed for images {batch}: {exc}") from exc
        locate_time += time.perf_counter() - toc
    print(f"per image: parsing {parse_time / len(names):.2f}s, localization {locate_time / len(names):.2f}s")


if __name__ == "__main__":
    main()
