import argparse
import json
import os
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
    parser.add_argument("--vlm", default="qwen35", choices=BACKENDS)
    parser.add_argument("--vlm-model-id", help="override the default checkpoint of --vlm")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--padding", type=int, default=10, help="pixels added around each box")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.instruction_jsonl, encoding="utf-8") as f:
        instructions = {r["image"]: r["editing_instruction"] for r in (json.loads(line) for line in f)}
    names = sorted(instructions)
    vlm = load_vlm(args.vlm, args.vlm_model_id, args.device, getattr(torch, args.dtype))

    parse_time = locate_time = 0.0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(names), args.batch_size)):
            batch = names[start : start + args.batch_size]
            images = {name: Image.open(os.path.join(args.image_root, name)).convert("RGB") for name in batch}

            tic = time.perf_counter()
            parsed = parse_instructions(vlm, [instructions[name] for name in batch])
            edits = [
                (name, k, ref, inst)
                for name, pairs in zip(batch, parsed)
                for k, (ref, inst) in enumerate(pairs, start=1)
            ]
            toc = time.perf_counter()
            boxes = vlm.locate([images[name] for name, *_ in edits], [ref for _, _, ref, _ in edits])
            parse_time += toc - tic
            locate_time += time.perf_counter() - toc

            for (name, k, ref, inst), box in zip(edits, boxes):
                if box is None:
                    raise ValueError(f"{args.vlm} found no box for '{ref}' in {name}")
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
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"per image: parsing {parse_time / len(names):.2f}s, localization {locate_time / len(names):.2f}s")


if __name__ == "__main__":
    main()
