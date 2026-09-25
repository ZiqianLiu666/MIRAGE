import argparse
import csv
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import load_samples, polygon_mask
from traditional import MetricsCalculator

COLUMNS = ["Structure Distance", "PSNR", "LPIPS", "MSE", "SSIM", "CLIP Whole", "CLIP Edited"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structure distance, background preservation (outside the union of the target masks) "
        "and CLIP similarity. Appends one row per method to --result-csv."
    )
    parser.add_argument("--annotations-jsonl", required=True)
    parser.add_argument("--crop-instruction-jsonl", required=True)
    parser.add_argument("--input-image-root", required=True)
    parser.add_argument("--edited-image-root", required=True)
    parser.add_argument("--result-csv", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    calculator = MetricsCalculator(args.device)
    scores = {column: [] for column in COLUMNS}

    for name, instruction, masks, edits in tqdm(load_samples(args.annotations_jsonl, args.crop_instruction_jsonl)):
        source = Image.open(os.path.join(args.input_image_root, name)).convert("RGB")
        edited = Image.open(os.path.join(args.edited_image_root, name)).convert("RGB")

        edit_masks = []
        for i in edits:
            mask = Image.fromarray(polygon_mask(masks[i], source.size))
            if mask.size != edited.size:
                mask = mask.resize(edited.size, Image.Resampling.NEAREST)
            edit_masks.append(np.array(mask)[:, :, None].repeat(3, axis=2))
        background = 1 - np.maximum.reduce(np.stack(edit_masks, axis=0))
        source = source.resize(edited.size, resample=Image.Resampling.BILINEAR)

        scores["Structure Distance"].append(calculator.structure_distance(source, edited))
        scores["PSNR"].append(calculator.psnr_score(source, edited, background))
        scores["LPIPS"].append(calculator.lpips_score(source, edited, background))
        scores["MSE"].append(calculator.mse_score(source, edited, background))
        scores["SSIM"].append(calculator.ssim_score(source, edited, background))
        scores["CLIP Whole"].append(calculator.clip_similarity(edited, instruction))
        clip_edited = [calculator.clip_similarity(edited, text, mask) for text, mask in zip(edits.values(), edit_masks)]
        scores["CLIP Edited"].append(sum(clip_edited) / len(clip_edited))

    os.makedirs(os.path.dirname(args.result_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.result_csv) or os.path.getsize(args.result_csv) == 0
    method = os.path.basename(os.path.normpath(args.edited_image_root))
    with open(args.result_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["split", "model", *COLUMNS])
        writer.writerow(["all", method, *(f"{sum(v) / len(v):.4f}" for v in scores.values())])
    print(f"Saved to {args.result_csv}")


if __name__ == "__main__":
    main()
