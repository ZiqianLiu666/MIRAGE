import argparse
import json
import os

import torch
from diffusers import AutoModel, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="FLUX.2-dev text-to-image from JSONL source_prompt."
    )
    p.add_argument(
        "--jsonl",
        required=True,
        help="Input JSONL. Each line has {image, source_prompt, ...}.",
    )
    p.add_argument(
        "--results-dir", required=True, help="Directory to save generated images."
    )

    p.add_argument(
        "--repo-id",
        default="black-forest-labs/FLUX.2-dev",
        help="Model repo id.",
    )
    p.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use enable_model_cpu_offload() instead of keeping the full pipeline on --device.",
    )
    p.add_argument("--device", default="cuda:0", help="Device string.")
    p.add_argument(
        "--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Torch dtype."
    )

    p.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Inference steps. FLUX.2-dev card uses 50 (and notes ~28 as speed/quality trade-off).",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="CFG guidance scale. FLUX.2-dev card uses 4.0.",
    )
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument("--height", type=int, default=1024, help="Output height.")
    p.add_argument("--width", type=int, default=1024, help="Output width.")

    p.add_argument("--num-images", type=int, default=1, help="Images per prompt.")
    p.add_argument(
        "--batch-size", type=int, default=1, help="Number of prompts per batch."
    )
    p.add_argument(
        "--skip-existing", action="store_true", help="Skip if output exists."
    )

    return p.parse_args()


def dtype_from_str(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def load_pipeline(repo_id: str, device: str, torch_dtype, cpu_offload: bool = False):
    print("Loading text encoder...")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if not cpu_offload:
        text_encoder = text_encoder.to(device)

    print("Loading DiT transformer...")
    dit = AutoModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if not cpu_offload:
        dit = dit.to(device)

    print("Loading Flux2Pipeline...")
    pipe = Flux2Pipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=dit,
        torch_dtype=torch_dtype,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    torch_dtype = dtype_from_str(args.dtype)
    pipe = load_pipeline(
        args.repo_id,
        args.device,
        torch_dtype,
        cpu_offload=args.cpu_offload,
    )
    with open(args.jsonl, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    records = []
    for idx, line in enumerate(lines):
        rec = json.loads(line)

        prompt = rec.get("source_prompt", "")
        if not prompt:
            print(f"[WARN] line {idx}: missing source_prompt, skip.")
            continue

        stem, ext = os.path.splitext(rec["image"].strip())

        first_output_name = f"{stem}{ext}" if args.num_images == 1 else f"{stem}_00{ext}"
        if args.skip_existing and os.path.exists(
            os.path.join(args.results_dir, first_output_name)
        ):
            continue

        records.append(
            {
                "prompt": prompt,
                "stem": stem,
                "ext": ext,
                "seed_id": int(stem),
            }
        )

    total_batches = (len(records) + args.batch_size - 1) // args.batch_size
    for start in tqdm(range(0, len(records), args.batch_size), total=total_batches, desc="t2i"):
        batch = records[start : start + args.batch_size]
        prompts = [r["prompt"] for r in batch]
        generators = [
            torch.Generator(device=args.device).manual_seed(args.seed + r["seed_id"])
            for r in batch
        ]

        with torch.inference_mode():
            images = pipe(
                prompt=prompts,
                generator=generators,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                num_images_per_prompt=args.num_images,
            ).images

        if args.num_images == 1:
            for r, img in zip(batch, images):
                img.save(os.path.join(args.results_dir, f"{r['stem']}{r['ext']}"))
        else:
            for i, r in enumerate(batch):
                for j in range(args.num_images):
                    img = images[i * args.num_images + j]
                    img.save(
                        os.path.join(args.results_dir, f"{r['stem']}_{j:02d}{r['ext']}")
                    )

    print("Done. Saved to:", args.results_dir)


if __name__ == "__main__":
    main()
