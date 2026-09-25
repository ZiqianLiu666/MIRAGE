import argparse
import json
import os

import torch
from diffusers import Flux2Pipeline
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Render the image descriptions with FLUX.2 [dev] (Sec. A.2).")
    parser.add_argument("--jsonl", required=True, help="output of generate_source_prompts.py")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--repo-id", default="black-forest-labs/FLUX.2-dev")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42, help="image i is generated with seed + i")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cpu-offload", default="none", choices=["none", "model", "sequential"])
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = Flux2Pipeline.from_pretrained(args.repo_id, torch_dtype=getattr(torch, args.dtype))
    if args.cpu_offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(args.device)

    with open(args.jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    os.makedirs(args.results_dir, exist_ok=True)
    for start in tqdm(range(0, len(records), args.batch_size)):
        batch = records[start : start + args.batch_size]
        generators = [
            torch.Generator(args.device).manual_seed(args.seed + int(os.path.splitext(r["image"])[0])) for r in batch
        ]
        images = pipe(
            prompt=[r["source_prompt"] for r in batch],
            generator=generators,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
        ).images
        for record, image in zip(batch, images):
            image.save(os.path.join(args.results_dir, record["image"]))


if __name__ == "__main__":
    main()
