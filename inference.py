import argparse
import json
import os
import time
from collections import defaultdict

import torch
from diffusers import Flux2KleinPipeline, Flux2Pipeline, QwenImageEditPlusPipeline
from PIL import Image
from tqdm import tqdm

from mirage.flux2 import run_flux2
from mirage.qwen_image_edit import run_qwen_image_edit

# pipeline class, checkpoint, default number of steps
MODELS = {
    "flux2_dev": (Flux2Pipeline, "black-forest-labs/FLUX.2-dev", 50),
    "flux2_klein9b": (Flux2KleinPipeline, "black-forest-labs/FLUX.2-klein-base-9B", 50),
    "qwen2511": (QwenImageEditPlusPipeline, "Qwen/Qwen-Image-Edit-2511", 40),
}


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-instance image editing with MIRAGE.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--instruction-jsonl", required=True)
    parser.add_argument("--crop-instruction-jsonl", required=True, help="output of crop_image.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--patch-ratio",
        type=float,
        default=0.4,
        help="fraction of the denoising steps run by the region branches, i.e. 1 - rho",
    )
    parser.add_argument("--num-steps", type=int, help="defaults to 50 for FLUX.2 and 40 for Qwen")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="true CFG scale for Qwen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cpu-offload", default="none", choices=["none", "model", "sequential"])
    return parser


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pipeline(model, device, dtype, cpu_offload):
    pipeline_cls, repo_id, _ = MODELS[model]
    pipe = pipeline_cls.from_pretrained(repo_id, torch_dtype=dtype)
    if cpu_offload == "model":
        pipe.enable_model_cpu_offload()
    elif cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def qwen_prompt(record):
    instruction = record["new_instruction"].strip()
    if instruction[-1] not in ".!?":
        instruction += "."
    return f"Target: {record['refer_object'].strip().rstrip('.')}. Instruction: {instruction}"


def main(args):
    instructions = {r["image"]: r["editing_instruction"] for r in load_jsonl(args.instruction_jsonl)}
    crops = defaultdict(list)
    for record in load_jsonl(args.crop_instruction_jsonl):
        crops[record["original_image"]].append(record)

    pipe = load_pipeline(args.model, args.device, getattr(torch, args.dtype), args.cpu_offload)
    num_steps = args.num_steps or MODELS[args.model][2]
    os.makedirs(args.output_dir, exist_ok=True)

    for name in tqdm(sorted(crops)):
        records = sorted(crops[name], key=lambda r: r["image"])
        image = Image.open(os.path.join(args.image_root, name)).convert("RGB")
        bboxes = [r["bbox"] for r in records]
        generator = torch.Generator(args.device).manual_seed(args.seed)
        options = dict(num_inference_steps=num_steps, patch_ratio=args.patch_ratio, generator=generator)

        start = time.perf_counter()
        if args.model == "qwen2511":
            sub_prompts = [qwen_prompt(r) for r in records]
            edited = run_qwen_image_edit(
                pipe, image, instructions[name], sub_prompts, bboxes, true_cfg_scale=args.guidance_scale, **options
            )
        else:
            sub_prompts = [r["new_instruction"] for r in records]
            edited = run_flux2(
                pipe, image, instructions[name], sub_prompts, bboxes, guidance_scale=args.guidance_scale, **options
            )
        tqdm.write(f"{name}: {time.perf_counter() - start:.1f}s")
        edited.save(os.path.join(args.output_dir, name))


if __name__ == "__main__":
    main(build_parser().parse_args())
