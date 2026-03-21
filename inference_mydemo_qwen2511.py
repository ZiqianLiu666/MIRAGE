import argparse
import json
import os
import time
from collections import defaultdict

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from utils.runner_qwen2511 import run_qwen_multi_branch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-branch Qwen-Image-Edit inference (single image or batch)."
    )

    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to a single image. If set to a directory, batch mode is used.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction string for single image mode.",
    )

    parser.add_argument(
        "--image-root",
        default=None,
        help="Folder containing original images (batch mode).",
    )
    parser.add_argument(
        "--instruction-jsonl",
        default=None,
        help="JSONL file with image->instruction mapping (batch mode only).",
    )
    parser.add_argument(
        "--crop-dir",
        default=None,
        help="Folder containing crop images (must include crop_instruction.jsonl).",
    )

    parser.add_argument(
        "--results-full-dir",
        default="/home/infres/ziliu-24/instruct-pix2pix/results_ours_qwen2511",
        help="Output folder for full images.",
    )

    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen-Image-Edit-2511",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Pipeline device.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Torch dtype.",
    )
    parser.add_argument(
        "--cpu-offload",
        default="model",
        choices=["none", "model", "sequential"],
        help=(
            "CPU offload mode. "
            "'model' uses enable_model_cpu_offload (recommended), "
            "'sequential' uses enable_sequential_cpu_offload (slower, lower memory), "
            "'none' keeps full model on --device."
        ),
    )

    parser.add_argument(
        "--patch-ratio",
        type=float,
        default=0.2,
        help="Early-stage patch ratio.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=40,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="True CFG scale.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=" ",
        help="Negative prompt. Keep a blank string by default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def dtype_from_str(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def load_instruction_map(jsonl_path: str):
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mapping[rec["image"]] = rec["editing_instruction"]
    return mapping


def load_crop_records(jsonl_path: str):
    mapping = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mapping[rec["original_image"]].append(rec)
    return mapping


def load_qwen_pipeline(
    model_id: str,
    device: str,
    torch_dtype,
    cpu_offload: str,
):
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype
    )

    if cpu_offload != "none":
        if cpu_offload == "sequential":
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)

    return pipeline


def collect_crop_inputs(crop_dir: str, image_name: str, crop_records):
    crop_images = []
    crop_prompts = []
    bboxes = []
    for rec in crop_records:
        crop_filename = rec["image"]
        crop_prompt = rec["new_instruction"]
        bbox = rec["bbox"]

        crop_image_path = os.path.join(crop_dir, image_name, crop_filename)
        crop_images.append(Image.open(crop_image_path))
        crop_prompts.append(crop_prompt)
        bboxes.append(bbox)

    return crop_images, crop_prompts, bboxes


def save_outputs(image_name: str, full_out, results_full_dir: str):
    os.makedirs(results_full_dir, exist_ok=True)
    full_save_path = os.path.join(results_full_dir, image_name)
    full_out.save(full_save_path)



def run_inference_loop(
    pipe,
    image_names: list,
    image_root: str,
    inst_map: dict,
    crop_map: dict,
    crop_dir: str,
    results_full_dir: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    negative_prompt: str,
    generator_device: str,
    seed: int,
    patch_ratio: float,
):
    for idx, img_name in enumerate(image_names):
        if len(image_names) > 1:
            print(f"\n=== [{idx + 1}/{len(image_names)}] Processing {img_name} ===")
        else:
            print(f"\n=== Processing {img_name} ===")

        full_image_path = os.path.join(image_root, img_name)
        full_prompt = inst_map[img_name]
        crop_records = sorted(crop_map[img_name], key=lambda r: r.get("image", ""))

        full_image = Image.open(full_image_path)
        crop_images, crop_prompts, bboxes = collect_crop_inputs(
            crop_dir, img_name, crop_records
        )

        generator = torch.Generator(device=generator_device).manual_seed(seed)
        infer_start = time.perf_counter()
        full_out = run_qwen_multi_branch(
            pipe=pipe,
            full_image=full_image,
            crop_images=crop_images,
            full_prompt=full_prompt,
            crop_prompts=crop_prompts,
            bboxes=bboxes,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            patch_ratio=patch_ratio,
        )
        infer_elapsed = time.perf_counter() - infer_start

        print(f"[Runtime] {img_name} inference: {infer_elapsed:.3f}s")
        save_outputs(img_name, full_out, results_full_dir)


def main():
    args = parse_args()

    if args.image_path:
        inst_map = {os.path.basename(args.image_path): args.instruction}
    else:
        inst_map = load_instruction_map(args.instruction_jsonl)
    crop_jsonl_path = os.path.join(args.crop_dir, "crop_instruction.jsonl")
    crop_map = load_crop_records(crop_jsonl_path)

    torch_dtype = dtype_from_str(args.dtype)
    pipeline = load_qwen_pipeline(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=torch_dtype,
        cpu_offload=args.cpu_offload,
    )

    generator_device = "cuda" if args.device == "cuda" else "cpu"

    if args.image_path:
        image_root = os.path.dirname(args.image_path)
        image_names = [os.path.basename(args.image_path)]
    else:
        image_root = args.image_root
        image_names = sorted(crop_map.keys())
        print(f"Found {len(image_names)} images with crop records.")
        if image_names:
            print(
                f"Forward order: start from {image_names[0]}, end at {image_names[-1]}."
            )

    run_inference_loop(
        pipe=pipeline,
        image_names=image_names,
        image_root=image_root,
        inst_map=inst_map,
        crop_map=crop_map,
        crop_dir=args.crop_dir,
        results_full_dir=args.results_full_dir,
        num_inference_steps=args.num_steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        generator_device=generator_device,
        seed=args.seed,
        patch_ratio=args.patch_ratio,
    )


if __name__ == "__main__":
    main()
