import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MIRAGE inference directly with the MIRAGE benchmark on Hugging Face."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["flux2_dev", "flux2_klein9b", "qwen2511"],
        help="Base model used by MIRAGE.",
    )
    parser.add_argument(
        "--hf-dataset-id",
        default="ziqiangoodgood/MIRAGE",
        help="Hugging Face dataset id for MIRAGE-Bench.",
    )
    parser.add_argument(
        "--results-full-dir",
        default=None,
        help="Output directory for edited images.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string.",
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
        help="CPU offload mode for supported models.",
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
        default=None,
        help="Inference steps. Uses each model's default when omitted.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Guidance scale. Uses each model's default when omitted.",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=None,
        help="True CFG scale for Qwen. Uses the model default when omitted.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for Qwen. Uses the model default when omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def download_and_resolve_mirage_benchmark(dataset_id: str):
    from huggingface_hub import snapshot_download

    snapshot_root = Path(
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
        )
    ).resolve()

    benchmark_root = snapshot_root / "benchmark"
    instruction_jsonl = benchmark_root / "annotations.jsonl"
    crop_dir = benchmark_root / "crops"
    crop_instruction_jsonl = crop_dir / "crop_instruction.jsonl"

    required_paths = (
        benchmark_root,
        instruction_jsonl,
        crop_dir,
        crop_instruction_jsonl,
    )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "The Hugging Face dataset does not match the expected MIRAGE benchmark layout: "
            + ", ".join(missing_paths)
        )

    return {
        "snapshot_root": snapshot_root,
        "image_root": benchmark_root,
        "instruction_jsonl": instruction_jsonl,
        "crop_dir": crop_dir,
        "crop_instruction_jsonl": crop_instruction_jsonl,
    }


def resolve_results_dir(args):
    if args.results_full_dir is not None:
        return args.results_full_dir
    return f"results/{args.model}"


def run_flux2_dev(args, benchmark):
    import inference_mydemo_flux2_dev as flux2_dev

    num_steps = 50 if args.num_steps is None else args.num_steps
    guidance_scale = 4.0 if args.guidance_scale is None else args.guidance_scale
    inst_map = flux2_dev.load_instruction_map(str(benchmark["instruction_jsonl"]))
    crop_map = flux2_dev.load_crop_records(str(benchmark["crop_instruction_jsonl"]))
    pipe = flux2_dev.load_flux2_pipeline(
        repo_id="black-forest-labs/FLUX.2-dev",
        device=args.device,
        torch_dtype=flux2_dev.dtype_from_str(args.dtype),
        cpu_offload=args.cpu_offload,
    )
    flux2_dev.run_inference_loop(
        pipe=pipe,
        image_names=sorted(crop_map.keys()),
        image_root=str(benchmark["image_root"]),
        inst_map=inst_map,
        crop_map=crop_map,
        crop_dir=str(benchmark["crop_dir"]),
        results_full_dir=resolve_results_dir(args),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator_device=args.device if str(args.device).startswith("cuda") else "cpu",
        seed=args.seed,
        patch_ratio=args.patch_ratio,
    )


def run_flux2_klein9b(args, benchmark):
    import inference_mydemo_flux2_klein9B as flux2_klein9b

    num_steps = 50 if args.num_steps is None else args.num_steps
    guidance_scale = 4.0 if args.guidance_scale is None else args.guidance_scale
    inst_map = flux2_klein9b.load_instruction_map(str(benchmark["instruction_jsonl"]))
    crop_map = flux2_klein9b.load_crop_records(str(benchmark["crop_instruction_jsonl"]))
    pipe = flux2_klein9b.load_flux2_klein_pipeline(
        repo_id="black-forest-labs/FLUX.2-klein-base-9B",
        device=args.device,
        torch_dtype=flux2_klein9b.dtype_from_str(args.dtype),
    )
    flux2_klein9b.run_inference_loop(
        pipe=pipe,
        image_names=sorted(crop_map.keys()),
        image_root=str(benchmark["image_root"]),
        inst_map=inst_map,
        crop_map=crop_map,
        crop_dir=str(benchmark["crop_dir"]),
        results_full_dir=resolve_results_dir(args),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator_device=args.device if str(args.device).startswith("cuda") else "cpu",
        seed=args.seed,
        patch_ratio=args.patch_ratio,
    )


def run_qwen2511(args, benchmark):
    import inference_mydemo_qwen2511 as qwen2511

    num_steps = 40 if args.num_steps is None else args.num_steps
    guidance_scale = 1.0 if args.guidance_scale is None else args.guidance_scale
    true_cfg_scale = 4.0 if args.true_cfg_scale is None else args.true_cfg_scale
    negative_prompt = " " if args.negative_prompt is None else args.negative_prompt
    inst_map = qwen2511.load_instruction_map(str(benchmark["instruction_jsonl"]))
    crop_map = qwen2511.load_crop_records(str(benchmark["crop_instruction_jsonl"]))
    pipe = qwen2511.load_qwen_pipeline(
        model_id="Qwen/Qwen-Image-Edit-2511",
        device=args.device,
        torch_dtype=qwen2511.dtype_from_str(args.dtype),
        cpu_offload=args.cpu_offload,
    )
    qwen2511.run_inference_loop(
        pipe=pipe,
        image_names=sorted(crop_map.keys()),
        image_root=str(benchmark["image_root"]),
        inst_map=inst_map,
        crop_map=crop_map,
        crop_dir=str(benchmark["crop_dir"]),
        results_full_dir=resolve_results_dir(args),
        num_inference_steps=num_steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        generator_device=args.device if str(args.device).startswith("cuda") else "cpu",
        seed=args.seed,
        patch_ratio=args.patch_ratio,
    )


def main():
    args = parse_args()
    benchmark = download_and_resolve_mirage_benchmark(args.hf_dataset_id)

    if args.model == "flux2_dev":
        run_flux2_dev(args, benchmark)
        return
    if args.model == "flux2_klein9b":
        run_flux2_klein9b(args, benchmark)
        return
    if args.model == "qwen2511":
        run_qwen2511(args, benchmark)
        return

    raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
