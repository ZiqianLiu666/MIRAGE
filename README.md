# MIRAGE: Benchmarking and Aligning Multi-Instance Image Editing
[Ziqian Liu](https://scholar.google.com/citations?view_op=list_works&hl=en&user=_1FGL3UAAAAJ) and [Stephan Alaniz*](https://scholar.google.com/citations?user=mzZa_yQAAAAJ&hl=en&oi=ao)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxxx)
[![dataset](https://img.shields.io/badge/🤗%20Dataset-MIRAGE-yellow)](https://huggingface.co/datasets/ziqiangoodgood/MIRAGE)

> **Abstract:** *Instruction-guided image editing has seen remarkable progress with models like FLUX.2 and Qwen-Image-Edit, yet they still struggle with complex scenarios involving multiple similar instances, each requiring individual edits. We observe that state-of-the-art models suffer from severe over-editing and spatial misalignment when faced with multiple identical instances and composite instructions. To address this, we introduce a comprehensive benchmark specifically designed to evaluate fine-grained consistency in multi-instance and multi-instruction settings. We further propose Multi-Instance Regional Alignment via Guided Editing (MIRAGE), a training-free framework for precise, localized editing. By leveraging a vision-language model to decompose complex instructions into region-specific subsets, MIRAGE employs a multi-branch parallel denoising strategy that injects target-region latents into the global representation while preserving background integrity through a reference trajectory. Extensive evaluations on MIRAGE-Bench and RefEdit-Bench demonstrate that our framework significantly outperforms existing methods in achieving precise instance-level modifications while maintaining strong background consistency.*

![overview](jpg/mybench_qualitative.jpg)
**Fig. 1: Example images and instructions involving multiple similar instances and compositional edits.** Such scenarios are challenging for state-of-the-art models, which often introduce unintended modifications. In contrast, MIRAGE achieves precise instance-level editing while preserving background consistency.

# Benchmark Access
We release **MIRAGE-Bench**, which can be downloaded on [Huggingface](https://huggingface.co/datasets/ziqiangoodgood/MIRAGE) or [Google Drive](https://drive.google.com/file/d/1VK8Vu7Vdw35GWb7IapZLFSugoJTblTDx/view?usp=sharing) directly. The benchmark contains 100 samples, each consisting of an image, a composite editing instruction formed by combining five sub-instructions, and the corresponding ground-truth mask. This benchmark is designed to evaluate image editing models in more complex referring-expression scenarios. 

Notably, the entire **MIRAGE-Bench** is constructed based on our proposed [2. Automatic Image Synthesis Pipeline](#2-Automatic-Image-Synthesis-Pipeline).

![benchmark](jpg/benchmark_example.jpg)
**Fig. 2: MIRAGE-bench sample examples.** The first row shows the synthesized original images, the second row presents the corresponding ground-truth (GT) masks of the target regions, and the third row displays the editing instructions constructed based on the generated image semantics and the source prompts.

# 1. Requirements
Install the required dependencies:
```bash
conda create -n mirage python=3.12 -y
conda activate mirage
pip install -r requirements.txt
```

Don't forget to log in to your Hugging Face account to get model access:
```
echo 'export HF_TOKEN=xxx' >> ~/.bashrc
source ~/.bashrc
```

# Quick Start
To quickly try MIRAGE, you can run the following commands directly. The benchmark will be automatically downloaded from Hugging Face, so no manual setup is required.

You can switch between different base models by modifying the `--model` argument.
```bash
# FLUX.2 [klein]-9B + MIRAGE
python quick_start.py \
  --model flux2_klein9b \
  --results-full-dir results/flux2_klein9B \
  --patch-ratio 0.2

# FLUX.2 [Dev] + MIRAGE (If GPU memory is insufficient, you can enable CPU offloading by adding `--cpu-offload model` or even `--cpu-offload sequential`)
python quick_start.py \
  --model flux2_dev \
  --results-full-dir results/flux2_dev \
  --patch-ratio 0.2

# Qwen-Image-Edit-2511 + MIRAGE (If GPU memory is insufficient, you can enable CPU offloading by adding `--cpu-offload model` or even `--cpu-offload sequential`)
python quick_start.py \
  --model qwen2511 \
  --results-full-dir results/qwen2511 \
  --patch-ratio 0.2
```

# 2. Automatic Image Synthesis Pipeline
We provide a fully automated pipeline for generating image with multiple similar instances and composite editing instructions. 

If you need, please run the following commands in sequence to obtain a complete synthesized dataset. Alternatively, you can directly download the benchmark from the link above and proceed to [3.2 Base model + MIRAGE](#32-base-model--mirage) for inference, skipping **Automatic Image Synthesis Pipeline**. 

```
## 2.1 Image Description Generation
python synthesis_pipeline/generate_source_prompts_batch_pairs.py \
  --pair-template synthesis_pipeline/prompt_template/image_description/prompt_pair_batch.txt \
  --generator-template synthesis_pipeline/prompt_template/image_description/prompt_draft_generator.txt \
  --judge-template synthesis_pipeline/prompt_template/image_description/prompt_judge.txt \
  --out synthesis_pipeline/source_prompts.jsonl \
  --num-samples 200

## 2.2 Image Generation (If GPU memory is insufficient, you can enable CPU offloading by adding `--cpu-offload model` or even `--cpu-offload sequential`)
python synthesis_pipeline/flux_t2i_generate.py \
  --jsonl synthesis_pipeline/source_prompts.jsonl \
  --results-dir benchmark \
  --batch-size 2

## 2.3 Editing Instruction Generation
python synthesis_pipeline/generate_instruction_refer.py \
  --image-dir benchmark \
  --jsonl synthesis_pipeline/source_prompts.jsonl \
  --out-jsonl benchmark/annotations.jsonl \
  --slot-template synthesis_pipeline/prompt_template/instruction/repeated_slot_plan.txt \
  --generator-template synthesis_pipeline/prompt_template/instruction/instruction_generate.txt \
  --extractor-template synthesis_pipeline/prompt_template/instruction/refer_extract.txt

## 2.4 Mask Generation
python synthesis_pipeline/generate_bbox_mask.py \
  --image-dir benchmark \
  --jsonl benchmark/annotations.jsonl \
  --vis-dir benchmark/bbox_mask_vis
```

# 3. Inference
We provide MIRAGE integration pipelines for multiple base image editing models.

## 3.1 Target Localization
Before running inference, first obtain cropped regions corresponding to the target objects:

```
python crop_image.py \
  --image-root benchmark \
  --instruction-jsonl benchmark/annotations.jsonl \
  --output-dir benchmark/crops \
  --out-jsonl benchmark/crops/crop_instruction.jsonl \
  --batch-size 2 \
  --padding 10
```

## 3.2 Base model + MIRAGE
Run MIRAGE on different base models:
```
# FLUX.2[klein]-9B + MIRAGE
python inference_mydemo_flux2_klein9B.py \
  --image-root benchmark \
  --instruction-jsonl benchmark/annotations.jsonl \
  --crop-dir benchmark/crops \
  --results-full-dir results/flux2_klein9B \
  --patch-ratio 0.2
```

```
# Flux.2[Dev] + MIRAGE (If GPU memory is insufficient, you can enable CPU offloading by adding `--cpu-offload model` or even `--cpu-offload sequential`)
python inference_mydemo_flux2_dev.py \
  --image-root benchmark \
  --instruction-jsonl benchmark/annotations.jsonl \
  --crop-dir benchmark/crops \
  --results-full-dir results/flux2_dev \
  --patch-ratio 0.2
```

```
# Qwen-Image-Edit-2511 + MIRAGE (If GPU memory is insufficient, you can enable CPU offloading by adding `--cpu-offload model` or even `--cpu-offload sequential`)
python inference_mydemo_qwen2511.py \
  --image-root benchmark \
  --instruction-jsonl benchmark/annotations.jsonl \
  --crop-dir benchmark/crops \
  --results-full-dir results/qwen2511 \
  --patch-ratio 0.2
```

# 4. Evaluation
## LLM-based Metrics
PF and Cons are computed using a local open-source Qwen model, while PQ is evaluated using the GPT API.
```
# PF, Cons, PQ
python metrics/EditScore/evaluation.py \
  --annotations-jsonl benchmark/annotations.jsonl \
  --crop-instruction-jsonl benchmark/crops/crop_instruction.jsonl \
  --input-image-root benchmark \
  --edited-image-root results/your_results \
  --result-dir metrics/results/LLM/your_results \
  --sc-model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --pq-model-name-or-path gpt-5.1 \
  --pq-key YOUR_OPENAI_API_KEY \
  --num-pass 3
```

## Traditional Metrics
Compute pixel-level similarity metrics:

```
# MSE, LPIPS, PSNR...
python metrics/traditional/evalaute_traditional.py \
  --annotation_mapping_file benchmark/annotations.jsonl \
  --src_image_folder benchmark \
  --crop-instruction-jsonl benchmark/crops/crop_instruction.jsonl \
  --tgt_method results/your_results \
  --result_path metrics/results/traditional/your_results/metric_summary.csv
```

# Citation
If you use this code or the benchmark in your research, please cite our paper:
```
xxx
```





















