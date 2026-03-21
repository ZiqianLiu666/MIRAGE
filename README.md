# MIRAGE: Benchmarking and Aligning Multi-Instance Image Editing
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxxx)

> **Abstract:** *Instruction-guided image editing has seen remarkable progress with models like FLUX.2 and Qwen-Image-Edit, yet they still struggle with complex scenarios involving multiple similar instances, each requiring individual edits. We observe that state-of-the-art models suffer from severe over-editing and spatial misalignment when faced with multiple identical instances and composite instructions. To address this, we introduce a comprehensive benchmark specifically designed to evaluate fine-grained consistency in multi-instance and multi-instruction settings. We further propose Multi-Instance Regional Alignment via Guided Editing (MIRAGE), a training-free framework for precise, localized editing. By leveraging a vision-language model to decompose complex instructions into region-specific subsets, MIRAGE employs a multi-branch parallel denoising strategy that injects target-region latents into the global representation while preserving background integrity through a reference trajectory. Extensive evaluations on MIRAGE-Bench and RefEdit-Bench demonstrate that our framework significantly outperforms existing methods in achieving precise instance-level modifications while maintaining strong background consistency.*

![overview](jpg/mybench_qualitative.jpg)
**Fig. 1: Example images and instructions involving multiple similar instances and compositional edits.** Such scenarios are challenging for state-of-the-art models, which often introduce unintended modifications. In contrast, MIRAGE achieves precise instance-level editing while preserving background consistency.

# 1. Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

# 2. Benchmark Access
We release **MIRAGE-Bench**, which can be downloaded [here](https://drive.google.com/file/d/1nOACtv7H3QXxE78ba9ok-5zZGUWB1vMW/view?usp=sharing). The benchmark contains 100 samples, each consisting of an image, a composite editing instruction formed by combining five sub-instructions, and the corresponding ground-truth mask. This benchmark is designed to evaluate image editing models in more complex referring-expression scenarios. 

Notably, the entire **MIRAGE-Bench** is constructed based on our proposed **Automatic Image Synthesis Pipeline**.

![benchmark](jpg/benchmark_example.jpg)
**Fig. 2: MIRAGE-bench sample examples.** The first row shows the synthesized original images, the second row presents the corresponding ground-truth (GT) masks of the target regions, and the third row displays the editing instructions constructed based on the generated image semantics and the source prompts.

# 3. Automatic Image Synthesis Pipeline
We provide a fully automated pipeline for generating image with multiple similar instances and composite editing instructions. 

Please run the following commands in sequence to obtain a complete synthesized dataset.

```
## 3.1 Image Description Generation
python synthesis_pipeline/generate_source_prompts_batch_pairs.py \
  --pair-template synthesis_pipeline/prompt_template/image_description/prompt_pair_batch.txt \
  --generator-template synthesis_pipeline/prompt_template/image_description/prompt_draft_generator.txt \
  --judge-template synthesis_pipeline/prompt_template/image_description/prompt_judge.txt \
  --out synthesis_pipeline/source_prompts.jsonl \
  --num-samples 200

## 3.2 Image Generation
python synthesis_pipeline/flux_t2i_generate.py \
  --jsonl synthesis_pipeline/source_prompts.jsonl \
  --results-dir synthesis_pipeline/benchmark

## 3.3 Editing Instruction Generation
python synthesis_pipeline/generate_instruction_refer.py \
  --image-dir synthesis_pipeline/benchmark \
  --jsonl synthesis_pipeline/source_prompts.jsonl \
  --out-jsonl synthesis_pipeline/benchmark/annotations.jsonl \
  --slot-template synthesis_pipeline/prompt_template/instruction/repeated_slot_plan.txt \
  --generator-template synthesis_pipeline/prompt_template/instruction/instruction_generate.txt \
  --extractor-template synthesis_pipeline/prompt_template/instruction/refer_extract.txt

## 3.4 Mask Generation
python synthesis_pipeline/generate_bbox_mask.py \
  --image-dir synthesis_pipeline/benchmark \
  --jsonl synthesis_pipeline/benchmark/annotations.jsonl \
  --vis-dir synthesis_pipeline/benchmark/bbox_mask_vis
```

# 4. 推理
在这里，我们分别提供了在各个基础模型上集成MIRAGE的方法。

## 4.1 目标框定位
在推理之前，请你运行以下命令先获取图像中目标区域的crop图:
```
python crop_image.py \
  --image-root synthesis_pipeline/benchmark \
  --instruction-jsonl synthesis_pipeline/benchmark/annotations.jsonl \
  --output-dir synthesis_pipeline/benchmark/crops \
  --out-jsonl synthesis_pipeline/benchmark/crops/crop_instruction.jsonl \
  --batch-size 2 \
  --padding 10
```

## 4.2 Base model + MIRAGE
想要获得以下基础模型集成的MIRAGE的结果，请使用以下命令：
```
# FLUX.2[klein]-9B
python3 inference_mydemo_flux2_klein9B.py \
  --image-root synthesis_pipeline/benchmark \
  --instruction-jsonl synthesis_pipeline/benchmark/annotations.jsonl \
  --crop-dir synthesis_pipeline/benchmark/crops \
  --results-full-dir synthesis_pipeline/results_flux2_klein9B \
  --patch-ratio 0.2
```

```
# Flux.2[Dev] + MIRAGE
python inference_mydemo_flux2_dev.py \
  --image-root synthesis_pipeline/benchmark \
  --instruction-jsonl synthesis_pipeline/benchmark/annotations.jsonl \
  --crop-dir synthesis_pipeline/benchmark/crops \
  --results-full-dir synthesis_pipeline/results_flux2_dev \
  --patch-ratio 0.2
```

```
# Qwen-Image-Edit-2511 + MIRAGE
python inference_mydemo_qwen2511.py \
  --image-root synthesis_pipeline/benchmark \
  --instruction-jsonl synthesis_pipeline/benchmark/annotations.jsonl \
  --crop-dir synthesis_pipeline/benchmark/crops \
  --results-full-dir synthesis_pipeline/results_qwen2511 \
  --patch-ratio 0.2
```

# 5. 指标测评
## LLM打分
注意这里的PF和Cons是用本地开源的 Qwen 运行得到，而PQ调用的是GPT的API
```
# PF, Cons, PQ
python metrics/EditScore/evaluation.py \
  --annotations-jsonl synthesis_pipeline/benchmark/annotations.jsonl \
  --crop-instruction-jsonl synthesis_pipeline/benchmark/crops/crop_instruction.jsonl \
  --input-image-root synthesis_pipeline/benchmark \
  --edited-image-root your_results \
  --result-dir metrics/EditScore/runs/your_results \
  --sc-model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --pq-model-name-or-path gpt-4.1 \
  --pq-key YOUR_OPENAI_API_KEY \
  --num-pass 3 \
```

## 传统指标
```
# MSE, LPIPS, PSNR...
python metrics/traditional/evalaute_traditional.py \
  --annotation_mapping_file generate_benchmark/instruction.fixed.jsonl \
  --src_image_folder generate_benchmark/filtered_benchmark \
  --crop-instruction-jsonl generate_benchmark/crops/crop_instruction.jsonl \
  --tgt_method results_ours_flux2_dev \
  --result_path metrics/traditional/table5_summary.csv \
  --per_image_result_path metrics/traditional/table5_per_image.csv \
  --device cuda:0
```

# Citation
If you use this code or the dataset in your research, please cite our paper:
```
xxx
```





















