# RefEdit: A Benchmark and Method for Improving Instruction-based Image Editing Model for Referring Expression
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxxx)

## Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Benchmark Access
We release **MIRAGE-Bench**, which can be downloaded [here](https://drive.google.com/file/d/1nOACtv7H3QXxE78ba9ok-5zZGUWB1vMW/view?usp=sharing). The benchmark contains 100 samples, each consisting of an image, a composite editing instruction formed by combining five sub-instructions, and the corresponding ground-truth mask. This benchmark is designed to evaluate image editing models in more complex referring-expression scenarios. 

Notably, the entire **MIRAGE-Bench** is constructed based on our proposed **Automatic Image Synthesis Pipeline**.

## Automatic Image Synthesis Pipeline
We provide a fully automated pipeline for generating image with multiple similar instances and composite editing instructions. 
Please run the following commands in sequence to obtain a complete synthesized dataset.

### Image Description Generation
Generate diverse image descriptions:
```
python synthesis_pipeline/generate_source_prompts_batch_pairs.py \
  --pair-template synthesis_pipeline/prompt_template/image_description/prompt_pair_batch.txt \
  --generator-template synthesis_pipeline/prompt_template/image_description/prompt_draft_generator.txt \
  --judge-template synthesis_pipeline/prompt_template/image_description/prompt_judge.txt \
  --out synthesis_pipeline/source_prompts.jsonl \
  --num-samples 200
```

### Image Generation
Synthesize images from the generated image descriptions:
```
python synthesis_pipeline/flux_t2i_generate.py \
  --jsonl synthesis_pipeline/source_prompts.jsonl \
  --results-dir synthesis_pipeline/benchmark
```

### Editing Instruction Generation
Generate composite editing instructions based on synthetic images and image descriptions:
```
python synthesis_pipeline/generate_instruction_refer.py \
  --image-dir synthesis_pipeline/benchmark \
  --jsonl synthesis_pipeline/source_prompts.jsonl \
  --out-jsonl synthesis_pipeline/instruction.jsonl \
  --slot-template synthesis_pipeline/prompt_template/instruction/repeated_slot_plan.txt \
  --generator-template synthesis_pipeline/prompt_template/instruction/instruction_generate.txt \
  --extractor-template synthesis_pipeline/prompt_template/instruction/refer_extract.txt
```

### Mask Generation
Generate target masks:
```
python synthesis_pipeline/generate_bbox_mask.py \
  --image-dir synthesis_pipeline/benchmark \
  --jsonl synthesis_pipeline/instruction.jsonl \
  --vis-dir synthesis_pipeline/bbox_mask_vis
```

## 推理
在这里，我们分别提供了在各个基础模型上集成MIRAGE的方法。

### 目标框定位
请运行以下命令来获取图像中目标区域的crop图:
```

```

### FLUX.2[klein]-9B + MIRAGE
要获得FLUX.2[klein]-9B集成的MIRAGE的结果，请使用以下命令：
```

```

### Flux.2[Dev] + MIRAGE
要获得Flux.2[Dev]集成的MIRAGE的结果，请使用以下命令：
```

```

### Qwen-Image-Edit-2511 + MIRAGE
要获得Qwen-Image-Edit-2511集成的MIRAGE的结果，请使用以下命令：
```

```

























