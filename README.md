# MIRAGE: Benchmarking and Aligning Multi-Instance Image Editing
[Ziqian Liu](https://scholar.google.com/citations?view_op=list_works&hl=en&user=_1FGL3UAAAAJ) and [Stephan Alaniz](https://scholar.google.com/citations?user=mzZa_yQAAAAJ&hl=en&oi=ao)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.05180)
[![dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/ziqiangoodgood/MIRAGE)

> **Abstract:** *Instruction-guided image editing has seen remarkable progress with models like FLUX.2 and Qwen-Image-Edit, yet they still struggle with complex scenarios involving multiple similar instances, each requiring individual edits. We observe that state-of-the-art models suffer from severe over-editing and spatial misalignment when faced with multiple identical instances and composite instructions. To address this, we introduce a comprehensive benchmark specifically designed to evaluate fine-grained consistency in multi-instance and multi-instruction settings. We further propose Multi-Instance Regional Alignment via Guided Editing (MIRAGE), a training-free framework for precise, localized editing. By leveraging a vision-language model to decompose complex instructions into region-specific subsets, MIRAGE employs a multi-branch parallel denoising strategy that injects target-region latents into the global representation while preserving background integrity through a reference trajectory. Extensive evaluations on MIRA-Bench and RefEdit-Bench demonstrate that our framework significantly outperforms existing methods in achieving precise instance-level modifications while maintaining strong background consistency.*

![overview](jpg/mybench_qualitative.pdf)
**Fig. 1: Example images and instructions involving multiple similar instances and compositional edits.** Such scenarios are challenging for state-of-the-art models, which often introduce unintended modifications. In contrast, MIRAGE achieves precise instance-level editing while preserving background consistency.

# Benchmark Access
We release **MIRA-Bench**, which can be downloaded on [Huggingface](https://huggingface.co/datasets/ziqiangoodgood/MIRAGE) or [Google Drive](https://drive.google.com/file/d/1VK8Vu7Vdw35GWb7IapZLFSugoJTblTDx/view?usp=sharing) directly. The benchmark contains 100 samples, each consisting of an image, a composite editing instruction formed by combining five sub-instructions, and the corresponding ground-truth mask. This benchmark is designed to evaluate image editing models in more complex referring-expression scenarios. 

Notably, the entire **MIRA-Bench** is constructed based on our proposed [Automatic Image Synthesis Pipeline](#2-Automatic-Image-Synthesis-Pipeline).

![benchmark](jpg/benchmark_example.pdf)
**Fig. 2: MIRA-Bench sample examples.** The first row shows the synthesized original images, the second row presents the corresponding ground-truth (GT) masks of the target regions, and the third row displays the editing instructions constructed based on the generated image semantics and the source prompts.

# 1. Requirements
Install the required dependencies:
```bash
conda create -n mirage python=3.12 -y
conda activate mirage
pip install -r requirements.txt
```

The mask generation step of the synthesis pipeline additionally needs [SAM 2](https://github.com/facebookresearch/sam2):
```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Don't forget to log in to your Hugging Face account to get model access:
```bash
echo 'export HF_TOKEN=xxx' >> ~/.bashrc
source ~/.bashrc
```

## Quick Start
To quickly try MIRAGE, you can run the following commands directly. The benchmark will be automatically downloaded from Hugging Face, so no manual setup is required.

```bash
# FLUX.2 [klein]-9B + MIRAGE
python quick_start.py --model flux2_klein9b --output-dir results/flux2_klein9b

# FLUX.2 [dev] + MIRAGE
python quick_start.py --model flux2_dev --output-dir results/flux2_dev

# Qwen-Image-Edit-2511 + MIRAGE
python quick_start.py --model qwen2511 --output-dir results/qwen2511
```

If GPU memory is insufficient, you can enable CPU offloading by adding `--cpu-offload model` or even `--cpu-offload sequential`.
`--patch-ratio` is the fraction of the denoising steps handled by the region branches, i.e. 1 - ρ in the paper. The default of 0.4 is the setting we use on MIRA-Bench; for single-instruction benchmarks such as RefEdit-Bench we use 0.2.

# 2. Inference
We provide MIRAGE integration pipelines for multiple base image editing models.

## 3.1 Target Localization
Before running inference, first decompose the instructions and obtain the cropped regions of the target objects with a VLM. `--vlm` supports `qwen35` (Qwen3.5-9B, default) and `gemma4`:

```bash
python crop_image.py \
  --image-root benchmark \
  --instruction-jsonl benchmark/annotations.jsonl \
  --output-dir benchmark/crops \
  --out-jsonl benchmark/crops/crop_instruction.jsonl \
  --batch-size 2
```

By default this command recomputes all images and overwrites the output records and crops.
Add `--resume` to reuse each saved crop with a JSONL record and an existing PNG file.
Resume does not open existing PNGs or compare their dimensions with the recorded bbox.
Complete images are skipped; missing crops are localized again.
For incomplete images, instruction decomposition is rerun and checked against the saved
crop numbers, referring phrases, edit instructions, and expected count before reusing crops.
If the decomposition differs, the command stops rather than mixing results.
Legacy crops without `refer_object` or `new_instruction` are recomputed; their old records
are replaced only after the new crop is successfully saved.
When annotations contain a nonempty list of per-edit masks, that list determines the expected
edit count for completion checks. A shorter or longer decomposition raises an error before
localization; a model's saved `num_crops` alone cannot mark such an image complete.
The optional `--overwrite` flag explicitly selects the default and cannot be combined with
`--resume`. Each successful crop and its JSONL record are committed immediately. An empty
grounding result (`[]`), malformed reply, or multiple boxes raises an error and stops the run;
earlier successful crops remain saved. For example, if `crop_05.png` fails, crops 01–04 and
their JSONL records are already on disk. `num_crops` keeps the expected total, so a partial
image is never marked complete. On `--resume`, saved crops 01–04 are kept and only crop 05 is
localized again. No pending state file is needed. Legacy outputs without completion counts
recheck the final image.

All backends share the instruction decomposition prompt. Its examples preserve the
existing object's posture (`man sitting`) and shape (`kite shaped like a butterfly`),
as required by the extraction rules. Qwen and Gemma use their official sampling profiles
with Thinking disabled:

| Backend / stage | Output token limit | Temperature | top_p | top_k | Presence penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5, both stages | 32768 | 0.7 | 0.8 | 20 | 1.5 |
| Gemma4, both stages | 2048 | 1.0 | 0.95 | 64 | 0 |

The Qwen limits follow the official recommended output budgets; Gemma's 2048-token limit
is a project setting, not an official maximum. The Qwen presence penalty is implemented
as an additive logit penalty on tokens already generated in the current answer, excluding
the input prompt; Transformers does not natively expose this parameter. Repetition
penalty remains 1.0. Existing checkpoint EOS and suppression settings are preserved.
See the official [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)
and [Gemma4](https://huggingface.co/google/gemma-4-12B-it#best-practices) profiles.

Gemma4 uses the maximum supported image token budget of 1120
(`image_processor.max_soft_tokens`); this controls image input detail separately from
the output generation limit. Qwen retains its checkpoint-native image processing.
`crop_image.py --seed` defaults to 42. Sampling can still vary with batch composition,
hardware, or resume position. Use a fresh output directory after changing a prompt or
generation profile; old complete records do not record these settings and are skipped
by `--resume`. Incomplete records are still checked for decomposition mismatches.
A string value in either decomposition field is treated as a one-element list for all
backends. Empty values, unequal list lengths, and invalid element types still raise errors;
no missing targets or instructions are filled in.
Both backends share a grounding prompt body and user-only messages; only the requested coordinate
format differs:

| Backend | Grounding output |
| --- | --- |
| `qwen35` | `bbox_2d`: `[x1, y1, x2, y2]`, normalized to 0–1000 |
| `gemma4` | `box_2d`: `[y1, x1, y2, x2]`, normalized to 0–1000 |

All backends use one grounding pass on the full image without Part Refinement and require
exactly one valid box per target. They share box selection, finite/range/positive-area
validation, pixel conversion, and strict failure without retries or parent-box fallback.
Model-specific parsers convert native coordinate formats to normalized xyxy.
Image loading supplies full RGB images; checkpoint-native
processors and tokenizers produce different model inputs.

Each backend decomposes the same original editing instructions itself and grounds its own
extracted referring phrases. The prompts and processing rules are shared; extracted
`refer_object` values may differ because instruction understanding is part of the evaluation.
Use distinct output paths for each `--vlm` so crops and resume records belong to that model.

To audit fresh Qwen3.5/Gemma4 replies on the two real-image datasets without modifying
existing crop records, run `experiment_script/audit_vlm_benchmarks.py --vlm qwen35
--output-dir reports/vlm_validation_qwen35` (or `gemma4` with its own output directory).
It saves every raw reply and failure, checks edit counts, and reports end-to-end IoU
against human crop boxes by edit slot. A well-formed reply or matching count is not a
semantic correctness guarantee; the human boxes include context padding. Only this audit
continues after individual invalid replies to count failures; production cropping still
raises an error. The audit never substitutes another model's or human boxes.

For the explicitly requested completion of the two real-image benchmarks, run
`python experiment_script/complete_vlm_crops.py` to preview the saved full-audit results,
then add `--apply` to create their four `crops_qwen35_vlm` / `crops_gemma4_vlm`
directories. An absent or invalid crop uses the other model's aligned result; if both
fail, it uses `crops_human`. Reviewed edit-slot mappings handle the three known Qwen
decomposition count errors. Existing result directories are never overwritten.
Every record contains `provenance`, and the report lists all substitutions and remaps.
These completed directories contain mixed model/human results; use the unchanged
audit outputs for pure model accuracy measurements. Legal boxes are retained even
when their IoU is low.

## 3.2 Base model + MIRAGE
Run MIRAGE on a base model, where `--model` is one of `flux2_klein9b`, `flux2_dev` and `qwen2511`:
```bash
python inference.py \
  --model flux2_klein9b \
  --image-root benchmark \
  --instruction-jsonl benchmark/annotations.jsonl \
  --crop-instruction-jsonl benchmark/crops/crop_instruction.jsonl \
  --output-dir results/flux2_klein9b
```

# 4. Evaluation
Use `--mask-alignment-policy full-frame` for MIRAGE outputs and `flux2-crop` for outputs of the official FLUX.2 pipelines. The OpenAI judges read the API key from `OPENAI_API_KEY`.

## LLM-based Metrics
**PF**, **Cons** and **PQ** are all judged by GPT models:
```bash
python metrics/evaluate_gpt.py \
  --annotations-jsonl benchmark/annotations.jsonl \
  --crop-instruction-jsonl benchmark/crops/crop_instruction.jsonl \
  --input-image-root benchmark \
  --edited-image-root results/your_results \
  --result-dir metrics/results/gpt/your_results \
  --mask-alignment-policy full-frame
```

Alternatively, **PF** and **Cons** can be computed with a local EditScore model, while **PQ** is still evaluated with the GPT API. The EditScore judge runs on [vLLM](https://github.com/vllm-project/vllm) by default. vLLM does not support Transformers 5 yet, so install it in a separate environment (we used `pip install vllm==0.11.0 transformers==4.57.6` on top of `requirements.txt`), or pass `--sc-backbone qwen3vl` to run the judge with Transformers in the main environment.
```bash
python metrics/evaluate_editscore.py \
  --annotations-jsonl benchmark/annotations.jsonl \
  --crop-instruction-jsonl benchmark/crops/crop_instruction.jsonl \
  --input-image-root benchmark \
  --edited-image-root results/your_results \
  --result-dir metrics/results/editscore/your_results \
  --mask-alignment-policy full-frame \
  --sc-model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --sc-lora-path EditScore/EditScore-Qwen3-VL-8B-Instruct \
  --pq-model gpt-5.1 \
  --num-pass 3
```

## Traditional Metrics
Compute pixel-level similarity metrics:

```bash
# MSE, LPIPS, PSNR...
python metrics/evaluate_traditional.py \
  --annotations-jsonl benchmark/annotations.jsonl \
  --crop-instruction-jsonl benchmark/crops/crop_instruction.jsonl \
  --input-image-root benchmark \
  --edited-image-root results/your_results \
  --result-csv metrics/results/traditional/metric_summary.csv
```

# Citation
If you use this code or the benchmark in your research, please cite our paper:
```
@misc{liu2026miragebenchmarkingaligningmultiinstance,
      title={MIRAGE: Benchmarking and Aligning Multi-Instance Image Editing}, 
      author={Ziqian Liu and Stephan Alaniz},
      year={2026},
      eprint={2604.05180},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.05180}, 
}
```

# License
This code is released under the [MIT License](LICENSE). The files in `metrics/editscore` are adapted from [EditScore](https://github.com/VectorSpaceLab/EditScore) and remain under the [Apache License 2.0](metrics/editscore/LICENSE).
