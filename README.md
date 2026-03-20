# RefEdit: A Benchmark and Method for Improving Instruction-based Image Editing Model for Referring Expression
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxxx)

## Benchmark Access
We release **MIRAGE-Bench**, which can be downloaded [here](https://drive.google.com/file/d/1nOACtv7H3QXxE78ba9ok-5zZGUWB1vMW/view?usp=sharing). The benchmark contains 100 samples, each consisting of an image, a composite editing instruction formed by combining five sub-instructions, and the corresponding ground-truth mask. This benchmark is designed to evaluate image editing models in more complex referring-expression scenarios. 

Notably, the entire **MIRAGE-Bench** is constructed based on our proposed automatic data synthesis pipeline.

## 自动数据合成管道
### 图像描述
```

```

### 图像生成
请使用以下命令进行基于图像描述的图像合成：
```
python synthesis_pipeline/flux_t2i_generate.py \
  --jsonl synthesis_pipeline/source_prompts.jsonl \
  --results-dir synthesis_pipeline/benchmark \
```

### 编辑指令
```

```

### 图像掩码 
```

```
