# EditScore Reward Model Training Guide

This guide explains how to train EditScore reward models using LLaMA-Factory.

## 1. Environment Setup

### Clone LLaMA-Factory and Configure Virtual Environment

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
conda create -n llama-factory python=3.10
conda activate llama-factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

## 2. Directory Structure Configuration

Create necessary folders and files in the LLaMA-Factory root directory:

```bash
# Create log and output directories
mkdir -p logs
mkdir -p output

# Create training configuration directory
mkdir -p examples/train_editscore

# Copy training configuration files
cp EditScore/examples/EditScore-train/config/*.yaml examples/train_editscore/

# Copy training script
cp EditScore/examples/EditScore-train/train.sh .
```

## 3. Dataset Registration

Register the EditScore-Reward-Data dataset in `LLaMA-Factory/data/dataset_info.json`:

```json
"EditScore-Reward-Data": {
  "file_name": "/path/to/your/reward.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "images": "images"
  }
}
```

## 4. Training Configuration Description

### Single-Machine Training Configuration
- `editscore_7B.yaml` - Train EditScore-7B model (single machine)
- `editscore_qwen3_vl_4B_instruct.yaml` - Train EditScore_Qwen3_Vl_4B_Instruct model (single machine)
- `editscore_qwen3_vl_8B_instruct.yaml` - Train EditScore_Qwen3_Vl_8B_Instruct model (single machine)

### Multi-Machine Training Configuration
- `editscore_32B.yaml` - Train EditScore-32B model (two machines)
- `editscore_72B.yaml` - Train EditScore-72B model (two machines)

## 5. Start Training

### Single-Machine Training

```bash
# Modify experiment_name in train.sh to the corresponding configuration file name
# For example: name=editscore_7B
bash train.sh
```

### Multi-Machine Training

**Master node (rank=0):**
```bash
bash train.sh --rank=0 --world_size=2 --master_addr=MASTER_NODE_IP --master_port=29500
```

**Worker node (rank=1):**
```bash
bash train.sh --rank=1 --world_size=2 --master_addr=MASTER_NODE_IP --master_port=29500
```

## 6. Parameter Configuration

Users can modify the following parameters in the YAML configuration files as needed:

- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Learning rate
- `num_train_epochs`: Number of training epochs
- `max_samples`: Maximum number of samples
- `output_dir`: Output directory

## 7. Output Files

After training completion, model files will be saved in the corresponding output directories:
- Single-machine training: `LLaMA-Factory/output/model_name/`
- Log files: `LLaMA-Factory/logs/experiment_name_rank.log`


