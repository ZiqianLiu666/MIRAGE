#!/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# Activate conda environment of LLaMA-Factory
conda activate llama-factory


RANK=0
WORLD_SIZE=1
MASTER_ADDR="localhost"
MASTER_PORT=29500


while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank=*)
            RANK="${1#*=}"
            shift
            ;;
        --world_size=*)
            WORLD_SIZE="${1#*=}"
            shift
            ;;
        --master_addr=*)
            MASTER_ADDR="${1#*=}"
            shift
            ;;
        --master_port=*)
            MASTER_PORT="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

name=experiment_name
log_dir="LLaMA-Factory/logs"

log_file="${log_dir}/${name}_${RANK}.log"


CONFIG_YAML="LLaMA-Factory/examples/train_editscore/${name}.yaml"
FORCE_TORCHRUN=1 \
NNODES=${WORLD_SIZE} \
NODE_RANK=${RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
llamafactory-cli train ${CONFIG_YAML} 2>&1 | tee ${log_file}
