#!/bin/bash

DISTRIBUTED_ARGS="
    --nproc_per_node 4 \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

# Rotate train.log before starting new run
LOG_FILE="train.log"
MAX_BACKUPS=100

if [ -f "$LOG_FILE" ]; then
    # Shift older backups
    for ((i=MAX_BACKUPS; i>=1; i--)); do
        if [ -f "${LOG_FILE}.${i}" ]; then
            mv "${LOG_FILE}.${i}" "${LOG_FILE}.$((i+1))"
        fi
    done
    mv "$LOG_FILE" "${LOG_FILE}.1"
fi

export HF_HOME=/ocean/projects/cis210027p/qwang20/open-r1-multimodal


torchrun \
    --nproc_per_node="4" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir results \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name /ocean/projects/cis210027p/qwang20/R1-Multimodal-Journey/geo_data \
    --max_prompt_length 2048 \
    --num_generations 8 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 1000000 \
    --save_total_limit 30 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-7B-GRPO-8k \
    >> train.log 2>&1
