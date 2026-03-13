#!/bin/bash
# Training script for OmniNavBench action prediction with LoRA
#
# Usage:
#   1. First preprocess the data:
#      python scripts/preprocess_omninav_actions.py \
#          --data_root /mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchData \
#          --output_root /mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchActionData \
#          --split train
#
#   2. Then run training:
#      bash scripts/train_omninav_action.sh

# Model path (Uni-NaVid base model)
MODEL_PATH="/mnt/dataset/shuhzeng/Uni-NaVid"

# Data paths
ACTION_ROOT="/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchActionData"
VIDEO_ROOT="/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchVideos"

# Output directory
OUTPUT_DIR="./checkpoints/omninav_action_lora"

# Instruction types and agent types (comma-separated)
INST_TYPES="original,concise,first_person,verbose"
AGENT_TYPES="car,dog,human"

# Training
deepspeed --num_gpus=1 train_omninav_action.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version imgsp_v1 \
    --action_root $ACTION_ROOT \
    --video_root $VIDEO_ROOT \
    --split train \
    --inst_types $INST_TYPES \
    --agent_types $AGENT_TYPES \
    --max_frames 16 \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05
