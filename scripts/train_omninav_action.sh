#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate uninavid

# Model path (Uni-NaVid base model)
MODEL_PATH="/mnt/dataset/shuzheng/model/uninavid-7b-full-224-video-fps-1-grid-2"
VISION_TOWER="/mnt/dataset/shuzheng/model/eva_vit_g.pth"
IMAGE_PROCESSOR="/mnt/dataset/shuzheng/Uni-NaVid_waypoints/uninavid/processor/clip-patch14-224"

# Data paths
ACTION_ROOT="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchActionData"
VIDEO_ROOT="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchVideos"

# Output directory
OUTPUT_DIR="/mnt/dataset/shuzheng/Uni-NaVid_waypoints/checkpoints/omninav_action_lora_official"

# Instruction types and agent types (comma-separated)
INST_TYPES="original,concise,first_person,verbose"
AGENT_TYPES="car,dog,human"

# Set PYTHONPATH
export PYTHONPATH="/mnt/dataset/shuzheng/Uni-NaVid_waypoints:$PYTHONPATH"

deepspeed --num_gpus=1 /mnt/dataset/shuzheng/Uni-NaVid_waypoints/uninavid/train/train_mem.py \
    --deepspeed /mnt/dataset/shuzheng/Uni-NaVid_waypoints/scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --image_processor $IMAGE_PROCESSOR \
    --compress_type "grid:2" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_vision_encoder False \
    --version imgsp_v1 \
    --use_omninav_action True \
    --omninav_action_root $ACTION_ROOT \
    --omninav_action_video_root $VIDEO_ROOT \
    --omninav_action_inst_types $INST_TYPES \
    --omninav_action_agent_types $AGENT_TYPES \
    --omninav_action_video_fps 1 \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 250 \
    --val_split_ratio 0.1 \
    --val_split_seed 42 \
    --val_split_by_episode True \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 3 \
    --learning_rate 1.5e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --tune_mm_projector True