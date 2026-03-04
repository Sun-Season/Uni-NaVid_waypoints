#!/bin/bash

# Model and data paths
PREV_MODEL="/mnt/dataset/wj_zqc/VLN/model/uninavid-7b-full-224-video-fps-1-grid-2"
VISION_TOWER="/mnt/dataset/wj_zqc/VLN/model/eva_vit_g.pth"
DATA_BASE_PATH="/mnt/dataset/ssz/OmniNavBench/OmniNavBenchData/train"
VIDEO_BASE_PATH="/mnt/dataset/ssz/OmniNavBench/OmniNavBenchVideos/train"
OUTPUT_DIR="./model_zoo/uninavid-7b-omninav-waypoint"

# Training hyperparameters
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_FRAMES=16
LEARNING_RATE=1e-5
NUM_EPOCHS=3

# Waypoint parameters
NUM_WAYPOINTS=5
WAYPOINT_STRIDE=5
VIDEO_FPS=30

# Loss weights (already defined in ModelArguments)
WAYPOINT_LOSS_WEIGHT=1.0
ANGLE_LOSS_WEIGHT=0.5
ARRIVE_LOSS_WEIGHT=0.5

deepspeed uninavid/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $PREV_MODEL \
    --version imgsp_v1 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --vision_tower $VISION_TOWER \
    --image_processor ./uninavid/processor/clip-patch14-224 \
    --tune_vision_encoder False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --video_fps 30 \
    --compress_type "grid:2" \
    --use_waypoint_head True \
    --use_lm_loss_for_waypoint True \
    --num_waypoints $NUM_WAYPOINTS \
    --waypoint_loss_weight $WAYPOINT_LOSS_WEIGHT \
    --angle_loss_weight $ANGLE_LOSS_WEIGHT \
    --arrive_loss_weight $ARRIVE_LOSS_WEIGHT \
    --use_omninav_waypoint True \
    --omninav_data_base_path $DATA_BASE_PATH \
    --omninav_video_base_path $VIDEO_BASE_PATH \
    --omninav_agent_types "human,car,dog" \
    --omninav_max_frames $MAX_FRAMES \
    --omninav_video_fps $VIDEO_FPS \
    --omninav_num_future_waypoints $NUM_WAYPOINTS \
    --omninav_waypoint_stride $WAYPOINT_STRIDE \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb
