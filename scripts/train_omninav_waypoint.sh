#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate uninavid

# ============================================================
# Model Paths
# ============================================================
MODEL_PATH="/mnt/dataset/shuzheng/model/uninavid-7b-full-224-video-fps-1-grid-2"
VISION_TOWER="/mnt/dataset/shuzheng/model/eva_vit_g.pth"
IMAGE_PROCESSOR="/mnt/dataset/shuzheng/Uni-NaVid_waypoints/uninavid/processor/clip-patch14-224"

# ============================================================
# Data Paths
# ============================================================
DATA_BASE_PATH="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchData/train"
VIDEO_BASE_PATH="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchVideos/train"

# ============================================================
# Output Directory
# ============================================================
OUTPUT_DIR="/mnt/dataset/shuzheng/Uni-NaVid_waypoints/checkpoints/omninav_waypoint_lora"

# ============================================================
# Training Hyperparameters
# ============================================================
NUM_GPUS=8
BATCH_SIZE=8
GRAD_ACCUM=2
NUM_EPOCHS=3
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.05
LR_SCHEDULER="cosine"
MODEL_MAX_LENGTH=2048

# ============================================================
# Video/Frame Parameters
# ============================================================
MAX_FRAMES=16
VIDEO_FPS=30

# ============================================================
# Waypoint Head Parameters
# ============================================================
NUM_WAYPOINTS=5
WAYPOINT_STRIDE=5
WAYPOINT_LOSS_WEIGHT=1.0
ANGLE_LOSS_WEIGHT=0.5
ARRIVE_LOSS_WEIGHT=0.5
USE_LM_LOSS_FOR_WAYPOINT=True

# ============================================================
# OmniNav Dataset Parameters
# ============================================================
AGENT_TYPES="human,car,dog"
# INSTRUCTION_TYPES=""  # 留空表示使用所有类型
SAMPLE_STRIDE=5  # 每隔多少个 waypoint 采样一次（每个 waypoint 约 5cm，stride=5 约等于每 0.25m 采样）

# ============================================================
# LoRA Parameters
# ============================================================
LORA_ENABLE=True
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
TUNE_MM_PROJECTOR=True

# ============================================================
# Saving & Logging
# ============================================================
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=10
REPORT_TO="wandb"

# ============================================================
# Set PYTHONPATH
# ============================================================
export PYTHONPATH="/mnt/dataset/shuzheng/Uni-NaVid_waypoints:$PYTHONPATH"

# ============================================================
# Run Training
# ============================================================
deepspeed --num_gpus=$NUM_GPUS /mnt/dataset/shuzheng/Uni-NaVid_waypoints/uninavid/train/train_mem.py \
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
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --video_fps $VIDEO_FPS \
    --use_waypoint_head True \
    --use_lm_loss_for_waypoint $USE_LM_LOSS_FOR_WAYPOINT \
    --num_waypoints $NUM_WAYPOINTS \
    --waypoint_loss_weight $WAYPOINT_LOSS_WEIGHT \
    --angle_loss_weight $ANGLE_LOSS_WEIGHT \
    --arrive_loss_weight $ARRIVE_LOSS_WEIGHT \
    --use_omninav_waypoint True \
    --omninav_data_base_path $DATA_BASE_PATH \
    --omninav_video_base_path $VIDEO_BASE_PATH \
    --omninav_agent_types $AGENT_TYPES \
    --omninav_max_frames $MAX_FRAMES \
    --omninav_video_fps $VIDEO_FPS \
    --omninav_num_future_waypoints $NUM_WAYPOINTS \
    --omninav_waypoint_stride $WAYPOINT_STRIDE \
    --omninav_sample_stride $SAMPLE_STRIDE \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER \
    --logging_steps $LOGGING_STEPS \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --lora_enable $LORA_ENABLE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --tune_mm_projector $TUNE_MM_PROJECTOR \
    --report_to $REPORT_TO