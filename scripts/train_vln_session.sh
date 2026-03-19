#!/bin/bash

# Training script for VLN Session waypoint prediction
# Example usage for training on collected VLN session data
source activate base
conda activate uninavid

# Set paths
MODEL_PATH="/mnt/dataset/shuzheng/model/uninavid-7b-full-224-video-fps-1-grid-2"  # Pretrained model
VISION_TOWER="/mnt/dataset/shuzheng/model/eva_vit_g.pth"  # Vision tower
DATA_PATH="/mnt/dataset/shuzheng/Uni-NaVid_waypoints/passed_samples"  # VLN session directory
OUTPUT_DIR="output/vln_session_waypoint_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-5
MAX_FRAMES=16
NUM_WAYPOINTS=5
WAYPOINT_STRIDE=5

# Waypoint loss weights
POS_WEIGHT=1.0
YAW_WEIGHT=0.5
ARRIVE_WEIGHT=0.5

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training with DeepSpeed
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train_vln_session.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version imgsp_v1 \
    --data_path $DATA_PATH \
    --vision_tower $VISION_TOWER \
    --image_processor ./uninavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --compress_type "grid:2" \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
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
    --report_to wandb \
    --max_frames $MAX_FRAMES \
    --num_future_waypoints $NUM_WAYPOINTS \
    --waypoint_stride $WAYPOINT_STRIDE \
    --waypoint_position_weight $POS_WEIGHT \
    --waypoint_yaw_weight $YAW_WEIGHT \
    --waypoint_arrive_weight $ARRIVE_WEIGHT \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed! Output saved to: $OUTPUT_DIR"
