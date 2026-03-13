#!/bin/bash

# Training script for VLN Action Text prediction with LoRA
# Uses text generation to predict discrete actions (following original Uni-NaVid)

source activate base
conda activate uninavid

# Set paths
MODEL_PATH="/mnt/dataset/wj_zqc/VLN/model/uninavid-7b-full-224-video-fps-1-grid-2"
VISION_TOWER="/mnt/dataset/wj_zqc/VLN/model/eva_vit_g.pth"
DATA_PATH="/mnt/dataset/wj_zqc/ssz/Uni-NaVid_waypoints/passed_samples"
OUTPUT_DIR="output/vln_action_text_lora_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters
NUM_EPOCHS=10
BATCH_SIZE=8
GRAD_ACCUM=2
LEARNING_RATE=5e-5
MAX_FRAMES=16
SAMPLE_STRIDE=60  # Frame stride for waypoint-to-action conversion
MIN_HISTORY=4

# Action prediction settings (waypoint-to-action mode)
NUM_FUTURE_ACTIONS=4  # Fixed 4 actions output
ACTION_STRIDE=30  # Frame stride for waypoint-to-action conversion

# LoRA hyperparameters
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training with DeepSpeed and LoRA
CUDA_VISIBLE_DEVICES=0 deepspeed train_vln_action_text.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version imgsp_v1 \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
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
    --report_to none \
    --max_frames $MAX_FRAMES \
    --sample_stride $SAMPLE_STRIDE \
    --min_history_frames $MIN_HISTORY \
    --num_future_actions $NUM_FUTURE_ACTIONS \
    --action_stride $ACTION_STRIDE \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed! Output saved to: $OUTPUT_DIR"
