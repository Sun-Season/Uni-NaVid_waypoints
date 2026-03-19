source activate base
conda activate uninavid

# Model path (Uni-NaVid base model)
MODEL_PATH="/mnt/dataset/shuzheng/model/uninavid-7b-full-224-video-fps-1-grid-2"
VISION_TOWER="/mnt/dataset/shuzheng/model/eva_vit_g.pth"

# Data paths
ACTION_ROOT="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchActionData"
VIDEO_ROOT="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchVideos"

# Output directory
OUTPUT_DIR="/mnt/dataset/shuzheng/Uni-NaVid_waypoints/checkpoints/omninav_action_lora"

# Instruction types and agent types (comma-separated)
INST_TYPES="original,concise,first_person,verbose"
AGENT_TYPES="car,dog,human"

# Training (8 GPUs)
deepspeed --num_gpus=8 /mnt/dataset/shuzheng/Uni-NaVid_waypoints/train_omninav_action.py \
    --deepspeed /mnt/dataset/shuzheng/Uni-NaVid_waypoints/scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --vision_tower $VISION_TOWER \
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
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05
