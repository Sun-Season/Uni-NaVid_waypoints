#!/bin/bash

# Quick test script for VLN Action Text training (waypoint-to-action mode)
# Run this in uninavid conda environment

CUDA_VISIBLE_DEVICES=0 deepspeed train_vln_action_text.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/dataset/wj_zqc/VLN/model/uninavid-7b-full-224-video-fps-1-grid-2 \
    --version imgsp_v1 \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --data_path /mnt/dataset/shuzheng/Uni-NaVid_waypoints/passed_samples \
    --vision_tower /mnt/dataset/shuzheng/model/eva_vit_g.pth \
    --image_processor ./uninavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --compress_type "grid:2" \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir output/vln_action_text_test \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
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
    --max_frames 16 \
    --sample_stride 60 \
    --min_history_frames 4 \
    --num_future_actions 4 \
    --action_stride 5
