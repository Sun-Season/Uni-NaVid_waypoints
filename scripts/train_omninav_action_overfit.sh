#!/bin/bash
# 过拟合测试脚本：用少量数据验证训练 pipeline 是否正常工作
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
OUTPUT_DIR="/mnt/dataset/shuzheng/Uni-NaVid_waypoints/checkpoints/omninav_action_overfit_test"

# Instruction types and agent types (comma-separated)
INST_TYPES="original,concise,first_person,verbose"
AGENT_TYPES="car,dog,human"

# Set PYTHONPATH
export PYTHONPATH="/mnt/dataset/shuzheng/Uni-NaVid_waypoints:$PYTHONPATH"

# 单卡训练，50 轮，小 batch size
deepspeed --num_gpus=1 /mnt/dataset/shuzheng/Uni-NaVid_waypoints/uninavid/train/train_mem.py \
    --deepspeed /mnt/dataset/shuzheng/Uni-NaVid_waypoints/scripts/zero2.json \
    # DeepSpeed 配置文件路径，使用 ZeRO Stage 2 优化策略
    --model_name_or_path $MODEL_PATH \
    # 预训练模型路径，加载 Uni-NaVid 7B 基础模型
    --vision_tower $VISION_TOWER \
    # 视觉编码器权重路径，使用 EVA-ViT-G 作为视觉塔
    --image_processor $IMAGE_PROCESSOR \
    # 图像预处理器配置路径，使用 CLIP ViT-L/14 的预处理方式
    --compress_type "grid:2" \
    # 视觉特征压缩方式，grid:2 表示将特征图按 2x2 网格压缩
    --mm_projector_type mlp2x_gelu \
    # 多模态投影器类型，使用 2 层 MLP + GELU 激活函数
    --mm_vision_select_layer -2 \
    # 选择视觉编码器的倒数第 2 层特征作为输出
    --mm_use_im_start_end False \
    # 是否在图像 token 前后添加特殊标记 <im_start>/<im_end>
    --mm_use_im_patch_token False \
    # 是否使用图像 patch token 标记
    --tune_vision_encoder False \
    # 是否微调视觉编码器，False 表示冻结视觉编码器参数
    --version imgsp_v1 \
    # 对话模板版本，imgsp_v1 是图像空间理解的对话格式
    --use_omninav_action True \
    # 是否使用 OmniNav Action 数据集进行训练
    --omninav_action_root $ACTION_ROOT \
    # OmniNav Action 数据集的标注文件根目录
    --omninav_action_video_root $VIDEO_ROOT \
    # OmniNav Action 数据集的视频文件根目录
    --omninav_action_inst_types $INST_TYPES \
    # 使用的指令类型：original(原始)、concise(简洁)、first_person(第一人称)、verbose(详细)
    --omninav_action_agent_types $AGENT_TYPES \
    # 使用的智能体类型：car(汽车)、dog(狗)、human(人类)
    --omninav_action_max_frames 16 \
    # 每个视频采样的最大帧数
    --image_aspect_ratio pad \
    # 图像宽高比处理方式，pad 表示填充到正方形
    --bf16 True \
    # 是否使用 bfloat16 混合精度训练，节省显存并加速训练
    --output_dir $OUTPUT_DIR \
    # 模型检查点和日志的输出目录
    --num_train_epochs 50 \
    # 训练总轮数，过拟合测试使用较多轮数确保收敛
    --per_device_train_batch_size 8 \
    # 每个 GPU 的训练 batch size
    --per_device_eval_batch_size 4 \
    # 每个 GPU 的评估 batch size
    --gradient_accumulation_steps 2 \
    # 梯度累积步数，有效 batch size = 8 * 1 * 2 = 16
    --evaluation_strategy "no" \
    # 评估策略，"no" 表示训练过程中不进行评估
    --save_strategy "steps" \
    # 模型保存策略，按步数保存
    --save_steps 500 \
    # 每 500 步保存一次检查点
    --save_total_limit 3 \
    # 最多保留 3 个检查点，节省磁盘空间
    --learning_rate 5e-5 \
    # 初始学习率
    --weight_decay 0.01 \
    # 权重衰减系数，用于 L2 正则化防止过拟合
    --warmup_ratio 0.03 \
    # 学习率预热比例，前 3% 的步数用于线性预热
    --lr_scheduler_type "cosine" \
    # 学习率调度器类型，使用余弦退火策略
    --logging_steps 1 \
    # 每 1 步记录一次训练日志
    --model_max_length 2048 \
    # 模型最大序列长度（token 数）
    --gradient_checkpointing True \
    # 是否启用梯度检查点，用时间换显存
    --dataloader_num_workers 4 \
    # 数据加载器的工作进程数
    --lora_enable True \
    # 是否启用 LoRA (Low-Rank Adaptation) 微调
    --lora_r 128 \
    # LoRA 的秩 (rank)，越大表示可训练参数越多
    --lora_alpha 256 \
    # LoRA 的缩放因子，通常设为 2 * lora_r
    --lora_dropout 0.05 \
    # LoRA 层的 dropout 比例，防止过拟合
    --tune_mm_projector_with_lora True
    # 是否同时用 LoRA 微调多模态投影器
