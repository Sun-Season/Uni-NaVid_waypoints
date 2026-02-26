# Waypoint Training Guide

本文档说明如何使用 OmniNavBench 数据训练 Waypoint 模型。

## 当前状态

✅ **Pipeline 已打通**，可以直接开始训练，无需预处理数据。

### 已完成的功能

- [x] Waypoint Head 模型实现
- [x] OmniNavBench DataLoader 实现
- [x] 单位转换处理（自动处理 cm 和 m）
- [x] 帧率自动检测和转换
- [x] 训练脚本集成
- [x] 推理脚本实现

## 数据要求

### 目录结构

```
OmniNavBench/
├── OminNavBenchData/
│   └── train/
│       ├── original/          # 或 concise, verbose, first_person
│       │   ├── human/
│       │   │   ├── <scene_name>/
│       │   │   │   └── *.json
│       │   ├── car/
│       │   └── dog/
└── OmniNavBenchVideos/
    └── train/
        ├── human/
        │   ├── <scene_name>/
        │   │   ├── <episode_name>/
        │   │   │   └── rgb.mp4
        ├── car/
        └── dog/
```

### 数据格式

JSON 文件包含：
- `scenarios[0].scene.units_in_meters`: 坐标单位（1 或 0.01）
- `scenarios[0].task.navigation.instruction`: 导航指令
- `scenarios[0].robots.entries[0].rb_gt_waypoints`: 轨迹 waypoints
  - `frame`: 帧号
  - `xyz`: 位置 [x, y, z]
  - `yaw_deg`: 朝向角度
  - `time_s`: 时间戳

## 训练参数说明

### 必需参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--use_waypoint_head` | 启用 waypoint head | `True` |
| `--use_omninav_waypoint` | 使用 OmniNavBench 数据 | `True` |
| `--omninav_data_base_path` | JSON 数据路径 | `/path/to/OminNavBenchData/train` |
| `--omninav_video_base_path` | 视频数据路径 | `/path/to/OmniNavBenchVideos/train` |

### 可选参数

| 参数 | 说明 | 默认值 | 建议值 |
|------|------|--------|--------|
| `--num_waypoints` | 预测的 waypoint 数量 | 5 | 5-10 |
| `--omninav_max_frames` | 输入的最大历史帧数 | 32 | 16-64 |
| `--omninav_instruction_type` | 指令类型 | `original` | `original`/`concise`/`verbose`/`first_person` |
| `--omninav_agent_types` | Agent 类型（逗号分隔） | `human,car,dog` | 根据需要选择 |
| `--waypoint_loss_weight` | Waypoint 位置损失权重 | 1.0 | 0.5-2.0 |
| `--angle_loss_weight` | 角度损失权重 | 0.5 | 0.3-1.0 |
| `--arrive_loss_weight` | 到达损失权重 | 0.5 | 0.3-1.0 |
| `--use_lm_loss_for_waypoint` | 是否同时训练语言模型 | `False` | `False`（推荐先只训练 waypoint head） |

## 训练流程

### 1. 测试 Pipeline

在开始训练前，先测试数据加载是否正常：

```bash
python test_waypoint_pipeline.py
```

如果测试通过，会显示：
- ✓ 数据路径检查
- ✓ Tokenizer 加载
- ✓ Image processor 加载
- ✓ Dataset 创建
- ✓ Sample 加载
- ✓ Batch collation

### 2. 阶段 1：训练 Waypoint Head

**目标**：只训练新增的 waypoint head，冻结 VLM 和 Vision Encoder

```bash
python -m uninavid.train.train \
    --model_name_or_path model_zoo/uninavid-7b \
    --version v1 \
    --use_waypoint_head True \
    --use_omninav_waypoint True \
    --omninav_data_base_path /path/to/OminNavBenchData/train \
    --omninav_video_base_path /path/to/OmniNavBenchVideos/train \
    --omninav_instruction_type original \
    --omninav_agent_types human,car,dog \
    --omninav_max_frames 32 \
    --num_waypoints 5 \
    --waypoint_loss_weight 1.0 \
    --angle_loss_weight 0.5 \
    --arrive_loss_weight 0.5 \
    --use_lm_loss_for_waypoint False \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/uninavid-waypoint-stage1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

**训练策略**：
- 冻结 Vision Encoder 和 LLM
- 只训练 `waypoint_head` 和 `query_action` 参数
- 学习率可以设置较高（2e-4）

### 3. 阶段 2：微调整个模型（可选）

如果阶段 1 效果不错，可以进行端到端微调：

```bash
python -m uninavid.train.train \
    --model_name_or_path ./checkpoints/uninavid-waypoint-stage1/checkpoint-xxx \
    # ... 其他参数同上 ...
    --use_lm_loss_for_waypoint True \  # 启用语言模型损失
    --learning_rate 2e-5 \  # 降低学习率
    --output_dir ./checkpoints/uninavid-waypoint-stage2
```

## 关键设计说明

### 1. 历史帧采样策略

DataLoader 会自动处理历史帧：
- 从轨迹起点到当前位置的所有帧
- 如果帧数超过 `max_frames`，会均匀采样
- 保留第一帧和最后一帧（当前帧）

**示例**：
```python
# 如果轨迹有 100 个 waypoints，当前在第 50 个
# max_frames = 32
# 会加载 waypoint 0-50 对应的视频帧，然后采样到 32 帧
```

### 2. Waypoint 预测数量

`num_waypoints` 控制预测多少个未来 waypoints：
- 默认 5 个
- 从当前位置的下一个 waypoint 开始
- 如果剩余 waypoints 不足，会用零填充

**示例**：
```python
# 当前在 waypoint 50，num_waypoints=5
# 预测 waypoint 51, 52, 53, 54, 55
# 如果轨迹只到 waypoint 52，则 53-55 用零填充，arrive=1
```

### 3. 单位转换

DataLoader 自动处理不同的坐标单位：
- `units_in_meters: 1` → 坐标已经是米
- `units_in_meters: 0.01` → 坐标是厘米，自动乘以 0.01
- 所有输出的 waypoint 位置统一为米

### 4. 坐标系

Waypoints 使用**机器人坐标系**（相对坐标）：
- **x 轴**：前进方向（正值向前）
- **y 轴**：左右方向（正值向左）
- **yaw**：相对朝向（逆时针为正）

输出格式：
- `positions`: [N, 2] - (x, y) 相对位置，单位米
- `yaws`: [N, 2] - (sin, cos) 相对角度
- `arrive`: [N] - 到达概率（0-1）

## 监控训练

### 关键指标

训练时会输出以下 loss：
- `loss`: 总损失
- `lm_loss`: 语言模型损失（如果启用）
- `waypoint_loss`: 位置预测损失（L1）
- `angle_loss`: 角度预测损失（cosine）
- `arrive_loss`: 到达预测损失（BCE）

### 预期值

- `waypoint_loss`: 应该逐渐降低到 0.1-0.5（米级误差）
- `angle_loss`: 应该逐渐降低到 0.1-0.3
- `arrive_loss`: 应该逐渐降低到 0.1-0.3

## 推理测试

训练完成后，使用推理脚本测试：

```bash
python inference_waypoint.py <test_case_dir> <output_dir> \
    --model_path ./checkpoints/uninavid-waypoint-stage1/checkpoint-xxx \
    --num_waypoints 5
```

## 常见问题

### Q1: 需要预处理数据吗？

**不需要**。DataLoader 会在训练时动态加载和处理数据：
- 自动读取 JSON 和视频
- 自动转换坐标单位
- 自动采样历史帧
- 自动计算相对 waypoints

### Q2: 如何选择 max_frames？

建议根据 GPU 显存调整：
- 16 帧：适合显存较小的 GPU（<16GB）
- 32 帧：推荐值，平衡性能和显存
- 64 帧：适合显存较大的 GPU（>40GB）

### Q3: 如何选择 num_waypoints？

建议根据任务特点：
- 短期规划（室内导航）：5 个
- 中期规划：10 个
- 长期规划：15-20 个

注意：waypoints 越多，训练越困难。

### Q4: 训练时显存不足怎么办？

尝试以下方法：
1. 减少 `omninav_max_frames`（32 → 16）
2. 减少 `per_device_train_batch_size`（1 → 1，已经最小）
3. 增加 `gradient_accumulation_steps`（16 → 32）
4. 启用 `gradient_checkpointing True`（已启用）
5. 使用 DeepSpeed ZeRO-2 或 ZeRO-3

### Q5: 可以混合使用不同的 agent 类型吗？

可以。通过 `--omninav_agent_types` 参数控制：
- 全部：`human,car,dog`
- 只有人：`human`
- 人和车：`human,car`

不同 agent 的运动特性不同，建议先单独训练，再混合训练。

## 下一步

训练完成后，可以：
1. 在 OmniNavBench 测试集上评估
2. 集成到 VLN-CE 或其他导航任务
3. 部署到真实机器人

详细信息见 [WAYPOINT_HEAD_README.md](./WAYPOINT_HEAD_README.md)。
