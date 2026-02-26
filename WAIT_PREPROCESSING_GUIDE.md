# Wait Segment Preprocessing Guide

本文档说明如何预处理 OmniNavBench 数据以支持"等待"行为的学习。

## 概述

### 问题
- 原始数据按空间间隔记录 waypoints（~0.05m）
- 等待时几乎没有 waypoints（长时间间隔）
- 模型无法学会"等待"行为

### 解决方案
**混合策略**：
- **运动段**：保持原始 waypoints（无插值误差）
- **等待段**：时间插值（生成多个相同位置的 waypoints）

## 预处理步骤

### 1. 运行预处理脚本

```bash
python preprocess_wait_segments.py \
    /path/to/OmniNavBench/OminNavBenchData/train \
    /path/to/OmniNavBenchData_preprocessed/train \
    --min_wait_s 2.0 \
    --yaw_skip_deg 20.0 \
    --wait_interval 0.5 \
    --instruction_types original \
    --agent_types human,car,dog
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--min_wait_s` | 2.0 | 等待时间阈值（秒）。只有时间间隔 ≥ 2s 的段才会被识别为等待 |
| `--yaw_skip_deg` | 20.0 | 角度变化阈值（度）。角度变化 > 20° 的段被视为转向，不算等待 |
| `--wait_interval` | 0.5 | 等待段插值间隔（秒）。等待时每 0.5s 生成一个 waypoint |
| `--instruction_types` | original | 要处理的指令类型 |
| `--agent_types` | human,car,dog | 要处理的 agent 类型 |

### 2. 等待检测逻辑

脚本使用与 UniNavBench replay 相同的逻辑：

**判定为等待的条件**（同时满足）：
1. ✅ 相邻 waypoints 时间间隔 ≥ 2.0 秒
2. ✅ 角度变化 < 20 度（排除原地转向）

**示例**：
```
原始数据:
wp[0]: t=0.0s, pos=(0, 0), yaw=180°
wp[1]: t=6.5s, pos=(0.05, 0), yaw=180°  <- 间隔 6.5s, 角度变化 0°
判定: 等待段 ✓

预处理后:
wp[0]: t=0.0s, pos=(0, 0), yaw=180°
wp[1]: t=0.5s, pos=(0, 0), yaw=180°  <- 插值: 等待
wp[2]: t=1.0s, pos=(0, 0), yaw=180°  <- 插值: 等待
...
wp[12]: t=6.0s, pos=(0, 0), yaw=180°  <- 插值: 等待
wp[13]: t=6.5s, pos=(0.05, 0), yaw=180°
```

### 3. 输出统计

预处理完成后会显示统计信息：

```
Statistics by agent type:
  HUMAN:
    Files: 150
    Wait segments: 450
    Movement segments: 18000
    Interpolated waypoints: 5400
    Wait ratio: 2.4%

  CAR:
    Files: 150
    Wait segments: 600
    Movement segments: 16000
    Interpolated waypoints: 9000
    Wait ratio: 3.6%

  DOG:
    Files: 150
    Wait segments: 300
    Movement segments: 19000
    Interpolated waypoints: 3600
    Wait ratio: 1.5%
```

**关键指标**：
- `Wait ratio`: 等待段占总段数的比例
- `Interpolated waypoints`: 插值生成的 waypoints 数量

## 训练配置

### 使用预处理数据训练

```bash
python -m uninavid.train.train \
    --use_waypoint_head True \
    --use_omninav_waypoint True \
    --omninav_data_base_path /path/to/OminNavBenchData_preprocessed/train \
    --omninav_video_base_path /path/to/OmniNavBenchVideos/train \
    --num_waypoints 5 \
    --omninav_max_frames 32 \
    --waypoint_loss_weight 1.0 \
    --angle_loss_weight 0.5 \
    --arrive_loss_weight 0.5 \
    # ... 其他参数
```

**重要**：使用预处理后的数据路径（`OminNavBenchData_preprocessed`）

### DataLoader 行为

DataLoader 会使用固定 stride 采样：

```python
# 例如 stride=10, num_waypoints=5
current_idx = 50
future_waypoints = [60, 70, 80, 90, 100]
```

**运动段**：
- Waypoints 密集（~0.05m/个）
- stride=10 → 预测约 0.5m 处的 5 个点

**等待段**（预处理后）：
- Waypoints 稀疏（0.5s/个）
- stride=10 → 预测约 5s 内的 5 个点
- **所有 5 个点位置相同** → 模型学会预测"等待"

## 预期效果

### 训练时
- 模型看到"有障碍物"的视频 + 等待段数据
- 学习：这种视觉输入 → 预测多个相同位置的 waypoints

### 推理时
- 模型看到"有障碍物"的视频
- 预测：5 个 waypoints 都在当前位置附近 → **等待行为**
- 障碍物移开后，模型会预测移动

## 参数调优

### 调整等待阈值

如果发现等待段太少或太多，调整 `min_wait_s`：

| 现象 | 调整 |
|------|------|
| 等待段太少 (< 1%) | 降低阈值：`--min_wait_s 1.5` |
| 等待段太多 (> 10%)，包含很多减速段 | 提高阈值：`--min_wait_s 2.5` |
| 合理范围 | 保持 2.0s |

### 调整插值间隔

`wait_interval` 应该与训练时的 stride 对应：

| Stride | 建议 wait_interval |
|--------|-------------------|
| 5 | 0.25s |
| 10 | 0.5s（推荐） |
| 20 | 1.0s |

**公式**：预测时间范围 = stride × wait_interval × num_waypoints

例如：stride=10, interval=0.5s, num=5 → 预测 25s 内的轨迹

## 验证

### 检查预处理结果

随机抽查一个预处理后的文件：

```bash
python3 -c "
import json
with open('/path/to/preprocessed/file.json') as f:
    data = json.load(f)

wps = data['scenarios'][0]['robots']['entries'][0]['rb_gt_waypoints']
print(f'Total waypoints: {len(wps)}')

# 检查是否有等待段
for i in range(len(wps)-4):
    if (wps[i]['xyz'] == wps[i+1]['xyz'] == wps[i+2]['xyz'] and
        wps[i+1]['time_s'] - wps[i]['time_s'] < 1.0):
        print(f'Wait detected at idx {i}: t={wps[i][\"time_s\"]:.1f}s')
"
```

### 训练时监控

观察训练 loss：
- `waypoint_loss`：应该能正常下降
- 如果 loss 不下降，可能是等待段过多，导致样本不平衡

## 注意事项

1. **预处理不影响原始数据**
   - 原始数据保持不变
   - 预处理输出到新目录

2. **视频不需要预处理**
   - 视频路径仍然使用原始路径
   - DataLoader 会自动对齐

3. **单位转换已在 DataLoader 中处理**
   - 预处理保留原始单位
   - 训练时自动转换为米

4. **可以多次预处理**
   - 尝试不同的 `min_wait_s` 参数
   - 观察等待段比例和训练效果

## 下一步

预处理完成后：
1. 检查统计信息（wait ratio 是否合理）
2. 开始训练
3. 观察模型是否学会预测等待行为
4. 根据效果调整参数

详细训练流程见 [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)。
