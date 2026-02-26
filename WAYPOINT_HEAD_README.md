# Waypoint Head 改造文档

本文档记录了将 Uni-NaVid 从离散动作输出改造为连续 waypoint 输出的所有修改。

## 概述

原始 Uni-NaVid 输出离散动作（forward, left, right, stop），改造后输出连续 waypoints：
- 位置 (x, y)：相对于当前位置的偏移，单位为米
- 角度 (sin, cos)：相对于当前朝向的偏转
- 到达概率：是否已到达目标

## 新增文件

### 1. `uninavid/train/omninav_dataset.py`

OmniNavBench 数据集的 DataLoader，用于加载新的 waypoint 格式数据。

**主要类和函数：**

| 名称 | 说明 |
|------|------|
| `OmniNavDataArguments` | 数据参数配置类 |
| `calculate_trajectory_fps()` | 自动计算轨迹采集帧率（60/72/239 FPS） |
| `trajectory_frame_to_video_frame()` | 将轨迹帧号转换为视频帧号 |
| `compute_relative_waypoints()` | 计算机器人坐标系下的相对 waypoints |
| `OmniNavBenchDataset` | PyTorch Dataset 类 |
| `OmniNavDataCollator` | 批处理 Collator |
| `make_omninav_data_module()` | 创建训练数据模块 |

**数据格式：**
```python
# 输入
{
    'input_ids': torch.Tensor,      # 文本 token
    'labels': torch.Tensor,         # 标签
    'image': torch.Tensor,          # 视频帧 [T, C, H, W]
    'waypoint_positions': torch.Tensor,  # [N, 2] 位置 (x, y)
    'waypoint_yaws': torch.Tensor,       # [N, 2] 角度 (sin, cos)
    'waypoint_arrive': torch.Tensor,     # [N] 到达概率
}
```

**帧对齐说明：**
- 轨迹数据中的 `frame` 字段是仿真器内部帧号
- 视频是 30 FPS
- 使用 `trajectory_frame_to_video_frame()` 进行转换：
  ```python
  video_frame = int(traj_frame / (traj_fps / 30))
  ```

---

### 2. `uninavid/model/waypoint_head.py`

Waypoint 预测头模块，基于 MLP + Cross-Attention。

**主要类：**

| 名称 | 说明 |
|------|------|
| `WaypointConfig` | 模型配置，继承自 `LlavaConfig` |
| `WaypointHead` | Waypoint 预测头（MLP + Cross-Attention） |
| `LlavaWaypointForCausalLM` | 完整模型，继承自 `LlavaLlamaAttForCausalLM` |
| `WaypointPredictionOutput` | 输出数据类 |

**WaypointHead 结构：**
```
VLM Hidden States [B, L, D]
        ↓
Cross-Attention (query_action 作为 query)
        ↓
Action Features [B, D]
        ↓
    ┌───┴───┐───────┐
    ↓       ↓       ↓
position  angle   arrive
predictor predictor predictor
    ↓       ↓       ↓
[B,N,2]  [B,N,2]  [B,N]
(x,y)   (sin,cos) (prob)
```

**Loss 函数：**
```python
total_loss = lm_loss 
           + waypoint_loss_weight * L1_loss(pred_pos, gt_pos)
           + angle_loss_weight * (1 - cosine_similarity(pred_angle, gt_angle))
           + arrive_loss_weight * BCE_loss(pred_arrive, gt_arrive)
```

**预训练权重兼容性：**
- 可以加载原始 Uni-NaVid 权重（`strict=False`）
- 新增的 waypoint_head 会随机初始化

---

### 3. `inference_waypoint.py`

Waypoint 推理脚本。

**主要类和函数：**

| 名称 | 说明 |
|------|------|
| `load_waypoint_model()` | 加载 waypoint 模型 |
| `WaypointAgent` | 推理 Agent |
| `draw_waypoints_fpv()` | 可视化 waypoints |

**使用方法：**
```bash
python inference_waypoint.py <test_case_dir> <output_dir> \
    --model_path model_zoo/uninavid-waypoint-7b \
    --num_waypoints 5
```

**输出格式：**
```python
{
    'step': int,                    # 当前步数
    'positions': np.ndarray,        # [N, 2] 位置
    'angles': np.ndarray,           # [N] 角度（弧度）
    'arrive': np.ndarray,           # [N] 到达概率
    'path': List[List[float]],      # 轨迹 [[x, y, yaw], ...]
}
```

---

## 关键修改文件

以下文件保持不变，waypoint 模型通过继承复用：

- `uninavid/model/language_model/llava_llama_vid.py` - 基础模型（复用）
- `uninavid/model/uninavid_arch.py` - 模型架构（复用）
- `uninavid/model/builder.py` - 已修改：支持 `llava_waypoint` 模型加载
- `uninavid/model/__init__.py` - 已修改：导出 waypoint 模型类
- `uninavid/train/train.py` - 已修改：接入 waypoint 模型与 OmniNavBench DataLoader 分支

---

## 训练流程

### 1. 准备数据

确保 OmniNavBench 数据目录结构：
```
OmniNavBench/
├── OminNavBenchData/
│   └── train/
│       ├── original/
│       │   ├── human/
│       │   ├── car/
│       │   └── dog/
│       ├── concise/
│       ├── verbose/
│       └── first_person/
└── OmniNavBenchVideos/
    └── train/
        ├── human/
        ├── car/
        └── dog/
```

### 2. 配置参数

```python
from uninavid.train.omninav_dataset import OmniNavDataArguments

data_args = OmniNavDataArguments(
    data_base_path="/path/to/OmniNavBench/OminNavBenchData/train",
    video_base_path="/path/to/OmniNavBench/OmniNavBenchVideos/train",
    instruction_type="original",  # 或 concise, verbose, first_person
    agent_types=['human', 'car', 'dog'],
    video_fps=30,
    max_frames=32,
    num_future_waypoints=5,
)
```

使用 `train.py` 启动 waypoint 训练时，需要显式打开两个开关：

```bash
--use_waypoint_head True \
--use_omninav_waypoint True \
--omninav_data_base_path <.../OminNavBenchData/train> \
--omninav_video_base_path <.../OmniNavBenchVideos/train> \
--num_waypoints 5
```

### 3. 创建模型

```python
from uninavid.model.waypoint_head import LlavaWaypointForCausalLM, WaypointConfig

# 从预训练权重加载
config = WaypointConfig.from_pretrained("model_zoo/uninavid-7b")
config.num_waypoints = 5
config.waypoint_loss_weight = 1.0
config.angle_loss_weight = 0.5
config.arrive_loss_weight = 0.5

model = LlavaWaypointForCausalLM.from_pretrained(
    "model_zoo/uninavid-7b",
    config=config,
    torch_dtype=torch.float16,
)
```

### 4. 训练策略

建议分阶段训练：

| 阶段 | 冻结 | 训练 | 说明 |
|------|------|------|------|
| 1 | Vision + LLM | Waypoint Head | 先训练新增的 head |
| 2 | Vision | LLM + Waypoint Head | 微调 LLM |
| 3 | 无 | 全部 | 端到端微调（可选） |

```python
# 阶段 1：只训练 waypoint head
for name, param in model.named_parameters():
    if any(x in name for x in ['waypoint_head', 'query_action']):
        param.requires_grad = True
    else:
        param.requires_grad = False
```

---

## 推理流程

### 1. 加载模型

```python
from inference_waypoint import WaypointAgent

agent = WaypointAgent(
    model_path="model_zoo/uninavid-waypoint-7b",
    num_waypoints=5
)
```

### 2. 执行推理

```python
agent.reset()

result = agent.act({
    'instruction': "Go to the kitchen",
    'observations': rgb_image  # np.ndarray [H, W, 3]
})

print(result['positions'])  # [[0.5, 0.1], [1.0, 0.2], ...]
print(result['angles'])     # [0.1, 0.15, ...]
print(result['arrive'])     # [0.0, 0.0, 0.0, 0.0, 0.8]
```

### 3. 可视化

```python
from inference_waypoint import draw_waypoints_fpv

vis_image = draw_waypoints_fpv(
    rgb_image,
    result['positions'],
    result['angles'],
    result['arrive']
)
```

---

## 与 OmniNav 的对比

| 特性 | Uni-NaVid (原始) | Uni-NaVid (Waypoint) | OmniNav |
|------|-----------------|---------------------|---------|
| 输出类型 | 离散动作文本 | 连续 waypoints | 连续 waypoints |
| Head 类型 | LM Head | MLP + Cross-Attention | MLP + Cross-Attention |
| 预测数量 | 4 个动作 | N 个 waypoints | 5 个 waypoints |
| 角度表示 | 无 | (sin, cos) | (sin, cos) |
| 到达预测 | 无 | 有 | 有 |

---

## 注意事项

1. **帧率差异**：OmniNavBench 数据中不同 agent 类型帧率不同
   - human/dog: 239 FPS
   - car: 60 或 72 FPS
   - DataLoader 会自动计算并处理

2. **坐标系**：waypoints 使用机器人坐标系
   - x: 前进方向
   - y: 左右方向
   - yaw: 逆时针为正

3. **单位转换**：OmniNavBench 数据中存在不同的坐标单位
   - `units_in_meters: 1` - 坐标已经是米（约2925个文件）
   - `units_in_meters: 0.01` - 坐标是厘米（约4156个文件）
   - DataLoader 会自动读取 `units_in_meters` 字段并转换为米
   - 所有输出的 waypoint 位置统一为米

4. **预训练权重**：加载时使用 `strict=False`，新增的 waypoint_head 会随机初始化
