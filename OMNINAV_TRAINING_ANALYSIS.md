# OmniNav 训练代码问题分析与修复计划

## 问题总览

| 问题 | 严重程度 | 影响 | 修复优先级 |
|------|----------|------|------------|
| Waypoint训练每episode只随机采样1个点 | **致命** | 数据利用率<5%，模型基本学不到东西 | P0 |
| Action生成逻辑与原始VLN不一致 | 中等 | 训练行为与原始不同，可能影响效果 | P1 |
| Stop action padding过多 | 中等 | 模型偏向预测stop，影响导航能力 | P1 |
| 滑动窗口可能的数据泄露 | 轻微 | 如果按episode划分train/val则无影响 | P2 |

---

## 修复计划

### Phase 1: 修复致命问题 (P0)

#### 1.1 Waypoint训练：改为遍历整个轨迹
- 文件：`uninavid/train/omninav_dataset.py`
- 修改：`_build_sample_list()` 和 `__getitem__()`
- 目标：每个episode产生多个样本，覆盖整个轨迹

### Phase 2: 修复中等问题 (P1)

#### 2.1 统一Action生成参数
- 文件：`scripts/preprocess_omninav_actions.py`
- 修改：TURN_ANGLE从10°改为30°，与原始VLN一致

#### 2.2 减少Stop padding
- 文件：`scripts/preprocess_omninav_actions.py`
- 修改：只添加1个stop而非4个

### Phase 3: 验证与测试 (P2)

#### 3.1 确认数据划分��式
- 确保train/val按episode划分，避免滑动窗口导致的数据泄露

---

## 详细问题分析

### 一、三种训练方式对比

| 方面 | 原始VLN训练 | 新Waypoint训练 | 新Action训练 |
|------|-------------|----------------|--------------|
| 文件 | `vln_action_text_dataset.py` | `omninav_dataset.py` | `omninav_action_dataset.py` |
| 数据源 | VLN session格式 | OmniNavBench JSON | 预处理后的actions.json |
| 采样方式 | 固定stride遍历整个轨迹 | **随机采样一个点** | 滑动窗口切分 |
| 每episode样本数 | 多个 (轨迹长度/stride) | **1个** | 多个 (窗口数) |
| 视频帧加载 | trajectory中连续关键帧 | waypoint对应的视频帧 | sample对应的关键帧 |
| 输出 | 4个离散action文本 | 连续waypoint坐标 | 4个离散action文本 |

---

### 二、Waypoint训练的核心问题（致命）

#### 问题：每个episode只采样一个点

位置：`omninav_dataset.py:421-425`
```python
# 采样一个训练点
current_idx = self._sample_training_point(
    waypoints,
    self.data_args.num_future_waypoints,
    self.data_args.waypoint_stride
)
```

位置：`omninav_dataset.py:329-333`
```python
max_idx = len(waypoints) - num_future * stride - 1
if max_idx <= 0:
    return 0
return random.randint(0, max_idx)  # 完全随机
```

**问题分析：**
- 一个有1000个waypoint的轨迹，每个epoch只训练1个随机位置
- 原始VLN训练会用stride=60遍历整个轨迹，产生约16个样本
- 数据利用率极低，模型看不到轨迹的完整信息
- 每次epoch同一个episode采样的位置完全不同，训练不稳定

#### 修复方案

```python
def _build_sample_list(self) -> List[dict]:
    """构建所有训练样本的列表 - 遍历整个轨迹"""
    samples = []
    stride = self.data_args.waypoint_stride  # 比如5
    num_future = self.data_args.num_future_waypoints

    for episode_info in self._load_episodes():
        waypoints = episode_info['waypoints']

        # 计算可采样的最大索引
        max_idx = len(waypoints) - num_future * stride - 1
        if max_idx <= 0:
            continue

        # 遍历整个轨迹，而不是只采样一个点
        sample_stride = stride * 2  # 采样间隔，可调整
        for current_idx in range(0, max_idx, sample_stride):
            samples.append({
                **episode_info,
                'current_idx': current_idx,  # 固定的采样点
            })

    return samples

def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """获取一个训练样本 - 使用预计算的current_idx"""
    sample = self.samples[idx]
    current_idx = sample['current_idx']  # 使用固定的索引，不再随机

    # ... 后续处理保持不变
```

---

### 三、Action训练的问题

#### 问题1：Action生成逻辑与原始不一致（中等）

**原始VLN** (`vln_action_text_dataset.py:57-116`):
- 输入：当前pose和未来pose（stride帧后）
- 逻辑：计算到达目标需要的turn和forward
- 特点：先转向目标方向，再前进（目��导向）
- TURN_ANGLE = 30°

**新Action预处理** (`preprocess_omninav_actions.py:61-162`):
- 输入：整个waypoint序列
- 逻辑：累积yaw变化和位移，达到阈值就生成action
- 特点：按时间顺序累积，不考虑目标方向（累积触发）
- TURN_ANGLE = 10°

**参数差异：**
| 参数 | 原始VLN | 新Action |
|------|---------|----------|
| TURN_ANGLE | 30° | 10° |
| FORWARD_DISTANCE | 0.25m | 0.25m |
| 生成逻辑 | 目标导向 | 累积触发 |

**影响示例：**
- 原始：看到目标在左边45°，生成 `left left forward forward`
- 新版：累积yaw变化10°就生成一个turn，可能生成 `left left left left forward`

#### 修复方案

修改 `preprocess_omninav_actions.py`:
```python
# 将 TURN_ANGLE 从 10° 改为 30°，与原始VLN一致
TURN_ANGLE = 30.0  # degrees per turn action (原来是10.0)
```

#### 问题2：Stop action的padding问题（中��）

位置：`preprocess_omninav_actions.py:151-160`
```python
# 4. Add final 'stop' actions (4 stops to ensure final window is all stops)
for _ in range(WINDOW_SIZE):
    action_sequence.append({
        'action': 'stop',
        ...
    })
```

**问题分析：**
- 每个轨迹末尾强制添加4个stop
- 滑动窗口会产生多个包含stop的样本
- 模型会过度学习预测stop

#### 修复方案

```python
# 只添加1个stop，而不是4个
action_sequence.append({
    'action': 'stop',
    'wp_idx': len(waypoints) - 1,
    'future_wp_idx': len(waypoints) - 1,
    'time_s': last_wp['time_s'],
    'video_frame': get_video_frame(last_wp['time_s']),
})
```

#### 问题3：滑动窗口的数据泄露（轻微）

位置：`preprocess_omninav_actions.py:198`
```python
for i in range(0, len(action_sequence) - window_size + 1, stride):
```

**问题分析：**
- window_size=4, stride=2
- 相邻样本有50%的action重叠
- 如果按sample划分train/val，会有严重的数据泄露

#### 修复方案

确保数据划分按episode进行，而非按sample。

---

## 关键代码位置索引

| 文件 | 行号 | 问题 |
|------|------|------|
| `uninavid/train/omninav_dataset.py` | 314-333 | `_sample_training_point()` 随机采样 |
| `uninavid/train/omninav_dataset.py` | 421-425 | `__getitem__()` 调用随机采样 |
| `scripts/preprocess_omninav_actions.py` | 36 | `TURN_ANGLE = 10.0` 参数不一致 |
| `scripts/preprocess_omninav_actions.py` | 151-160 | Stop padding过多 |
| `uninavid/train/vln_action_text_dataset.py` | 42 | 原始 `TURN_ANGLE = 30°` |

---

## 验证检查清单

修复后需要验证：

- [ ] Waypoint训练：每个episode产生多个样本（检查 `len(dataset)` 是否显著增加）
- [ ] Waypoint训练：同一episode的样本覆盖整个轨迹（打印current_idx分布）
- [ ] Action训练：action分布合理（stop不应该占比过高）
- [ ] 数据划分：train/val按episode划分，无数据泄露
