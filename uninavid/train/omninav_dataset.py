# -*- coding: utf-8 -*-
# OmniNavBench 数据集用于航点预测
# 该数据集从 OmniNavBench 格式加载视频帧和轨迹数据

import os
import copy
import json
import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader, cpu

import transformers
from uninavid.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    VIDEO_START_SPECIAL_TOKEN, VIDEO_END_SPECIAL_TOKEN,
    IMAGE_START_TOKEN, IMAGE_END_TOKEN, NAVIGATION_SPECIAL_TOKEN,
    IAMGE_SEPARATOR, NAVIGATION_IDENTIFIER
)
from uninavid.mm_utils import tokenizer_image_token
from uninavid import conversation as conversation_lib


@dataclass
class OmniNavDataArguments:
    """OmniNavBench 数据集的参数配置"""
    data_base_path: str = None  # JSON轨迹数据的基础路径
    video_base_path: str = None  # 视频文件的基础路径
    instruction_types: List[str] = None  # 指令类型列表: ['original', 'concise', 'verbose', 'first_person']
    agent_types: List[str] = None  # 智能体类型列表: ['human', 'car', 'dog']
    video_fps: int = 30  # 目标视频帧率
    max_frames: int = 32  # 最大采样帧数
    num_future_waypoints: int = 5  # 要预测的未来航点数量
    waypoint_stride: int = 5  # 采样未来航点的步长
    image_processor: Optional[object] = None  # 图像处理器
    mm_use_im_start_end: bool = False  # 是否使用图像开始/结束标记
    is_multimodal: bool = True  # 是否为多模态数据集


def calculate_trajectory_fps(waypoints: List[dict]) -> float:
    """
    从航点数据计算轨迹的FPS

    Args:
        waypoints: 航点列表

    Returns:
        float: 计算得到的FPS，如果数据不足则返回默认值72.0
    """
    if len(waypoints) < 2:
        return 72.0  # 默认值

    total_frames = waypoints[-1]['frame'] - waypoints[0]['frame']
    total_time = waypoints[-1]['time_s'] - waypoints[0]['time_s']

    return total_frames / total_time if total_time > 0 else 72.0


def trajectory_frame_to_video_frame(traj_frame: int, traj_fps: float, video_fps: int = 30) -> int:
    """
    将轨迹帧号转换为视频帧号

    Args:
        traj_frame: 轨迹中的帧号（基于原始高帧率，如239 FPS）
        traj_fps: 轨迹数据的FPS
        video_fps: 视频文件的FPS（默认30）

    Returns:
        int: 对应的视频帧号
    """
    fps_ratio = traj_fps / video_fps
    return int(traj_frame / fps_ratio)


def compute_relative_waypoints(
    waypoints: List[dict],
    current_idx: int,
    num_future: int = 5,
    units_in_meters: float = 1.0,
    stride: int = 5,
    goal_position: np.ndarray = None,
    success_radius: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算相对于当前位置的未来航点

    Args:
        waypoints: 航点字典列表，包含 'xyz' 和 'yaw_deg' 字段
        current_idx: 当前位置在航点列表中的索引
        num_future: 要返回的未来航点数量
        units_in_meters: 坐标转换为米的比例因子（例如 0.01 表示坐标单位是厘米）
        stride: 采样未来航点的步长（例如 stride=5 表示采样第 5, 10, 15, 20, 25 个航点）
        goal_position: 目标位置的 [2] 或 [3] 数组 (x, y) 或 (x, y, z)，单位为原始单位
        success_radius: 判定到达的距离阈值（单位：米）

    Returns:
        relative_positions: [num_future, 2] 数组，相对位置 (dx, dy)，单位米
        relative_yaws: [num_future, 2] 数组，相对朝向的 (sin, cos) 表示
        arrive_labels: [num_future] 数组，到达标签（1 表示在目标的 success_radius 范围内）
    """
    current_wp = waypoints[current_idx]
    # 应用单位转换得到米
    current_xyz = np.array(current_wp['xyz'][:2]) * units_in_meters  # 只取 x, y
    current_yaw = np.deg2rad(current_wp['yaw_deg'])

    # 如果提供了目标位置，转换为米
    goal_xy = None
    if goal_position is not None:
        goal_xy = np.array(goal_position[:2]) * units_in_meters

    # 旋转矩阵：将世界坐标转换为机器人中心坐标
    cos_yaw = np.cos(-current_yaw)
    sin_yaw = np.sin(-current_yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    relative_positions = []
    relative_yaws = []
    arrive_labels = []

    total_waypoints = len(waypoints)

    for i in range(num_future):
        future_idx = current_idx + (i + 1) * stride  # 使用步长采样

        if future_idx < total_waypoints:
            future_wp = waypoints[future_idx]
            # 应用单位转换得到米
            future_xyz = np.array(future_wp['xyz'][:2]) * units_in_meters
            future_yaw = np.deg2rad(future_wp['yaw_deg'])

            # 计算机器人中心坐标系下的相对位置
            delta_pos = future_xyz - current_xyz
            relative_pos = rotation_matrix @ delta_pos

            # 计算相对朝向
            relative_yaw = future_yaw - current_yaw
            # 归一化到 [-pi, pi]
            relative_yaw = np.arctan2(np.sin(relative_yaw), np.cos(relative_yaw))

            relative_positions.append(relative_pos)
            relative_yaws.append([np.sin(relative_yaw), np.cos(relative_yaw)])

            # 到达标签：基于到目标的距离
            if goal_xy is not None:
                dist_to_goal = np.linalg.norm(future_xyz - goal_xy)
                is_arrived = dist_to_goal < success_radius
            else:
                # 后备方案：使用最后一个航点作为到达指示
                is_arrived = (future_idx == total_waypoints - 1)
            arrive_labels.append(1.0 if is_arrived else 0.0)
        else:
            # 如果到达轨迹末尾，用零填充
            relative_positions.append([0.0, 0.0])
            relative_yaws.append([0.0, 1.0])  # sin=0, cos=1 表示无旋转
            arrive_labels.append(1.0)  # 标记为已到达（超过轨迹末尾）

    return (
        np.array(relative_positions, dtype=np.float32),
        np.array(relative_yaws, dtype=np.float32),
        np.array(arrive_labels, dtype=np.float32)
    )


class OmniNavBenchDataset(Dataset):
    """OmniNavBench 航点预测数据集"""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: OmniNavDataArguments,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        # 构建样本列表
        self.samples = self._build_sample_list()
        print(f"OmniNavBenchDataset: Loaded {len(self.samples)} samples")

    def _build_sample_list(self) -> List[dict]:
        """构建所有训练样本的列表"""
        samples = []
        missing_video_count = 0

        agent_types = self.data_args.agent_types or ['human', 'car', 'dog']
        # 如果未指定，加载所有指令类型
        instruction_types = self.data_args.instruction_types or ['original', 'concise', 'verbose', 'first_person']

        data_base = self.data_args.data_base_path
        video_base = self.data_args.video_base_path

        for instruction_type in instruction_types:
            for agent_type in agent_types:
                agent_data_path = os.path.join(data_base, instruction_type, agent_type)
                agent_video_path = os.path.join(video_base, agent_type)

                if not os.path.exists(agent_data_path):
                    print(f"Warning: {agent_data_path} does not exist, skipping...")
                    continue

                # 遍历场景
                for scene in os.listdir(agent_data_path):
                    scene_data_path = os.path.join(agent_data_path, scene)
                    scene_video_path = os.path.join(agent_video_path, scene)

                    if not os.path.isdir(scene_data_path):
                        continue

                    # 遍历回合
                    for json_file in os.listdir(scene_data_path):
                        if not json_file.endswith('.json'):
                            continue

                        json_path = os.path.join(scene_data_path, json_file)
                        episode_name = json_file.replace('.json', '')
                        video_dir = os.path.join(scene_video_path, episode_name)
                        rgb_video_path = os.path.join(video_dir, 'rgb.mp4')
                        depth_video_path = os.path.join(video_dir, 'depth.mp4')

                        # 检查视频是否存在
                        if not os.path.exists(rgb_video_path):
                            missing_video_count += 1
                            continue

                        samples.append({
                            'json_path': json_path,
                            'rgb_video_path': rgb_video_path,
                            'depth_video_path': depth_video_path,
                            'agent_type': agent_type,
                            'scene': scene,
                            'episode': episode_name,
                            'instruction_type': instruction_type,  # 跟踪指令类型
                        })

        if missing_video_count > 0:
            print(f"OmniNavBenchDataset: skipped {missing_video_count} samples without rgb.mp4")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_trajectory_data(self, json_path: str) -> dict:
        """从JSON文件加载轨迹数据"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def _get_instruction(self, data: dict) -> str:
        """从数据中提取导航指令"""
        task = data['scenarios'][0].get('task', {})
        navigation = task.get('navigation', {})
        instruction = navigation.get('instruction', '')
        return instruction

    def _get_waypoints(self, data: dict) -> List[dict]:
        """从数据中提取机器人航点"""
        robots = data['scenarios'][0].get('robots', {})
        entries = robots.get('entries', [])
        if entries:
            return entries[0].get('rb_gt_waypoints', [])
        return []

    def _get_goal_info(self, data: dict) -> Tuple[Optional[np.ndarray], float]:
        """
        从数据中提取目标位置和成功半径

        Returns:
            goal_position: [3] 数组 (x, y, z)��单位为原始单位，如果未找到则为 None
            success_radius: 成功半径（单位：米，默认 0.5）
        """
        task = data['scenarios'][0].get('task', {})
        navigation = task.get('navigation', {})

        goal_position = navigation.get('goal_position', None)
        if goal_position is not None:
            goal_position = np.array(goal_position, dtype=np.float32)

        # success_radius 通常已经是米为单位
        success_radius = navigation.get('success_radius', 0.5)

        return goal_position, success_radius

    def _sample_training_point(
        self,
        waypoints: List[dict],
        num_future: int,
        stride: int = 5
    ) -> int:
        """
        在轨迹中随机采样一个训练点
        确保有足够的未来航点（考虑步长）

        Args:
            waypoints: 航点列表
            num_future: 要预测的未来航点数量
            stride: 采样航点的步长
        """
        # 需要: current_idx + num_future * stride < len(waypoints)
        max_idx = len(waypoints) - num_future * stride - 1
        if max_idx <= 0:
            return 0
        return random.randint(0, max_idx)
    
    def _load_video_frames(
        self,
        video_path: str,
        waypoints: List[dict],
        current_idx: int,
        max_frames: int = 32
    ) -> np.ndarray:
        """
        加载从起点到当前位置的视频帧（与原始训练代码对齐）

        采样策略：
        - 加载从起点到 current_idx 的所有帧
        - 如果帧数超过 max_frames，均匀采样并保留首尾帧

        Args:
            video_path: 视频文件路径
            waypoints: 航点列表
            current_idx: 当前位置在航点列表中的索引
            max_frames: 最大加载帧数

        Returns:
            video_frames: [T, H, W, 3] numpy 数组
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_video_frames = len(vr)
        video_fps = self.data_args.video_fps

        # 计算轨迹FPS
        traj_fps = calculate_trajectory_fps(waypoints)

        # 获取从起点到���前位置的所有帧索引
        frame_indices = []
        for i in range(current_idx + 1):
            traj_frame = waypoints[i]['frame']
            video_frame = trajectory_frame_to_video_frame(traj_frame, traj_fps, video_fps)
            video_frame = min(video_frame, total_video_frames - 1)
            frame_indices.append(video_frame)

        # 去重，保持顺序
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices

        # 如果帧数过多，进行均匀采样（保留首尾帧）
        if len(frame_indices) > max_frames:
            # 保留第一帧和最后一帧，均匀采样中间帧
            step = len(frame_indices) / max_frames
            sampled_indices = [frame_indices[int(i * step)] for i in range(max_frames - 1)]
            sampled_indices.append(frame_indices[-1])  # 保留最后一帧
            frame_indices = sampled_indices

        # 加载帧
        video_frames = vr.get_batch(frame_indices).asnumpy()

        return video_frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个训练样本"""
        max_retries = 10
        for retry in range(max_retries):
            try:
                # 重试时使用不同的样本
                actual_idx = (idx + retry) % len(self.samples)
                sample = self.samples[actual_idx]

                # 加载轨迹数据
                data = self._load_trajectory_data(sample['json_path'])
                instruction = self._get_instruction(data)
                waypoints = self._get_waypoints(data)
                goal_position, success_radius = self._get_goal_info(data)

                # 获取坐标转换的单位
                units_in_meters = data['scenarios'][0]['scene'].get('units_in_meters', 1.0)

                if len(waypoints) < self.data_args.num_future_waypoints + 2:
                    if retry < max_retries - 1:
                        continue
                    raise ValueError(
                        f"Sample has insufficient waypoints ({len(waypoints)}): {sample['json_path']}"
                    )

                # 采样一个训练点
                current_idx = self._sample_training_point(
                    waypoints,
                    self.data_args.num_future_waypoints,
                    self.data_args.waypoint_stride
                )

                # 加载视频帧
                video_frames = self._load_video_frames(
                    sample['rgb_video_path'],
                    waypoints,
                    current_idx,
                    self.data_args.max_frames
                )

                # 处理视频帧
                processor = self.data_args.image_processor
                if processor is None:
                    raise ValueError("`image_processor` is required for OmniNavBenchDataset")
                video_tensor = processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

                # 计算相对航点
                relative_positions, relative_yaws, arrive_labels = compute_relative_waypoints(
                    waypoints,
                    current_idx,
                    self.data_args.num_future_waypoints,
                    units_in_meters=units_in_meters,
                    stride=self.data_args.waypoint_stride,
                    goal_position=goal_position,
                    success_radius=success_radius
                )

                # 构建对话用于分词
                conversation = [
                    {
                        "from": "human",
                        "value": f"{DEFAULT_IMAGE_TOKEN}\n{NAVIGATION_IDENTIFIER}{instruction}"
                    },
                    {
                        "from": "gpt",
                        "value": "I will navigate to the target."  # 占位符，实际输出是航点
                    }
                ]

                # 分词
                data_dict = self._preprocess_conversation(
                    [conversation],
                    has_image=True,
                    video_or_not=True
                )

                # 构建输出字典
                output = {
                    'input_ids': data_dict['input_ids'][0],
                    'labels': data_dict['labels'][0],
                    'image': video_tensor,
                    # 航点预测目标
                    'waypoint_positions': torch.from_numpy(relative_positions),  # [N, 2]
                    'waypoint_yaws': torch.from_numpy(relative_yaws),  # [N, 2] (sin, cos)
                    'waypoint_arrive': torch.from_numpy(arrive_labels),  # [N]
                    # 导航提示（模型需要用它来识别导航任务）
                    'prompt': [f"{NAVIGATION_IDENTIFIER}{instruction}"],
                }

                return output

            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Warning: Failed to load sample {actual_idx} ({sample.get('rgb_video_path', 'unknown')}): {str(e)}")
                    continue
                else:
                    raise RuntimeError(f"Failed to load sample after {max_retries} retries. Last error: {str(e)}")
    
    def _preprocess_conversation(
        self,
        sources: List[List[dict]],
        has_image: bool = False,
        video_or_not: bool = False
    ) -> Dict:
        """
        预处理对话用于分词

        Args:
            sources: 对话源列表
            has_image: 是否包含图像
            video_or_not: 是否为视频

        Returns:
            包含 input_ids 和 labels 的字典
        """
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # 应用提示模板
        conversations = []
        for source in sources:
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # 使用特殊标记进行分词
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:]
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:]
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:]
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:]
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:]
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:]

        new_list_all = []
        for prompt in conversations:
            token_prompt = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt')
            indices_to_replace = torch.where(token_prompt == -200)[0]
            new_list = []

            while indices_to_replace.numel() > 0:
                idx = indices_to_replace[0]
                if video_or_not:
                    if NAVIGATION_IDENTIFIER in prompt:
                        new_list.append(token_prompt[:idx])
                        new_list.append(video_start_special_token)
                        new_list.append(image_seperator)
                        new_list.append(token_prompt[idx:idx + 1])
                        new_list.append(video_end_special_token)
                        new_list.append(image_start_special_token)
                        new_list.append(image_end_special_token)
                        new_list.append(navigation_special_token)
                        token_prompt = token_prompt[idx + 1:]
                    else:
                        new_list.append(token_prompt[:idx])
                        new_list.append(video_start_special_token)
                        new_list.append(image_seperator)
                        new_list.append(token_prompt[idx:idx + 1])
                        new_list.append(video_end_special_token)
                        token_prompt = token_prompt[idx + 1:]
                else:
                    new_list.append(token_prompt[:idx])
                    new_list.append(image_start_special_token)
                    new_list.append(token_prompt[idx:idx + 1])
                    new_list.append(image_end_special_token)
                    token_prompt = token_prompt[idx + 1:]
                indices_to_replace = torch.where(token_prompt == -200)[0]

            if token_prompt.numel() > 0:
                new_list.append(token_prompt)
            new_list_all.append(torch.cat(new_list, dim=0))

        input_ids = torch.stack(new_list_all, dim=0)
        targets = input_ids.clone()

        # 掩码目标（只在助手回复上计算损失）
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX

            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    if NAVIGATION_IDENTIFIER in conversation and video_or_not:
                        round_len = len(tokenizer_image_token(rou, self.tokenizer)) + 6
                        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) + 6 - 2
                    elif video_or_not:
                        round_len = len(tokenizer_image_token(rou, self.tokenizer)) + 3
                        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) + 3 - 2
                    else:
                        round_len = len(tokenizer_image_token(rou, self.tokenizer)) + 2
                        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) + 2 - 2
                else:
                    round_len = len(self.tokenizer(rou).input_ids)
                    instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len

            target[cur_len:] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets)


@dataclass
class OmniNavDataCollator:
    """OmniNavBench 数据集的数据整理器"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        将多个样本整理成一个批次

        Args:
            instances: 样本列表

        Returns:
            批次字典，包含 input_ids, labels, images, waypoint_positions 等
        """
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 处理图像/视频
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # 处理航点目标
        if 'waypoint_positions' in instances[0]:
            batch['waypoint_positions'] = torch.stack(
                [instance['waypoint_positions'] for instance in instances]
            )
            batch['waypoint_yaws'] = torch.stack(
                [instance['waypoint_yaws'] for instance in instances]
            )
            batch['waypoint_arrive'] = torch.stack(
                [instance['waypoint_arrive'] for instance in instances]
            )

        # 处理提示（模型需要用它来识别导航任务）
        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        return batch


def make_omninav_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: OmniNavDataArguments
) -> Dict:
    """
    创建 OmniNavBench 训练的数据集和整理器

    Args:
        tokenizer: 预训练的分词器
        data_args: 数据集参数

    Returns:
        包含 train_dataset, eval_dataset, data_collator 的字典
    """
    train_dataset = OmniNavBenchDataset(
        tokenizer=tokenizer,
        data_args=data_args
    )
    data_collator = OmniNavDataCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
