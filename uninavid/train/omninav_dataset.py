# OmniNavBench Dataset for Waypoint Prediction
# This dataset loads video frames and trajectory data from OmniNavBench format

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
    """Arguments for OmniNavBench dataset."""
    data_base_path: str = None  # Base path for JSON trajectory data
    video_base_path: str = None  # Base path for video files
    instruction_types: List[str] = None  # ['original', 'concise', 'verbose', 'first_person'] or None for all
    agent_types: List[str] = None  # ['human', 'car', 'dog'] or subset
    video_fps: int = 30  # Target video FPS
    max_frames: int = 32  # Maximum number of frames to sample
    num_future_waypoints: int = 5  # Number of future waypoints to predict
    waypoint_stride: int = 5  # Stride for sampling future waypoints
    image_processor: Optional[object] = None
    mm_use_im_start_end: bool = False
    is_multimodal: bool = True


def calculate_trajectory_fps(waypoints: List[dict]) -> float:
    """Calculate the FPS of trajectory data from waypoints."""
    if len(waypoints) < 2:
        return 72.0  # Default value
    
    total_frames = waypoints[-1]['frame'] - waypoints[0]['frame']
    total_time = waypoints[-1]['time_s'] - waypoints[0]['time_s']
    
    return total_frames / total_time if total_time > 0 else 72.0


def trajectory_frame_to_video_frame(traj_frame: int, traj_fps: float, video_fps: int = 30) -> int:
    """Convert trajectory frame number to video frame number."""
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
    Compute relative waypoints from current position.

    Args:
        waypoints: List of waypoint dicts with 'xyz' and 'yaw_deg' fields
        current_idx: Index of current position in waypoints
        num_future: Number of future waypoints to return
        units_in_meters: Scale factor to convert coordinates to meters
                         (e.g., 0.01 means coordinates are in cm)
        stride: Stride for sampling future waypoints (e.g., stride=5 means 5, 10, 15, 20, 25)
        goal_position: [2] or [3] array of goal position (x, y) or (x, y, z) in original units
        success_radius: Distance threshold (in meters) to consider as arrived

    Returns:
        relative_positions: [num_future, 2] array of (dx, dy) in meters
        relative_yaws: [num_future, 2] array of (sin, cos) of relative yaw
        arrive_labels: [num_future] array of arrive labels (1 if within success_radius of goal)
    """
    current_wp = waypoints[current_idx]
    # Apply unit conversion to get meters
    current_xyz = np.array(current_wp['xyz'][:2]) * units_in_meters  # Only x, y
    current_yaw = np.deg2rad(current_wp['yaw_deg'])

    # Convert goal position to meters if provided
    goal_xy = None
    if goal_position is not None:
        goal_xy = np.array(goal_position[:2]) * units_in_meters

    # Rotation matrix to convert to robot-centric coordinates
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
        future_idx = current_idx + (i + 1) * stride  # Use stride for sampling

        if future_idx < total_waypoints:
            future_wp = waypoints[future_idx]
            # Apply unit conversion to get meters
            future_xyz = np.array(future_wp['xyz'][:2]) * units_in_meters
            future_yaw = np.deg2rad(future_wp['yaw_deg'])

            # Compute relative position in robot-centric frame
            delta_pos = future_xyz - current_xyz
            relative_pos = rotation_matrix @ delta_pos

            # Compute relative yaw
            relative_yaw = future_yaw - current_yaw
            # Normalize to [-pi, pi]
            relative_yaw = np.arctan2(np.sin(relative_yaw), np.cos(relative_yaw))

            relative_positions.append(relative_pos)
            relative_yaws.append([np.sin(relative_yaw), np.cos(relative_yaw)])

            # Arrive label: based on distance to goal
            if goal_xy is not None:
                dist_to_goal = np.linalg.norm(future_xyz - goal_xy)
                is_arrived = dist_to_goal < success_radius
            else:
                # Fallback: use last waypoint as arrive indicator
                is_arrived = (future_idx == total_waypoints - 1)
            arrive_labels.append(1.0 if is_arrived else 0.0)
        else:
            # Pad with zeros if we've reached the end
            relative_positions.append([0.0, 0.0])
            relative_yaws.append([0.0, 1.0])  # sin=0, cos=1 means no rotation
            arrive_labels.append(1.0)  # Mark as arrived (past trajectory end)

    return (
        np.array(relative_positions, dtype=np.float32),
        np.array(relative_yaws, dtype=np.float32),
        np.array(arrive_labels, dtype=np.float32)
    )


class OmniNavBenchDataset(Dataset):
    """Dataset for OmniNavBench waypoint prediction."""
    
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: OmniNavDataArguments,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        # Build sample list
        self.samples = self._build_sample_list()
        print(f"OmniNavBenchDataset: Loaded {len(self.samples)} samples")
    
    def _build_sample_list(self) -> List[dict]:
        """Build list of all training samples."""
        samples = []
        missing_video_count = 0

        agent_types = self.data_args.agent_types or ['human', 'car', 'dog']
        # Load all instruction types if not specified
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

                # Iterate through scenes
                for scene in os.listdir(agent_data_path):
                    scene_data_path = os.path.join(agent_data_path, scene)
                    scene_video_path = os.path.join(agent_video_path, scene)

                    if not os.path.isdir(scene_data_path):
                        continue

                    # Iterate through episodes
                    for json_file in os.listdir(scene_data_path):
                        if not json_file.endswith('.json'):
                            continue

                        json_path = os.path.join(scene_data_path, json_file)
                        episode_name = json_file.replace('.json', '')
                        video_dir = os.path.join(scene_video_path, episode_name)
                        rgb_video_path = os.path.join(video_dir, 'rgb.mp4')
                        depth_video_path = os.path.join(video_dir, 'depth.mp4')

                        # Check if video exists
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
                            'instruction_type': instruction_type,  # Track which instruction type
                        })

        if missing_video_count > 0:
            print(f"OmniNavBenchDataset: skipped {missing_video_count} samples without rgb.mp4")

        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_trajectory_data(self, json_path: str) -> dict:
        """Load trajectory data from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _get_instruction(self, data: dict) -> str:
        """Extract navigation instruction from data."""
        task = data['scenarios'][0].get('task', {})
        navigation = task.get('navigation', {})
        instruction = navigation.get('instruction', '')
        return instruction
    
    def _get_waypoints(self, data: dict) -> List[dict]:
        """Extract robot waypoints from data."""
        robots = data['scenarios'][0].get('robots', {})
        entries = robots.get('entries', [])
        if entries:
            return entries[0].get('rb_gt_waypoints', [])
        return []

    def _get_goal_info(self, data: dict) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract goal position and success radius from data.

        Returns:
            goal_position: [3] array of (x, y, z) in original units, or None if not found
            success_radius: Success radius in meters (default 0.5)
        """
        task = data['scenarios'][0].get('task', {})
        navigation = task.get('navigation', {})

        goal_position = navigation.get('goal_position', None)
        if goal_position is not None:
            goal_position = np.array(goal_position, dtype=np.float32)

        # success_radius is typically in meters already
        success_radius = navigation.get('success_radius', 0.5)

        return goal_position, success_radius

    def _sample_training_point(
        self,
        waypoints: List[dict],
        num_future: int,
        stride: int = 5
    ) -> int:
        """
        Sample a random point in the trajectory for training.
        Ensures there are enough future waypoints considering stride.

        Args:
            waypoints: List of waypoints
            num_future: Number of future waypoints to predict
            stride: Stride for sampling waypoints
        """
        # Need: current_idx + num_future * stride < len(waypoints)
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
        Load video frames up to current position.
        
        Args:
            video_path: Path to video file
            waypoints: List of waypoints
            current_idx: Current position index in waypoints
            max_frames: Maximum number of frames to load
        
        Returns:
            video_frames: [T, H, W, 3] numpy array
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_video_frames = len(vr)
        video_fps = self.data_args.video_fps
        
        # Calculate trajectory FPS
        traj_fps = calculate_trajectory_fps(waypoints)
        
        # Get frame indices from start to current position
        frame_indices = []
        for i in range(current_idx + 1):
            traj_frame = waypoints[i]['frame']
            video_frame = trajectory_frame_to_video_frame(traj_frame, traj_fps, video_fps)
            video_frame = min(video_frame, total_video_frames - 1)
            frame_indices.append(video_frame)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices
        
        # Sample if too many frames
        if len(frame_indices) > max_frames:
            # Keep first and last, sample middle
            step = len(frame_indices) / max_frames
            sampled_indices = [frame_indices[int(i * step)] for i in range(max_frames - 1)]
            sampled_indices.append(frame_indices[-1])
            frame_indices = sampled_indices
        
        # Load frames
        video_frames = vr.get_batch(frame_indices).asnumpy()
        
        return video_frames
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        max_retries = 10
        for retry in range(max_retries):
            try:
                # Use a different sample on retry
                actual_idx = (idx + retry) % len(self.samples)
                sample = self.samples[actual_idx]

                # Load trajectory data
                data = self._load_trajectory_data(sample['json_path'])
                instruction = self._get_instruction(data)
                waypoints = self._get_waypoints(data)
                goal_position, success_radius = self._get_goal_info(data)

                # Get units_in_meters for coordinate conversion
                units_in_meters = data['scenarios'][0]['scene'].get('units_in_meters', 1.0)

                if len(waypoints) < self.data_args.num_future_waypoints + 2:
                    if retry < max_retries - 1:
                        continue
                    raise ValueError(
                        f"Sample has insufficient waypoints ({len(waypoints)}): {sample['json_path']}"
                    )

                # Sample a training point
                current_idx = self._sample_training_point(
                    waypoints,
                    self.data_args.num_future_waypoints,
                    self.data_args.waypoint_stride
                )

                # Load video frames
                video_frames = self._load_video_frames(
                    sample['rgb_video_path'],
                    waypoints,
                    current_idx,
                    self.data_args.max_frames
                )

                # Process video frames
                processor = self.data_args.image_processor
                if processor is None:
                    raise ValueError("`image_processor` is required for OmniNavBenchDataset")
                video_tensor = processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

                # Compute relative waypoints
                relative_positions, relative_yaws, arrive_labels = compute_relative_waypoints(
                    waypoints,
                    current_idx,
                    self.data_args.num_future_waypoints,
                    units_in_meters=units_in_meters,
                    stride=self.data_args.waypoint_stride,
                    goal_position=goal_position,
                    success_radius=success_radius
                )

                # Build conversation for tokenization
                conversation = [
                    {
                        "from": "human",
                        "value": f"{DEFAULT_IMAGE_TOKEN}\n{NAVIGATION_IDENTIFIER}{instruction}"
                    },
                    {
                        "from": "gpt",
                        "value": "I will navigate to the target."  # Placeholder, actual output is waypoints
                    }
                ]

                # Tokenize
                data_dict = self._preprocess_conversation(
                    [conversation],
                    has_image=True,
                    video_or_not=True
                )

                # Build output dict
                output = {
                    'input_ids': data_dict['input_ids'][0],
                    'labels': data_dict['labels'][0],
                    'image': video_tensor,
                    # Waypoint prediction targets
                    'waypoint_positions': torch.from_numpy(relative_positions),  # [N, 2]
                    'waypoint_yaws': torch.from_numpy(relative_yaws),  # [N, 2] (sin, cos)
                    'waypoint_arrive': torch.from_numpy(arrive_labels),  # [N]
                    # Prompt for navigation (required by model to identify navigation task)
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
        """Preprocess conversation for tokenization."""
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        
        # Apply prompt templates
        conversations = []
        for source in sources:
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]
            
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        
        # Tokenize with special tokens
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
        
        # Mask targets (only compute loss on assistant responses)
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
    """Data collator for OmniNavBench dataset."""
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
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
        
        # Handle images/videos
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        # Handle waypoint targets
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

        # Handle prompts (required by model to identify navigation task)
        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        return batch


def make_omninav_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: OmniNavDataArguments
) -> Dict:
    """Create dataset and collator for OmniNavBench training."""
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
