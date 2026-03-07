# VLN Session Dataset for Waypoint Prediction
# This dataset loads image sequences and trajectory data from VLN session format

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
class VLNSessionDataArguments:
    """Arguments for VLN Session dataset."""
    data_path: str = None  # Path to VLN session directory or list of directories
    max_frames: int = 32  # Maximum number of frames to sample
    num_future_waypoints: int = 5  # Number of future waypoints to predict
    waypoint_stride: int = 5  # Stride for sampling future waypoints
    image_processor: Optional[object] = None
    mm_use_im_start_end: bool = False
    is_multimodal: bool = True


def compute_relative_waypoints(
    frames: List[dict],
    current_idx: int,
    num_future: int = 5,
    stride: int = 5,
    goal_position: dict = None,
    success_radius: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute relative waypoints from current position.

    Args:
        frames: List of frame dicts with 'pose' field
        current_idx: Index of current position in frames
        num_future: Number of future waypoints to return
        stride: Stride for sampling future waypoints
        goal_position: Dict with 'x', 'y' keys for goal position
        success_radius: Distance threshold (in meters) to consider as arrived

    Returns:
        relative_positions: [num_future, 2] array of (dx, dy) in meters
        relative_yaws: [num_future, 2] array of (sin, cos) of relative yaw
        arrive_labels: [num_future] array of arrive labels (1 if within success_radius of goal)
    """
    current_frame = frames[current_idx]
    current_pose = current_frame['pose']
    current_x = current_pose['x']
    current_y = current_pose['y']
    current_yaw = current_pose['yaw']

    # Goal position
    goal_xy = None
    if goal_position is not None:
        goal_xy = np.array([goal_position['x'], goal_position['y']], dtype=np.float32)

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

    total_frames = len(frames)

    for i in range(num_future):
        future_idx = current_idx + (i + 1) * stride

        if future_idx < total_frames:
            future_frame = frames[future_idx]
            future_pose = future_frame['pose']
            future_x = future_pose['x']
            future_y = future_pose['y']
            future_yaw = future_pose['yaw']

            # Compute relative position in robot-centric frame
            delta_pos = np.array([future_x - current_x, future_y - current_y], dtype=np.float32)
            relative_pos = rotation_matrix @ delta_pos

            # Compute relative yaw
            relative_yaw = future_yaw - current_yaw
            # Normalize to [-pi, pi]
            relative_yaw = np.arctan2(np.sin(relative_yaw), np.cos(relative_yaw))

            relative_positions.append(relative_pos)
            relative_yaws.append([np.sin(relative_yaw), np.cos(relative_yaw)])

            # Arrive label: based on distance to goal
            if goal_xy is not None:
                future_xy = np.array([future_x, future_y], dtype=np.float32)
                dist_to_goal = np.linalg.norm(future_xy - goal_xy)
                is_arrived = dist_to_goal < success_radius
            else:
                # Fallback: use last frame as arrive indicator
                is_arrived = (future_idx == total_frames - 1)
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


class VLNSessionDataset(Dataset):
    """Dataset for VLN Session waypoint prediction."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: VLNSessionDataArguments,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        # Build sample list
        self.samples = self._build_sample_list()
        print(f"VLNSessionDataset: Loaded {len(self.samples)} samples")

    def _build_sample_list(self) -> List[dict]:
        """Build list of all training samples."""
        samples = []

        data_path = self.data_args.data_path

        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = data_path

        for path in data_paths:
            if not os.path.exists(path):
                print(f"Warning: {path} does not exist, skipping...")
                continue

            trajectory_file = os.path.join(path, 'trajectory.json')
            instructions_file = os.path.join(path, 'instructions.json')
            rgb_dir = os.path.join(path, 'rgb')

            if os.path.exists(trajectory_file) and os.path.exists(rgb_dir):
                samples.append({
                    'trajectory_file': trajectory_file,
                    'instructions_file': instructions_file,
                    'rgb_dir': rgb_dir,
                    'session_name': os.path.basename(path)
                })
            else:
                for session_name in os.listdir(path):
                    session_path = os.path.join(path, session_name)
                    if not os.path.isdir(session_path):
                        continue

                    trajectory_file = os.path.join(session_path, 'trajectory.json')
                    instructions_file = os.path.join(session_path, 'instructions.json')
                    rgb_dir = os.path.join(session_path, 'rgb')

                    if os.path.exists(trajectory_file) and os.path.exists(rgb_dir):
                        samples.append({
                            'trajectory_file': trajectory_file,
                            'instructions_file': instructions_file,
                            'rgb_dir': rgb_dir,
                            'session_name': session_name
                        })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_trajectory_data(self, trajectory_file: str) -> dict:
        """Load trajectory data from JSON file."""
        with open(trajectory_file, 'r') as f:
            data = json.load(f)
        return data

    def _load_instructions(self, instructions_file: str, task_type: str) -> str:
        """Load instruction from instructions.json."""
        if os.path.exists(instructions_file):
            with open(instructions_file, 'r') as f:
                data = json.load(f)
            # Get description_overall from instructions
            instructions_data = data.get('instructions', {})
            if isinstance(instructions_data, dict):
                description = instructions_data.get('description_overall')
                if description:
                    return description
        
        # Fallback only if file doesn't exist or no description_overall
        return 'Navigate to the target location.'

    def _sample_training_point(
        self,
        frames: List[dict],
        num_future: int,
        stride: int = 5
    ) -> int:
        """
        Sample a random point in the trajectory for training.
        Ensures there are enough future waypoints considering stride.

        Args:
            frames: List of frames
            num_future: Number of future waypoints to predict
            stride: Stride for sampling waypoints
        """
        # Need: current_idx + num_future * stride < len(frames)
        max_idx = len(frames) - num_future * stride - 1
        if max_idx <= 0:
            return 0
        return random.randint(0, max_idx)

    def _load_image_frames(
        self,
        rgb_dir: str,
        frames: List[dict],
        current_idx: int,
        max_frames: int = 32
    ) -> np.ndarray:
        """
        Load image frames up to current position.

        Args:
            rgb_dir: Directory containing RGB images
            frames: List of frames
            current_idx: Current position index in frames
            max_frames: Maximum number of frames to load

        Returns:
            image_frames: [T, H, W, 3] numpy array
        """
        # Get image files from start to current position
        image_files = []
        for i in range(current_idx + 1):
            image_file = frames[i]['image_file']
            image_path = os.path.join(rgb_dir, os.path.basename(image_file))
            if os.path.exists(image_path):
                image_files.append(image_path)

        # Sample if too many frames
        if len(image_files) > max_frames:
            # Keep first and last, sample middle
            step = len(image_files) / max_frames
            sampled_indices = [int(i * step) for i in range(max_frames - 1)]
            sampled_indices.append(len(image_files) - 1)
            image_files = [image_files[i] for i in sampled_indices]

        # Load images
        images = []
        for image_path in image_files:
            img = Image.open(image_path).convert('RGB')
            images.append(np.array(img))

        return np.stack(images, axis=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        max_retries = 10
        for retry in range(max_retries):
            try:
                # Use a different sample on retry
                actual_idx = (idx + retry) % len(self.samples)
                sample = self.samples[actual_idx]

                # Load trajectory data
                data = self._load_trajectory_data(sample['trajectory_file'])
                task_type = data.get('metadata', {}).get('task_type', 'unknown')
                frames = data['trajectory']
                
                # Load instruction from instructions.json or generate from task_type
                instruction = self._load_instructions(
                    sample.get('instructions_file', ''), 
                    task_type
                )
                goal_position = None
                success_radius = 2.0

                if len(frames) < self.data_args.num_future_waypoints * self.data_args.waypoint_stride + 2:
                    if retry < max_retries - 1:
                        continue
                    raise ValueError(
                        f"Sample has insufficient frames ({len(frames)}): {sample['trajectory_file']}"
                    )

                # Sample a training point
                current_idx = self._sample_training_point(
                    frames,
                    self.data_args.num_future_waypoints,
                    self.data_args.waypoint_stride
                )

                # Load image frames
                image_frames = self._load_image_frames(
                    sample['rgb_dir'],
                    frames,
                    current_idx,
                    self.data_args.max_frames
                )

                # Process image frames
                processor = self.data_args.image_processor
                if processor is None:
                    raise ValueError("`image_processor` is required for VLNSessionDataset")
                image_tensor = processor.preprocess(image_frames, return_tensors='pt')['pixel_values']

                # Compute relative waypoints
                relative_positions, relative_yaws, arrive_labels = compute_relative_waypoints(
                    frames,
                    current_idx,
                    self.data_args.num_future_waypoints,
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
                    'image': image_tensor,
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
                    print(f"Warning: Failed to load sample {actual_idx} ({sample.get('trajectory_file', 'unknown')}): {str(e)}")
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
class VLNSessionDataCollator:
    """Data collator for VLN Session dataset."""

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

        # Handle images
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


def make_vln_session_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: VLNSessionDataArguments
) -> Dict:
    """Create dataset and collator for VLN Session training."""
    train_dataset = VLNSessionDataset(
        tokenizer=tokenizer,
        data_args=data_args
    )
    data_collator = VLNSessionDataCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
