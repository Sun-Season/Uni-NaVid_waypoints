#!/usr/bin/env python3
"""
VLN Action Text Dataset for discrete action prediction via text generation.
Follows the original Uni-NaVid training format - predicts actions as text output.
"""

import os
import json
import copy
import random
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Sequence

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import transformers

from uninavid.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    VIDEO_START_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN,
    IAMGE_SEPARATOR,
    NAVIGATION_IDENTIFIER,
    NAVIGATION_SPECIAL_TOKEN,
)
from uninavid.mm_utils import tokenizer_image_token
from uninavid import conversation as conversation_lib


# Action labels
ACTION_LABELS = ['forward', 'left', 'right', 'wait']

# Waypoint to action conversion parameters (from Uni-NaVid config)
FORWARD_DISTANCE = 0.25  # meters per forward action
TURN_ANGLE = np.deg2rad(30)  # 30 degrees per turn action
MIN_DISPLACEMENT = 0.05  # minimum displacement threshold
NUM_ACTIONS = 4  # fixed number of actions to predict
DEFAULT_STRIDE = 60  # default frame stride for waypoint sampling


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def waypoint_to_actions(current_pose: dict, future_pose: dict) -> List[str]:
    """
    Convert waypoint (pose difference) to discrete action sequence.

    Args:
        current_pose: dict with 'x', 'y', 'yaw'
        future_pose: dict with 'x', 'y', 'yaw'

    Returns:
        List of 4 actions: ['forward', 'left', 'right', 'wait']
    """
    dx = future_pose['x'] - current_pose['x']
    dy = future_pose['y'] - current_pose['y']
    r = math.sqrt(dx*dx + dy*dy)
    yaw_diff = normalize_angle(future_pose['yaw'] - current_pose['yaw'])

    actions = []

    if r >= MIN_DISPLACEMENT:
        # Case 1: Significant displacement - compute actions to reach target
        global_theta = math.atan2(dy, dx)
        relative_theta = normalize_angle(global_theta - current_pose['yaw'])

        # First turn towards target
        while abs(relative_theta) > TURN_ANGLE / 2:
            if relative_theta > 0:
                actions.append('left')
                relative_theta -= TURN_ANGLE
            else:
                actions.append('right')
                relative_theta += TURN_ANGLE

        # Then move forward
        remaining_dist = r
        while remaining_dist > FORWARD_DISTANCE / 2:
            actions.append('forward')
            remaining_dist -= FORWARD_DISTANCE
    else:
        # Case 2: Small displacement - check for in-place rotation
        while abs(yaw_diff) > TURN_ANGLE / 2:
            if yaw_diff > 0:
                actions.append('left')
                yaw_diff -= TURN_ANGLE
            else:
                actions.append('right')
                yaw_diff += TURN_ANGLE

    # Handle action count
    if len(actions) == 0:
        # Stationary - output wait actions
        actions = ['wait'] * NUM_ACTIONS
    else:
        # Truncate if too many
        if len(actions) > NUM_ACTIONS:
            actions = actions[:NUM_ACTIONS]
        # Repeat last action if not enough
        while len(actions) < NUM_ACTIONS:
            actions.append(actions[-1])

    return actions


@dataclass
class VLNActionTextDataArguments:
    """Arguments for VLN Action Text dataset."""
    data_path: str = field(default=None)
    max_frames: int = field(default=16)
    image_processor: Optional[object] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    is_multimodal: bool = field(default=True)
    # Sampling
    sample_stride: int = field(default=DEFAULT_STRIDE)  # Frame stride for waypoint sampling
    min_history_frames: int = field(default=4)
    # Future actions (kept for compatibility but not used in waypoint mode)
    num_future_actions: int = field(default=4)
    action_stride: int = field(default=5)
    # Data augmentation
    video_fps: int = field(default=1)
    image_aspect_ratio: str = field(default='pad')


def duplicate_with_probability(lst, n):
    """Duplicate frames with probability (from original train.py)."""
    result = []
    for index, num in enumerate(lst):
        result.append(num)
        if random.random() < n or (index == len(lst)-1 and random.random() < 2 * n):
            result.append(num)
    return result


def random_color_jitter(video, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                        saturation_range=(0.8, 1.2), prob=0.05):
    """Random color jitter augmentation (from original train.py)."""
    def adjust_brightness(image, factor):
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def adjust_contrast(image, factor):
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    def adjust_saturation(image, factor):
        grayscale = np.mean(image, axis=2, keepdims=True)
        return np.clip((image - grayscale) * factor + grayscale, 0, 255).astype(np.uint8)

    n, H, W, C = video.shape
    augmented_video = np.copy(video)

    for i in range(n):
        if np.random.rand() < prob:
            brightness_factor = np.random.uniform(*brightness_range)
            augmented_video[i] = adjust_brightness(augmented_video[i], brightness_factor)

        if np.random.rand() < prob:
            contrast_factor = np.random.uniform(*contrast_range)
            augmented_video[i] = adjust_contrast(augmented_video[i], contrast_factor)

        if np.random.rand() < prob:
            saturation_factor = np.random.uniform(*saturation_range)
            augmented_video[i] = adjust_saturation(augmented_video[i], saturation_factor)

    return augmented_video


class VLNActionTextDataset(Dataset):
    """
    Dataset for VLN discrete action prediction via text generation.
    Follows the original Uni-NaVid training format.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: VLNActionTextDataArguments,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        # Build sample list
        self.samples = self._build_sample_list()
        print(f"VLNActionTextDataset: Loaded {len(self.samples)} samples from {len(self._sessions)} sessions")
        print(f"  Using waypoint-to-action conversion with action_stride={data_args.action_stride}")

    def _build_sample_list(self) -> List[tuple]:
        """Build list of (session_path, frame_idx) samples."""
        samples = []
        self._sessions = {}

        data_path = self.data_args.data_path
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")

        # Find all session directories
        session_dirs = []
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                traj_file = os.path.join(item_path, 'trajectory.json')
                rgb_dir = os.path.join(item_path, 'rgb')
                if os.path.exists(traj_file) and os.path.exists(rgb_dir):
                    session_dirs.append(item_path)

        # Build samples from each session
        stride = self.data_args.sample_stride
        min_history = self.data_args.min_history_frames

        for session_path in sorted(session_dirs):
            # Load trajectory
            with open(os.path.join(session_path, 'trajectory.json')) as f:
                traj_data = json.load(f)

            trajectory = traj_data['trajectory']
            num_frames = len(trajectory)

            # Load instruction
            inst_file = os.path.join(session_path, 'instructions.json')
            instruction = ""
            if os.path.exists(inst_file):
                with open(inst_file) as f:
                    inst_data = json.load(f)
                if 'instructions' in inst_data and 'description_overall' in inst_data['instructions']:
                    instruction = inst_data['instructions']['description_overall']

            # Cache session data
            self._sessions[session_path] = {
                'trajectory': trajectory,
                'instruction': instruction,
            }

            # Create samples starting from min_history
            for frame_idx in range(min_history, num_frames, stride):
                samples.append((session_path, frame_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def lengths(self):
        """For group_by_modality_length."""
        return [128] * len(self.samples)

    @property
    def modality_lengths(self):
        """For group_by_modality_length."""
        return [128] * len(self.samples)

    def _load_video_frames(self, session_path: str, end_idx: int) -> np.ndarray:
        """Load video frames from session."""
        session = self._sessions[session_path]
        trajectory = session['trajectory']

        max_frames = self.data_args.max_frames
        start_idx = max(0, end_idx - max_frames + 1)
        frame_indices = list(range(start_idx, end_idx + 1))

        # Apply NAV_ID style sampling (from original train.py)
        if len(frame_indices) > 1:
            last_frame_index = len(frame_indices) - 1
            max_drop_frames = math.ceil(0.1 * (len(frame_indices) - 1))
            num_frames_to_sample = len(frame_indices) - 1 - random.randint(0, max_drop_frames)

            if num_frames_to_sample > 0:
                sampled_indices = sorted(random.sample(range(len(frame_indices) - 1), num_frames_to_sample))
                sampled_indices.append(last_frame_index)
                sampled_indices = duplicate_with_probability(sampled_indices, 0.03)
                frame_indices = [frame_indices[i] for i in sampled_indices if i < len(frame_indices)]

        # Load images
        images = []
        for idx in frame_indices:
            img_file = trajectory[idx]['image_file']
            img_path = os.path.join(session_path, img_file)
            img = Image.open(img_path).convert('RGB')
            images.append(np.array(img))

        video = np.stack(images)

        # Apply color jitter augmentation
        video = random_color_jitter(video)

        return video

    def _get_future_actions(self, session_path: str, frame_idx: int) -> List[str]:
        """Get future action labels using waypoint-to-action conversion."""
        session = self._sessions[session_path]
        trajectory = session['trajectory']
        num_frames = len(trajectory)

        # Get current and future poses
        current_pose = trajectory[frame_idx]['pose']

        # Use action_stride to determine future frame (NOT sample_stride!)
        stride = self.data_args.action_stride
        future_idx = min(frame_idx + stride, num_frames - 1)
        future_pose = trajectory[future_idx]['pose']

        # Convert waypoint to actions
        actions = waypoint_to_actions(current_pose, future_pose)

        return actions

    def __getitem__(self, index: int) -> Dict:
        """Get a single sample."""
        max_retries = 10

        for retry in range(max_retries):
            try:
                return self._get_item_impl(index)
            except Exception as e:
                if retry < max_retries - 1:
                    index = random.randint(0, len(self.samples) - 1)
                else:
                    raise e

    def _get_item_impl(self, index: int) -> Dict:
        """Implementation of __getitem__."""
        session_path, frame_idx = self.samples[index]
        session = self._sessions[session_path]

        # Load video frames
        video = self._load_video_frames(session_path, frame_idx)

        # Process video with image processor
        processor = self.data_args.image_processor
        if processor is not None:
            image = processor.preprocess(video, return_tensors='pt')['pixel_values']
        else:
            raise ValueError("image_processor is required")

        # Get future actions
        actions = self._get_future_actions(session_path, frame_idx)
        action_str = " ".join(actions)  # e.g., "forward left right stop"

        # Build conversation (following original Uni-NaVid format)
        instruction = session['instruction']

        # Prompt template similar to original Uni-NaVid (from agent_uninavid.py)
        prompt = (
            f"{DEFAULT_IMAGE_TOKEN}\n"
            f"{NAVIGATION_IDENTIFIER}"
            f"This is a navigation video. The instruction is: {instruction}\n"
            f"Based on the visual observation and instruction, determine your next four actions. "
            f"The predicted action should be one of the following: forward, left, right, or wait."
        )

        # Build conversation
        sources = [[
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": action_str}
        ]]

        # Preprocess conversation (following original train.py)
        data_dict = self._preprocess_conversation(sources, video_or_not=True)

        data_dict['image'] = image
        data_dict['prompt'] = [f"{NAVIGATION_IDENTIFIER}{instruction}"]

        return data_dict

    def _preprocess_conversation(
        self,
        sources: List[List[Dict]],
        video_or_not: bool = True,
    ) -> Dict:
        """Preprocess conversation following original train.py logic (preprocess_imgsp_v1)."""
        conv = conversation_lib.conv_templates["imgsp_v1"].copy()
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

        # Tokenize with special video tokens (following original train.py)
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:]
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:]
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:]
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:]
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:]
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:]

        new_list_all = []
        for prompt in conversations:
            token_prompt = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt')
            indices_to_replace = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
            new_list = []

            while indices_to_replace.numel() > 0:
                idx = indices_to_replace[0]
                if video_or_not:
                    # Navigation video format (from original train.py line 742-744)
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
                indices_to_replace = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]

            if token_prompt.numel() > 0:
                new_list.append(token_prompt)
            new_list_all.append(torch.cat(new_list, dim=0))

        input_ids = new_list_all[0]  # Single sample

        # Create labels (mask user input)
        labels = input_ids.clone()

        # Mask targets following original train.py logic (line 714-765)
        conversation = conversations[0]
        sep = conv.sep + conv.roles[1] + ": "

        total_len = int(labels.ne(self.tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        labels[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            # Navigation video (from original train.py line 742-744)
            if NAVIGATION_IDENTIFIER in conversation and video_or_not:
                round_len = len(tokenizer_image_token(rou, self.tokenizer)) + 6
                instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) + 6 - 2
            elif video_or_not:
                round_len = len(tokenizer_image_token(rou, self.tokenizer)) + 3
                instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) + 3 - 2
            else:
                round_len = len(tokenizer_image_token(rou, self.tokenizer)) + 2
                instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) + 2 - 2

            labels[cur_len:cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        labels[cur_len:] = IGNORE_INDEX

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


@dataclass
class VLNActionTextDataCollator:
    """Data collator for VLN Action Text dataset (following original train.py)."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst['input_ids'] for inst in instances]
        labels = [inst['labels'] for inst in instances]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # Truncate to model max length
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        # Attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Stack images
        images = [inst['image'] for inst in instances]
        if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
            images = torch.stack(images)

        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'images': images,
        }

        # Prompts (required by model)
        if 'prompt' in instances[0]:
            batch['prompts'] = [inst['prompt'] for inst in instances]

        return batch


def make_vln_action_text_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: VLNActionTextDataArguments,
) -> Dict:
    """Create data module for VLN Action Text training."""
    dataset = VLNActionTextDataset(tokenizer=tokenizer, data_args=data_args)
    collator = VLNActionTextDataCollator(tokenizer=tokenizer)

    return {
        'train_dataset': dataset,
        'eval_dataset': None,
        'data_collator': collator,
    }
