#!/usr/bin/env python3
"""
OmniNavBench Action Dataset for discrete action prediction via text generation.
Loads preprocessed data from OmniNavBenchActionData directory structure.

Directory structure:
  OmniNavBenchActionData/
    train/
      car/
        scene_id/
          episode_id/
            actions.json
"""

import os
import json
import random
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
import transformers
import cv2

from uninavid.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    NAVIGATION_IDENTIFIER,
)
from uninavid.mm_utils import tokenizer_image_token
from uninavid import conversation as conversation_lib


# Action labels
ACTION_LABELS = ['forward', 'left', 'right', 'wait', 'stop']

# Constants
NUM_ACTIONS = 4
VIDEO_FPS = 30


@dataclass
class OmniNavActionDataArguments:
    """Arguments for OmniNavBench Action dataset."""
    action_root: str = field(default=None)  # OmniNavBenchActionData root
    video_root: str = field(default=None)   # OmniNavBenchVideos root
    split: str = field(default='train')
    inst_types: List[str] = field(default_factory=lambda: ['original', 'concise', 'first_person', 'verbose'])
    agent_types: List[str] = field(default_factory=lambda: ['car', 'dog', 'human'])
    video_fps: int = field(default=1)  # Target fps for frame sampling (original video is 30fps)
    image_processor: Optional[object] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    is_multimodal: bool = field(default=True)
    image_aspect_ratio: str = field(default='pad')
    # 过采样配置 (控制总样本数，避免训练时间过长)
    # 策略：适度oversample，总样本数约为原始的2倍 (~28000)
    enable_oversampling: bool = field(default=True)
    oversample_pure_forward: float = field(default=0.05)  # 下采样到5% (10000->500)
    oversample_majority_left: int = field(default=8)      # 上采样8倍 (1000->8000)
    oversample_majority_right: int = field(default=8)     # 上采样8倍 (1000->8000)
    oversample_has_wait: int = field(default=10)          # 上采样10倍 (500->5000)
    oversample_has_stop: int = field(default=10)          # 上采样10倍 (500->5000)
    oversample_mixed_turn: float = field(default=1.5)     # 上采样1.5倍 (1000->1500)
    # 验证集划分配置
    val_split_ratio: float = field(default=0.1)  # 从train中划分10%作为验证集
    val_split_seed: int = field(default=42)  # 固定随机种子保证可复现
    val_split_by_episode: bool = field(default=True)  # 按episode级��划分


def duplicate_with_probability(lst, n):
    """Duplicate frames with probability."""
    result = []
    for index, num in enumerate(lst):
        result.append(num)
        if random.random() < n or (index == len(lst)-1 and random.random() < 2 * n):
            result.append(num)
    return result


def random_color_jitter(video, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                        saturation_range=(0.8, 1.2), prob=0.05):
    """Random color jitter augmentation."""
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


class OmniNavActionDataset(Dataset):
    """
    Dataset for OmniNavBench discrete action prediction.
    Loads from OmniNavBenchActionData directory structure.

    Supports oversampling to balance action distribution:
    - pure_forward: samples with all forward actions (downsampled)
    - majority_left: samples with >=2 left actions
    - majority_right: samples with >=2 right actions
    - has_wait: samples containing wait action
    - has_stop: samples containing stop action
    - mixed_turn: other samples with turns
    """

    # Sample category constants
    CAT_PURE_FORWARD = 'pure_forward'
    CAT_MAJORITY_LEFT = 'majority_left'
    CAT_MAJORITY_RIGHT = 'majority_right'
    CAT_HAS_WAIT = 'has_wait'
    CAT_HAS_STOP = 'has_stop'
    CAT_MIXED_TURN = 'mixed_turn'

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: OmniNavActionDataArguments,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        # Validate required arguments
        if not data_args.action_root:
            raise ValueError("action_root is required (path to OmniNavBenchActionData).")
        if not os.path.exists(data_args.action_root):
            raise FileNotFoundError(f"Action root not found: {data_args.action_root}")
        if not data_args.video_root:
            raise ValueError("video_root is required (path to OmniNavBenchVideos).")

        # Load all episodes from directory structure
        raw_samples, self._episodes = self._load_from_directory()
        print(f"OmniNavActionDataset: Loaded {len(raw_samples)} raw samples from {len(self._episodes)} episodes")

        # Apply oversampling if enabled
        if data_args.enable_oversampling and data_args.split == 'train':
            self.samples = self._apply_oversampling(raw_samples)
            print(f"OmniNavActionDataset: After oversampling: {len(self.samples)} samples")
            self._print_action_distribution()
        else:
            self.samples = raw_samples
            print(f"OmniNavActionDataset: Oversampling disabled, using {len(self.samples)} samples")

    def _classify_sample(self, actions: List[str]) -> str:
        """Classify a sample based on its actions.

        Categories (in priority order):
        - has_wait: contains 'wait' action
        - has_stop: contains 'stop' action (but not wait)
        - pure_forward: all actions are 'forward'
        - majority_left: contains >=2 'left' actions
        - majority_right: contains >=2 'right' actions
        - mixed_turn: other samples with some turns
        """
        has_wait = 'wait' in actions
        has_stop = 'stop' in actions
        all_forward = all(a == 'forward' for a in actions)
        left_count = actions.count('left')
        right_count = actions.count('right')

        if has_wait:
            return self.CAT_HAS_WAIT
        elif has_stop:
            return self.CAT_HAS_STOP
        elif all_forward:
            return self.CAT_PURE_FORWARD
        elif left_count >= 2:
            return self.CAT_MAJORITY_LEFT
        elif right_count >= 2:
            return self.CAT_MAJORITY_RIGHT
        else:
            return self.CAT_MIXED_TURN

    def _apply_oversampling(self, samples: List[tuple]) -> List[tuple]:
        """Apply oversampling based on sample categories.

        Oversampling strategy (控制总样本数方案):
        - pure_forward: downsample to 5% (multiply by 0.05, 10000->500)
        - majority_left: oversample 8x (1000->8000)
        - majority_right: oversample 8x (1000->8000)
        - has_wait: oversample 10x (500->5000)
        - has_stop: oversample 10x (500->5000)
        - mixed_turn: oversample 1.5x (1000->1500)

        Total samples: ~28000 (2x original), balancing training time and data balance.

        Uses random sampling with replacement to increase diversity and
        prevent overfitting to repeated samples.
        """
        # Group samples by category
        categories = {
            self.CAT_PURE_FORWARD: [],
            self.CAT_MAJORITY_LEFT: [],
            self.CAT_MAJORITY_RIGHT: [],
            self.CAT_HAS_WAIT: [],
            self.CAT_HAS_STOP: [],
            self.CAT_MIXED_TURN: [],
        }

        for sample in samples:
            ep_key, sample_idx, sample_data = sample
            actions = sample_data.get('actions', [])
            category = self._classify_sample(actions)
            categories[category].append(sample)

        # Print category statistics
        print("  Sample categories before oversampling:")
        for cat, cat_samples in categories.items():
            print(f"    {cat}: {len(cat_samples)}")

        # Get multipliers from data_args
        multipliers = {
            self.CAT_PURE_FORWARD: self.data_args.oversample_pure_forward,
            self.CAT_MAJORITY_LEFT: self.data_args.oversample_majority_left,
            self.CAT_MAJORITY_RIGHT: self.data_args.oversample_majority_right,
            self.CAT_HAS_WAIT: self.data_args.oversample_has_wait,
            self.CAT_HAS_STOP: self.data_args.oversample_has_stop,
            self.CAT_MIXED_TURN: self.data_args.oversample_mixed_turn,
        }

        # Apply sampling with random selection to increase diversity
        result = []
        for cat, mult in multipliers.items():
            cat_samples = categories[cat]
            if mult >= 1:
                # Oversample: randomly sample with replacement to increase diversity
                target_count = int(len(cat_samples) * mult)
                if target_count > 0:
                    # Use random.choices (with replacement) instead of simple repetition
                    result.extend(random.choices(cat_samples, k=target_count))
            else:
                # Downsample: randomly select subset
                n = int(len(cat_samples) * mult)
                if n > 0:
                    result.extend(random.sample(cat_samples, n))

        # Shuffle the result
        random.shuffle(result)
        return result

    def _print_action_distribution(self):
        """Print action distribution after oversampling."""
        from collections import Counter
        counter = Counter()
        for sample in self.samples:
            actions = sample[2].get('actions', [])
            for a in actions:
                counter[a] += 1

        total = sum(counter.values())
        print("  Action distribution after oversampling:")
        for action in ['forward', 'left', 'right', 'wait', 'stop']:
            count = counter.get(action, 0)
            print(f"    {action}: {count} ({count/total*100:.2f}%)")

    def _load_from_directory(self):
        """Load all actions.json files from directory structure."""
        samples = []
        episodes = {}

        action_root = Path(self.data_args.action_root)
        # 如果是val split，从train目录加载，稍后通过_split_train_val划分
        split = 'train' if self.data_args.split == 'val' else self.data_args.split
        inst_types = self.data_args.inst_types
        agent_types = self.data_args.agent_types

        split_dir = action_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for inst_type in inst_types:
            inst_dir = split_dir / inst_type
            if not inst_dir.exists():
                print(f"  Warning: Instruction type directory not found: {inst_dir}")
                continue

            for agent_type in agent_types:
                agent_dir = inst_dir / agent_type
                if not agent_dir.exists():
                    print(f"  Warning: Agent directory not found: {agent_dir}")
                    continue

                # Iterate through scenes
                for scene_dir in sorted(agent_dir.iterdir()):
                    if not scene_dir.is_dir():
                        continue

                    # Iterate through episodes
                    for episode_dir in sorted(scene_dir.iterdir()):
                        if not episode_dir.is_dir():
                            continue

                        actions_file = episode_dir / 'actions.json'
                        if not actions_file.exists():
                            continue

                        # Build episode key: inst_type/agent_type/scene/episode
                        ep_key = f"{inst_type}/{agent_type}/{scene_dir.name}/{episode_dir.name}"

                        # Check if video exists (skip inst_type for video path)
                        video_path = os.path.join(
                            self.data_args.video_root, split, agent_type,
                            scene_dir.name, episode_dir.name, 'rgb.mp4'
                        )
                        if not os.path.exists(video_path):
                            continue

                        # Check if video is valid (can be opened, has frames, and can read frames)
                        try:
                            cap = cv2.VideoCapture(video_path)
                            if not cap.isOpened():
                                print(f"  Warning: Cannot open video: {video_path}")
                                cap.release()
                                continue
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if frame_count <= 0:
                                print(f"  Warning: Video has no frames: {video_path}")
                                cap.release()
                                continue

                            # Try to read the first frame to ensure video is not corrupted
                            ret, frame = cap.read()
                            cap.release()
                            if not ret or frame is None:
                                print(f"  Warning: Cannot read frames from video: {video_path}")
                                continue
                        except Exception as e:
                            print(f"  Warning: Error checking video {video_path}: {e}")
                            continue

                        # Load actions.json
                        try:
                            with open(actions_file) as f:
                                ep_data = json.load(f)

                            episodes[ep_key] = ep_data

                            # Add samples with sample_idx for history lookup
                            for sample_idx, sample in enumerate(ep_data['samples']):
                                samples.append((ep_key, sample_idx, sample))

                        except Exception as e:
                            print(f"  Error loading {actions_file}: {e}")
                        continue

        # 如果启用episode级别划分，进行train/val split
        if self.data_args.val_split_by_episode and self.data_args.split in ['train', 'val']:
            samples, episodes = self._split_train_val(samples, episodes)

        return samples, episodes

    def _split_train_val(self, all_samples, all_episodes):
        """按episode级别划分train/val，确保同一episode不会同时出现在两个集合中。

        策略：
        1. 全局随机划分episodes到train/val
        2. 使用固定随机种子保证可复现
        """
        import random

        random.seed(self.data_args.val_split_seed)

        # 获取所有episode keys并打乱
        episode_keys = list(all_episodes.keys())
        random.shuffle(episode_keys)

        # 计算验证集大小
        n_val = int(len(episode_keys) * self.data_args.val_split_ratio)

        # 划分train/val
        val_episodes = set(episode_keys[:n_val])
        train_episodes = set(episode_keys[n_val:])

        # 根据当前split过滤
        target_episodes = train_episodes if self.data_args.split == 'train' else val_episodes

        filtered_samples = [
            (ep_key, sample_idx, sample)
            for ep_key, sample_idx, sample in all_samples
            if ep_key in target_episodes
        ]

        filtered_episodes = {
            ep_key: ep_data
            for ep_key, ep_data in all_episodes.items()
            if ep_key in target_episodes
        }

        print(f"  Split statistics:")
        print(f"    Total episodes: {len(all_episodes)}")
        print(f"    Train episodes: {len(train_episodes)}")
        print(f"    Val episodes: {len(val_episodes)}")
        print(f"    Current split ({self.data_args.split}): {len(filtered_episodes)} episodes, {len(filtered_samples)} samples")

        return filtered_samples, filtered_episodes

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def lengths(self):
        return [128] * len(self.samples)

    @property
    def modality_lengths(self):
        return [128] * len(self.samples)

    def _get_video_path(self, ep_key: str) -> str:
        """Get video path for episode.

        Note: Video path does NOT include inst_type since all instruction types
        share the same video files.
        ep_key format: inst_type/agent_type/scene_id/episode_id
        video path: video_root/split/agent_type/scene_id/episode_id/rgb.mp4

        For val split, videos are still in 'train' directory since we split
        train data into train/val.
        """
        parts = ep_key.split('/')
        # Skip inst_type (index 0), use agent_type, scene_id, episode_id
        inst_type, agent_type, scene_id, episode_id = parts
        video_root = self.data_args.video_root
        # Use 'train' for both train and val splits (val is split from train data)
        split = 'train' if self.data_args.split in ['train', 'val'] else self.data_args.split
        return os.path.join(video_root, split, agent_type, scene_id, episode_id, 'rgb.mp4')

    def _load_video_frames(self, video_path: str, ep_key: str, sample_idx: int) -> np.ndarray:
        """Load video frames as prefix up to current observation.

        Each sample has a video_frame field indicating the current observation frame.
        We load frames from [0, current_frame] as the prefix video:
        - Frames before current_frame: historical observations
        - current_frame (last frame): current observation

        Process:
        1. Sample frames from [0, current_frame] at video_fps interval
        2. Ensure current_frame is always included as the last frame
        3. Random frame dropping (up to 10%, always keep last frame)
        4. Random frame duplication (3% probability)
        5. Color jitter augmentation

        No max_frames truncation - model-side compression handles long sequences.
        """
        episode = self._episodes[ep_key]
        all_samples = episode['samples']
        current_sample = all_samples[sample_idx]

        # Get the current sample's video_frame as the end point (current observation)
        current_frame = current_sample['video_frame']

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps_original = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS  # Original video fps (usually 30)
        current_frame = min(current_frame, total_frames - 1)

        # Calculate sample interval: sample at target fps from original fps
        # e.g., original 30fps, target 1fps -> sample_interval = 30
        sample_interval = round(video_fps_original / self.data_args.video_fps)
        sample_interval = max(1, sample_interval)

        # Sample frames at interval from [0, current_frame]
        frame_indices = [i for i in range(0, current_frame + 1, sample_interval)]

        # Ensure current_frame is always included as the last frame (current observation)
        if len(frame_indices) == 0:
            frame_indices = [current_frame]
        elif frame_indices[-1] != current_frame:
            frame_indices.append(current_frame)

        # Apply augmentation only during training and if we have more than 1 frame
        if self.data_args.split == 'train' and len(frame_indices) > 1:
            last_frame_index = len(frame_indices) - 1

            # Random frame dropping: drop up to 10% of frames (except last frame)
            max_drop_frames = math.ceil(0.1 * (len(frame_indices) - 1))
            num_frames_to_keep = len(frame_indices) - 1 - random.randint(0, max_drop_frames)

            if num_frames_to_keep > 0:
                # Sample indices to keep (excluding last frame which is always kept)
                sampled_indices = sorted(random.sample(range(len(frame_indices) - 1), num_frames_to_keep))
                sampled_indices.append(last_frame_index)

                # Random frame duplication (3% probability)
                sampled_indices = duplicate_with_probability(sampled_indices, 0.03)

                # Map back to actual frame indices
                frame_indices = [frame_indices[i] for i in sampled_indices if i < len(frame_indices)]

        # Read frames
        images = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(frame)
            elif images:
                # Fallback: use last valid frame
                images.append(images[-1].copy())

        cap.release()

        if len(images) == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        video = np.stack(images)

        # Apply color jitter during training
        if self.data_args.split == 'train':
            video = random_color_jitter(video)

        return video

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
        ep_key, sample_idx, sample = self.samples[index]
        episode = self._episodes[ep_key]

        # Load video frames from previous samples
        video_path = self._get_video_path(ep_key)
        video = self._load_video_frames(video_path, ep_key, sample_idx)

        # Process video with image processor
        processor = self.data_args.image_processor
        if processor is not None:
            image = processor.preprocess(video, return_tensors='pt')['pixel_values']
        else:
            raise ValueError("image_processor is required")

        # Get actions from preprocessed data
        actions = sample['actions']
        action_str = " ".join(actions)

        # Build conversation (using inference-style long prompt for consistency)
        instruction = episode['instruction']

        # 使用和推理时一致的长 prompt 格式
        prompt_template = (
            "This is a navigation video. The instruction is: {}\n"
            "Based on the visual observation and instruction, determine your next four actions. "
            "The predicted action should be one of the following: forward, left, right, wait, or stop."
        )
        question = prompt_template.format(instruction)
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}"

        sources = [[
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": action_str}
        ]]

        data_dict = self._preprocess_conversation(sources)

        data_dict['image'] = image
        data_dict['prompt'] = [question]

        return data_dict

    def _preprocess_conversation(self, sources: List[List[Dict]]) -> Dict:
        """Preprocess conversation for training."""
        conv = conversation_lib.conv_templates["imgsp_v1"].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conversations = []
        for source in sources:
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"Role mismatch: {role} vs {conv.roles[j % 2]}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize
        input_ids = torch.stack([
            tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt')
            for prompt in conversations
        ], dim=0)

        targets = input_ids.clone()

        # Mask human turns (only train on assistant response)
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            rounds = conversation.split(conv.sep2)
            cur_len = 1  # BOS token
            target[:cur_len] = IGNORE_INDEX

            for rou in rounds:
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                round_len = len(tokenizer_image_token(rou, self.tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2

                target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len

            target[cur_len:] = IGNORE_INDEX

        return dict(
            input_ids=input_ids[0],
            labels=targets[0],
        )
