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
    max_frames: int = field(default=16)
    image_processor: Optional[object] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    is_multimodal: bool = field(default=True)
    image_aspect_ratio: str = field(default='pad')


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
    """

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
        self.samples, self._episodes = self._load_from_directory()

        print(f"OmniNavActionDataset: Loaded {len(self.samples)} samples from {len(self._episodes)} episodes")

    def _load_from_directory(self):
        """Load all actions.json files from directory structure."""
        samples = []
        episodes = {}

        action_root = Path(self.data_args.action_root)
        split = self.data_args.split
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

                        # Load actions.json
                        try:
                            with open(actions_file) as f:
                                ep_data = json.load(f)

                            # Build episode key: inst_type/agent_type/scene/episode
                            ep_key = f"{inst_type}/{agent_type}/{scene_dir.name}/{episode_dir.name}"
                            episodes[ep_key] = ep_data

                            # Add samples
                            for sample in ep_data['samples']:
                                samples.append((ep_key, sample))

                        except Exception as e:
                            print(f"  Error loading {actions_file}: {e}")
                        continue

        return samples, episodes

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
        """
        parts = ep_key.split('/')
        # Skip inst_type (index 0), use agent_type, scene_id, episode_id
        inst_type, agent_type, scene_id, episode_id = parts
        video_root = self.data_args.video_root
        split = self.data_args.split
        return os.path.join(video_root, split, agent_type, scene_id, episode_id, 'rgb.mp4')

    def _load_video_frames(self, video_path: str, end_frame: int) -> np.ndarray:
        """Load video frames ending at end_frame."""
        max_frames = self.data_args.max_frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total_frames - 1)
        start_frame = max(0, end_frame - max_frames + 1)

        frame_indices = list(range(start_frame, end_frame + 1))

        # Apply sampling augmentation
        if len(frame_indices) > 1:
            last_frame_index = len(frame_indices) - 1
            max_drop_frames = math.ceil(0.1 * (len(frame_indices) - 1))
            num_frames_to_sample = len(frame_indices) - 1 - random.randint(0, max_drop_frames)

            if num_frames_to_sample > 0:
                sampled_indices = sorted(random.sample(range(len(frame_indices) - 1), num_frames_to_sample))
                sampled_indices.append(last_frame_index)
                sampled_indices = duplicate_with_probability(sampled_indices, 0.03)
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
        ep_key, sample = self.samples[index]
        episode = self._episodes[ep_key]

        # Load video frames
        video_path = self._get_video_path(ep_key)
        video_frame = sample['video_frame']
        video = self._load_video_frames(video_path, video_frame)

        # Process video with image processor
        processor = self.data_args.image_processor
        if processor is not None:
            image = processor.preprocess(video, return_tensors='pt')['pixel_values']
        else:
            raise ValueError("image_processor is required")

        # Get actions from preprocessed data
        actions = sample['actions']
        action_str = " ".join(actions)

        # Build conversation
        instruction = episode['instruction']

        prompt = (
            f"{DEFAULT_IMAGE_TOKEN}\n"
            f"{NAVIGATION_IDENTIFIER}"
            f"This is a navigation video. The instruction is: {instruction}\n"
            f"Based on the visual observation and instruction, determine your next four actions. "
            f"The predicted action should be one of the following: forward, left, right, or wait."
        )

        sources = [[
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": action_str}
        ]]

        data_dict = self._preprocess_conversation(sources)

        data_dict['image'] = image
        data_dict['prompt'] = [f"{NAVIGATION_IDENTIFIER}{instruction}"]

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
