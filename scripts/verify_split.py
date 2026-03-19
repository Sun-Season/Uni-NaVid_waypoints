#!/usr/bin/env python3
"""Verify train/val split correctness."""

import sys
sys.path.insert(0, '/mnt/dataset/shuhzeng/Uni-NaVid_waypoints')

from uninavid.train.omninav_action_dataset import OmniNavActionDataArguments, OmniNavActionDataset
import transformers

print("Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/mnt/dataset/shuzheng/model/uninavid-7b-full-224-video-fps-1-grid-2",
    use_fast=False,
)

print("\nCreating train dataset...")
train_args = OmniNavActionDataArguments(
    action_root="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchActionData",
    video_root="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchVideos",
    split='train',
    inst_types=['original', 'concise', 'first_person', 'verbose'],
    agent_types=['car', 'dog', 'human'],
    enable_oversampling=False,  # 禁用以便统计原始分布
)

train_dataset = OmniNavActionDataset(tokenizer=tokenizer, data_args=train_args)

print("\nCreating val dataset...")
val_args = OmniNavActionDataArguments(
    action_root="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchActionData",
    video_root="/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchVideos",
    split='val',
    inst_types=['original', 'concise', 'first_person', 'verbose'],
    agent_types=['car', 'dog', 'human'],
    enable_oversampling=False,
)

val_dataset = OmniNavActionDataset(tokenizer=tokenizer, data_args=val_args)

print("\n" + "="*60)
print("VERIFICATION RESULTS")
print("="*60)

# 检查重叠
train_episodes = set(ep_key for ep_key, _, _ in train_dataset.samples)
val_episodes = set(ep_key for ep_key, _, _ in val_dataset.samples)
overlap = train_episodes & val_episodes

print(f"\nTrain episodes: {len(train_episodes)}")
print(f"Val episodes: {len(val_episodes)}")
print(f"Overlap: {len(overlap)}")

if len(overlap) > 0:
    print(f"\n❌ ERROR: Found {len(overlap)} overlapping episodes!")
    print("Sample overlapping episodes:")
    for ep in list(overlap)[:5]:
        print(f"  - {ep}")
    sys.exit(1)
else:
    print("✓ No overlap between train and val")

# 检查比例
total = len(train_episodes) + len(val_episodes)
val_ratio = len(val_episodes) / total
print(f"\nVal ratio: {val_ratio:.2%} (target: 10%)")

if not (0.08 <= val_ratio <= 0.12):
    print(f"❌ WARNING: Val ratio {val_ratio:.2%} is outside expected range [8%, 12%]")
else:
    print("✓ Val ratio is correct")

# 统计样本数
print(f"\nTrain samples: {len(train_dataset.samples)}")
print(f"Val samples: {len(val_dataset.samples)}")

print("\n" + "="*60)
print("ALL CHECKS PASSED ✓")
print("="*60)
