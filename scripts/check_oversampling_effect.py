#!/usr/bin/env python3
"""
检查过采样对动作分布的影响
"""

import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uninavid.train.omninav_action_dataset import OmniNavActionDataset, OmniNavActionDataArguments


def analyze_with_oversampling(data_root: str, enable_oversampling: bool):
    """分析启用/禁用过采样时的动作分布"""

    status = "启用" if enable_oversampling else "禁用"
    print(f"{'='*60}")
    print(f"{status}过采样时的训练集动作分布")
    print(f"{'='*60}\n")

    data_args = OmniNavActionDataArguments(
        action_root=data_root,
        video_root='/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchVideos',
        split='train',
        enable_oversampling=enable_oversampling,
    )

    dataset = OmniNavActionDataset(
        tokenizer=None,
        data_args=data_args
    )

    print(f"样本数: {len(dataset.samples)}\n")

    # 统计动作分布
    action_counter = Counter()

    # 统计包含left/right的样本中forward的占比
    samples_with_left_or_right = 0
    forward_in_left_right_samples = 0
    total_actions_in_left_right_samples = 0

    for ep_key, sample_idx, sample_data in dataset.samples:
        actions = sample_data.get('actions', [])

        has_left_or_right = 'left' in actions or 'right' in actions
        if has_left_or_right:
            samples_with_left_or_right += 1
            forward_in_left_right_samples += actions.count('forward')
            total_actions_in_left_right_samples += len(actions)

        for action in actions:
            action_counter[action] += 1

    # 输出结果
    total_actions = sum(action_counter.values())

    print(f"总动作数: {total_actions}\n")
    print("动作分布:")
    print(f"{'动作':<10} {'数量':>10} {'占比':>10}")
    print("-" * 32)

    for action in ['forward', 'left', 'right', 'wait', 'stop']:
        count = action_counter.get(action, 0)
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"{action:<10} {count:>10} {pct:>9.2f}%")

    print(f"\n{'='*60}")
    print("包含left/right的样本分析:")
    print(f"{'='*60}")
    print(f"包含left或right的样本数: {samples_with_left_or_right}")
    print(f"这些样本中的总动作数: {total_actions_in_left_right_samples}")
    print(f"这些样本中forward的数量: {forward_in_left_right_samples}")
    if total_actions_in_left_right_samples > 0:
        pct = forward_in_left_right_samples / total_actions_in_left_right_samples * 100
        print(f"这些样本中forward的占比: {pct:.2f}%")

    print()
    return action_counter


if __name__ == '__main__':
    data_root = '/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchActionData'

    # 禁用过采样
    print("1. 禁用过采样的情况：\n")
    counter_no_oversample = analyze_with_oversampling(data_root, enable_oversampling=False)

    print("\n\n")

    # 启用过采样
    print("2. 启用过采样的情况：\n")
    counter_with_oversample = analyze_with_oversampling(data_root, enable_oversampling=True)

    print("\n\n")
    print("="*60)
    print("对比分析")
    print("="*60)

    total_no = sum(counter_no_oversample.values())
    total_with = sum(counter_with_oversample.values())

    print(f"\n总动作数变化: {total_no} -> {total_with} (增加 {total_with - total_no})")
    print("\n各动作占比变化:")
    print(f"{'动作':<10} {'禁用过采样':>12} {'启用过采样':>12} {'变化':>10}")
    print("-" * 48)

    for action in ['forward', 'left', 'right', 'wait', 'stop']:
        pct_no = counter_no_oversample.get(action, 0) / total_no * 100
        pct_with = counter_with_oversample.get(action, 0) / total_with * 100
        change = pct_with - pct_no
        print(f"{action:<10} {pct_no:>11.2f}% {pct_with:>11.2f}% {change:>+9.2f}%")
